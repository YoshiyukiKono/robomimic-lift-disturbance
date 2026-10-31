"""Run a grid of pretrained Lift trials and save real simulation frames to MP4."""

import argparse
import csv
from contextlib import closing
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

from .checkpoint import DEFAULT_PATH
from .metrics import HoldMetric
from .disturbance import FORCE_FIELDS, validate_disturbance, pulse_steps, force_for_step

EPISODE_FIELDS = ["point_id", "x_m", "y_m", "trial", "seed", "success", "env_success",
                  "elapsed_s", "time_to_success_s", "max_height_m", "video_path", "steps", "max_hold_s"]


def validate_config(config):
    required = {"grid_x", "grid_y", "repeats", "seed", "horizon", "lift_height_m", "hold_seconds",
                "width", "height", "camera"}
    if "disturbance" in config:
        validate_disturbance(config["disturbance"])
    if set(config) - {"disturbance"} != required:
        raise ValueError(f"Config keys differ: missing={required-set(config)}, unknown={set(config)-required}")
    for key in ("grid_x", "grid_y"):
        values = config[key]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"{key} needs a nonempty list of distinct positions")
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or abs(v) > 0.25 for v in values):
            raise ValueError(f"{key} must contain finite positions within +/-0.25m of table centre")
    for key in ("repeats", "horizon", "width", "height"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or not (0 <= config["seed"] < 2**32-config["repeats"]):
        raise ValueError("Invalid random seed")
    if config["width"] % 16 or config["height"] % 16:
        raise ValueError("Video dimensions must be multiples of 16 to avoid implicit resizing")
    for key in ("hold_seconds", "lift_height_m"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not isinstance(config["camera"], str) or not config["camera"]:
        raise ValueError("camera must be a nonempty camera name")
    return config


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def annotate(frame, title, elapsed_s, height_m, metric, finished=False):
    from PIL import Image, ImageDraw
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 50), fill=(15, 23, 42))
    status = "SUCCESS" if metric.success else ("TIMEOUT" if finished else "RUNNING")
    draw.text((10, 6), title, fill="white")
    draw.text((10, 24), f"t={elapsed_s:.2f}s  lift={height_m:+.3f}m  hold={metric.consecutive_steps/metric.control_freq:.2f}s  {status}",
              fill=(116, 240, 180) if metric.success else (240, 240, 240))
    return __import__("numpy").asarray(image)


def run(config_path, output, checkpoint, device):
    config = validate_config(json.loads(Path(config_path).read_text(encoding="utf-8")))
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}. Choose a new --output; existing results are never overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("videos", "frames", "episodes"):
        (output / directory).mkdir(exist_ok=True)
    from .backend import LiftBackend
    import imageio.v2 as imageio
    import torch
    from PIL import Image

    backend = None
    started = time.monotonic()
    metadata = {"schema_version": 1, "recording_contract": "lift-tutorial-v1", "status": "starting", "config": config,
                "started_at": datetime.now(timezone.utc).isoformat(), "episodes_completed": 0}
    write_json(output / "run.json", metadata)
    try:
        backend = LiftBackend(checkpoint, device, config["camera"], config["width"], config["height"])
        frequency = backend.control_freq
        disturbance = config.get("disturbance")
        if disturbance is not None:
            pulse_steps(disturbance, frequency, config["horizon"])
        if config["hold_seconds"] < 1 / frequency:
            raise ValueError("hold_seconds must be at least one control interval")
        if config["hold_seconds"] > config["horizon"] / frequency:
            raise ValueError("hold_seconds exceeds episode duration")
        metadata["config"].update(control_freq=frequency, video_fps=frequency)
        versions = {}
        for name in ("torch", "torchvision", "robomimic", "robosuite", "mujoco", "numpy", "imageio", "imageio-ffmpeg"):
            versions[name] = importlib.metadata.version(name)
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = None
        metadata.update(status="running", checkpoint=backend.checkpoint_metadata, environment={
            "python": platform.python_version(), "platform": platform.platform(), "versions": versions,
            "device": str(backend.device), "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "mujoco_gl": os.environ.get("MUJOCO_GL"), "git_commit": git_commit,
            "original_checkpoint_environment": backend.original_environment,
            "effective_environment_kwargs": backend.environment_kwargs,
        })
        write_json(output / "run.json", metadata)
        total = len(config["grid_x"]) * len(config["grid_y"]) * config["repeats"]
        with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as csv_stream:
            writer = csv.DictWriter(csv_stream, fieldnames=EPISODE_FIELDS)
            writer.writeheader()
            csv_stream.flush()
            for iy, y in enumerate(config["grid_y"]):
                for ix, x in enumerate(config["grid_x"]):
                    for trial in range(config["repeats"]):
                        point_id = f"x{ix:02d}_y{iy:02d}"
                        episode_id = f"{point_id}_trial{trial:02d}"
                        seed = config["seed"] + trial  # matched seeds across positions
                        observation, initial = backend.reset(seed, x, y)
                        write_json(output / "episodes" / f"{episode_id}_initial.json", initial)
                        initial_z = initial["cube_position_world_m"][2]
                        metric = HoldMetric(initial_z, config["lift_height_m"], config["hold_seconds"], frequency)
                        first_frame = backend.frame()
                        Image.fromarray(first_frame).save(output / "frames" / f"{episode_id}_start.png")
                        video_relative = Path("videos") / f"{episode_id}.mp4"
                        title = f"XY=({x:+.2f},{y:+.2f})m  seed={seed}  sim-time 1x"
                        ever_env_success = False
                        trajectory_fields = ["step", "sim_time_s", "cube_x_m", "cube_y_m", "cube_z_m", "lift_height_m",
                                             "hold_s", "success", "env_success"] + [f"action_{i}" for i in range(7)]
                        if disturbance is not None:
                            trajectory_fields += FORCE_FIELDS
                        # Encode outside the watched output tree. Publish only a closed MP4;
                        # the CSV row below is its commit marker if a copy is interrupted.
                        with tempfile.TemporaryDirectory(prefix="lift-video-") as encoder_dir, \
                                (output / "episodes" / f"{episode_id}.csv").open("w", newline="", encoding="utf-8") as trajectory:
                            video_temporary = Path(encoder_dir) / "episode.mp4"
                            trace = csv.DictWriter(trajectory, fieldnames=trajectory_fields)
                            trace.writeheader()
                            # One initial frame + one per completed control interval.
                            with closing(imageio.get_writer(str(video_temporary), fps=frequency, codec="libx264",
                                                    quality=8, pixelformat="yuv420p", macro_block_size=16)) as video:
                                video.append_data(annotate(first_frame, title, 0, 0, metric))
                                for step in range(1, config["horizon"] + 1):
                                    force = force_for_step(disturbance, step, frequency, config["horizon"])
                                    backend.set_cube_force(force)
                                    try:
                                        observation, action, env_success = backend.step(observation)
                                    finally:
                                        backend.set_cube_force([0, 0, 0])
                                    force_record = dict(zip(FORCE_FIELDS, force)) if disturbance is not None else {}
                                    cube = backend.cube_position()
                                    metric.update(cube[2])
                                    ever_env_success = ever_env_success or env_success
                                    relative_height = float(cube[2] - initial_z)
                                    trace.writerow(dict(step=step, sim_time_s=step/frequency, cube_x_m=cube[0],
                                                        cube_y_m=cube[1], cube_z_m=cube[2], lift_height_m=relative_height,
                                                        hold_s=metric.consecutive_steps/frequency, success=int(metric.success),
                                                        env_success=int(env_success),
                                                        **{f"action_{i}": value for i, value in enumerate(action)}, **force_record))
                                    frame = backend.frame()
                                    video_title = title + (f"  PUSH ({force[0]:+.1f},{force[1]:+.1f}) N" if any(force) else "")
                                    video.append_data(annotate(frame, video_title, step/frequency, relative_height, metric,
                                                              finished=step == config["horizon"]))
                                    if metric.success:
                                        break  # task is complete; do not keep lifting indefinitely
                            shutil.copyfile(video_temporary, output / video_relative)
                        Image.fromarray(frame).save(output / "frames" / f"{episode_id}_end.png")
                        result = dict(point_id=point_id, x_m=x, y_m=y, trial=trial, seed=seed,
                                      success=int(metric.success), env_success=int(ever_env_success),
                                      elapsed_s=metric.steps/frequency,
                                      time_to_success_s=metric.time_to_success_s if metric.success else "",
                                      max_height_m=metric.max_height_m, video_path=video_relative.as_posix(),
                                      steps=metric.steps, max_hold_s=metric.max_hold_s)
                        writer.writerow(result)
                        csv_stream.flush()
                        metadata["episodes_completed"] += 1
                        write_json(output / "run.json", metadata)
                        print(f"[{metadata['episodes_completed']}/{total}] {episode_id}: success={metric.success}, "
                              f"max_lift={metric.max_height_m:.3f}m, max_hold={metric.max_hold_s:.2f}s -> {video_relative}", flush=True)
        metadata["status"] = "completed"
        metadata["wall_seconds"] = time.monotonic() - started
        write_json(output / "run.json", metadata)
        print(f"Recorded {total} episodes. Next: python -m lift_tutorial.report --run-dir {output}")
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}", wall_seconds=time.monotonic()-started)
        write_json(output / "run.json", metadata)
        raise
    finally:
        if backend is not None:
            backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/disturbance_baseline.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    run(args.config, args.output, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
