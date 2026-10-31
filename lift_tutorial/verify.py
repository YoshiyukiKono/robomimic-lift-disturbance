"""Validate recorded observations and decode every saved MP4 frame."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from .report import finite_float, load_inputs, validate_rows
from .disturbance import FORCE_FIELDS, force_for_step, pulse_steps


def video_file(run_dir: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"動画パスは実験フォルダ内の相対パスである必要があります: {relative_path}")
    target = (run_dir / relative).resolve()
    try:
        target.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"実験フォルダの外にある動画: {relative_path}") from exc
    if target.suffix.lower() != ".mp4" or not target.is_file() or target.stat().st_size == 0:
        raise ValueError(f"MP4が存在しないか空です: {relative_path}")
    return target


def inspect_video(path: Path) -> dict[str, Any]:
    """Decode all frames, including the last frame; do not trust headers alone."""
    import imageio.v2 as imageio
    import imageio_ffmpeg
    count = 0
    shape = None
    try:
        with imageio.get_reader(str(path), format="ffmpeg") as reader:
            metadata = reader.get_meta_data()
            fps = finite_float(metadata.get("fps"), f"{path.name}.fps")
            if fps <= 0:
                raise ValueError("fps は正の値である必要があります")
            for frame in reader:
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    raise ValueError(f"RGB画像ではないフレーム: {path}")
                if shape is not None and frame.shape != shape:
                    raise ValueError(f"動画内で画像サイズが変化しています: {path}")
                shape = frame.shape
                count += 1
    except Exception as exc:
        raise ValueError(f"動画の全フレームを読み込めません: {path.name}: {exc}") from exc
    if count == 0:
        raise ValueError(f"動画にフレームがありません: {path}")
    # ffmpeg's reader can recover from some damaged frames. Explicit strict
    # decoding catches that case instead of certifying a recovered stream.
    try:
        strict = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-xerror", "-err_detect", "explode",
             "-i", str(path), "-f", "null", "-"],
            capture_output=True, text=True, timeout=300, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"動画の厳密デコードを完了できません: {path.name}: {exc}") from exc
    if strict.returncode or strict.stderr.strip():
        raise ValueError(f"動画の厳密デコードに失敗: {path.name}: {strict.stderr[-1500:]}")
    return {"frames": count, "fps": fps, "duration_s": count / fps,
            "width": shape[1], "height": shape[0], "size_bytes": path.stat().st_size}


def check_close(actual: Any, expected: float, label: str, *, tolerance: float = 1e-7) -> None:
    value = finite_float(actual, label)
    if not math.isclose(value, expected, abs_tol=tolerance, rel_tol=1e-7):
        raise ValueError(f"{label}: 記録値 {value} が再計算値 {expected} と一致しません")


def replay_trajectory(run_dir: Path, metadata: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Independently rebuild the height/holding metric from the saved positions."""
    from PIL import Image

    config = metadata["config"]
    episode_id = f"{row['point_id']}_trial{row['trial']:02d}"
    expected_video = f"videos/{episode_id}.mp4"
    if row["video_path"] != expected_video:
        raise ValueError(f"動画名が試行IDと一致しません: {row['video_path']}, expected {expected_video}")
    initial = json.loads((run_dir / "episodes" / f"{episode_id}_initial.json").read_text(encoding="utf-8"))
    for field in ("cube_position_world_m", "table_offset_m"):
        if not isinstance(initial.get(field), list) or len(initial[field]) != 3:
            raise ValueError(f"{episode_id}: initial.{field} は3要素が必要です")
        initial[field] = [finite_float(v, f"initial.{field}") for v in initial[field]]
    initial_xyz = initial["cube_position_world_m"]
    table = initial["table_offset_m"]
    check_close(initial_xyz[0], table[0] + row["x_m"], f"{episode_id}.initial.x")
    check_close(initial_xyz[1], table[1] + row["y_m"], f"{episode_id}.initial.y")
    for ending in ("start", "end"):
        path = run_dir / "frames" / f"{episode_id}_{ending}.png"
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGB" or image.size != (config["width"], config["height"]):
                raise ValueError(f"{path.name}: PNGの形式・寸法が設定と一致しません")

    frequency = config["control_freq"]
    required_steps = max(1, math.ceil(config["hold_seconds"] * frequency - 1e-9))
    consecutive = 0
    max_consecutive = 0
    max_height = 0.0
    first_success_step = None
    any_env_success = False
    seen = 0
    fields = {"step", "sim_time_s", "cube_x_m", "cube_y_m", "cube_z_m", "lift_height_m",
              "hold_s", "success", "env_success"} | {f"action_{i}" for i in range(7)}
    disturbance = config.get("disturbance")
    if disturbance is not None:
        pulse_steps(disturbance, frequency, config["horizon"])
        fields.update(FORCE_FIELDS)
    with (run_dir / "episodes" / f"{episode_id}.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not fields.issubset(reader.fieldnames):
            raise ValueError(f"{episode_id}: ステップ別CSVの列が不足しています")
        for index, sample in enumerate(reader, start=1):
            if None in sample or any(sample.get(field) is None for field in fields):
                raise ValueError(f"{episode_id}: ステップ別CSVの列数が一致しません")
            if int(sample["step"]) != index:
                raise ValueError(f"{episode_id}: ステップ番号が連続していません")
            label = f"{episode_id}.step{index}"
            check_close(sample["sim_time_s"], index / frequency, f"{label}.sim_time_s")
            for field in fields - {"step", "success", "env_success"}:
                finite_float(sample[field], f"{label}.{field}")
            if disturbance is not None:
                expected_force = force_for_step(disturbance, index, frequency, config["horizon"])
                for key, expected in zip(FORCE_FIELDS, expected_force):
                    check_close(sample[key], expected, f"{label}.{key}")
            relative_height = float(sample["cube_z_m"]) - initial_xyz[2]
            check_close(sample["lift_height_m"], relative_height, f"{label}.lift_height_m")
            consecutive = consecutive + 1 if relative_height >= config["lift_height_m"] else 0
            max_consecutive = max(max_consecutive, consecutive)
            max_height = max(max_height, relative_height)
            if first_success_step is None and consecutive >= required_steps:
                first_success_step = index
            check_close(sample["hold_s"], consecutive / frequency, f"{label}.hold_s")
            if sample["success"] != str(int(first_success_step is not None)):
                raise ValueError(f"{label}: success が高さ・保持時間の再計算と一致しません")
            if sample["env_success"] not in ("0", "1"):
                raise ValueError(f"{label}: env_success は0または1が必要です")
            any_env_success = any_env_success or sample["env_success"] == "1"
            seen = index
    if seen != row["steps"]:
        raise ValueError(f"{episode_id}: ステップ別CSVは {seen} 行、集計は {row['steps']} steps です")
    if row["success"] != int(first_success_step is not None) or row["env_success"] != int(any_env_success):
        raise ValueError(f"{episode_id}: success/env_success がステップ別CSVと一致しません")
    check_close(row["max_height_m"], max_height, f"{episode_id}.max_height_m")
    check_close(row["max_hold_s"], max_consecutive / frequency, f"{episode_id}.max_hold_s")
    if first_success_step is not None:
        check_close(row["time_to_success_s"], first_success_step / frequency, f"{episode_id}.time_to_success_s")
        if first_success_step != seen:
            raise ValueError(f"{episode_id}: 成功後もステップが続いています（終了条件と不一致）")
    elif seen != config["horizon"]:
        raise ValueError(f"{episode_id}: 失敗試行が horizon より前に終了しています")
    return {"trajectory_steps": seen, "metrics_replayed": True, "snapshots_checked": 2}


def verify_run(run_dir: Path, *, allow_partial: bool = False) -> dict[str, Any]:
    run_dir = Path(run_dir)
    # Rechecking an edited or corrupted run must invalidate any earlier pass.
    (run_dir / "verification.json").unlink(missing_ok=True)
    metadata, rows = load_inputs(run_dir)
    validate_rows(metadata, rows, allow_partial=allow_partial)
    if metadata.get("recording_contract") != "lift-tutorial-v1":
        raise ValueError("recording_contract='lift-tutorial-v1' が必要です")
    status = metadata.get("status")
    if status not in ("starting", "running", "failed", "completed"):
        raise ValueError(f"run.json の status が不正です: {status!r}")
    if not allow_partial and status != "completed":
        raise ValueError(f"実行状態が completed ではありません: {status}")
    if metadata.get("episodes_completed") != len(rows):
        raise ValueError("run.json の episodes_completed と CSVの試行数が一致しません")
    config = metadata["config"]
    for dimension in ("width", "height"):
        if type(config.get(dimension)) is not int or config[dimension] <= 0:
            raise ValueError(f"config.{dimension} は正の整数が必要です")
    expected_fps = finite_float(config.get("video_fps", config["control_freq"]), "config.video_fps")
    if expected_fps <= 0:
        raise ValueError("config.video_fps は正の値が必要です")
    check_close(expected_fps, config["control_freq"], "config.video_fps")
    declared_videos = {video_file(run_dir, row["video_path"]) for row in rows}
    actual_videos = {path.resolve() for path in (run_dir / "videos").glob("*.mp4")}
    uncommitted = actual_videos - declared_videos
    if uncommitted and not allow_partial:
        raise ValueError("集計に含まれない動画（途中保存を含む）があります: " + ", ".join(sorted(path.name for path in uncommitted)))
    recordings = []
    for row in rows:
        path = video_file(run_dir, row["video_path"])
        replay = replay_trajectory(run_dir, metadata, row)
        info = inspect_video(path)
        if not math.isclose(info["fps"], expected_fps, abs_tol=0.02, rel_tol=0.001):
            raise ValueError(f"{path.name}: fps={info['fps']} は設定値 {expected_fps} と一致しません")
        if info["frames"] != row["steps"] + 1:
            raise ValueError(f"{path.name}: 動画は {info['frames']} フレーム、必要数は steps+1={row['steps'] + 1} です")
        if (info["width"], info["height"]) != (config["width"], config["height"]):
            raise ValueError(f"{path.name}: 動画の寸法が設定と一致しません")
        recordings.append({"video_path": row["video_path"], **info, **replay})
    expected = len(config["grid_x"]) * len(config["grid_y"]) * config["repeats"]
    result = {"schema_version": 1, "ok": True, "complete": len(rows) == expected and status == "completed" and not uncommitted,
              "run_status": status,
              "allow_partial": allow_partial, "episodes": len(rows), "expected_episodes": expected,
              "uncommitted_videos": sorted(str(path.relative_to(run_dir.resolve())) for path in uncommitted),
              "videos": recordings,
              "scope": "Committed episodes: CSV consistency, metric replay, initial/final snapshots, complete MP4 decoding. Does not certify policy quality or target-PC compatibility"}
    (run_dir / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="試行記録と動画保存の整合性を確認")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="不足試行を許可。壊れた動画・不正な記録は許可しません")
    args = parser.parse_args()
    try:
        result = verify_run(args.run_dir, allow_partial=args.allow_partial)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        # An old success marker must not survive a failed re-verification.
        (args.run_dir / "verification.json").unlink(missing_ok=True)
        parser.exit(1, f"検証失敗: {exc}\n")
    print(f"Verified {result['episodes']}/{result['expected_episodes']} episodes and {len(result['videos'])} MP4 files.")
    print(f"Complete={result['complete']}. Saved: {args.run_dir / 'verification.json'}")


if __name__ == "__main__":
    main()
