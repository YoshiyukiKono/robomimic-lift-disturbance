"""Check the Python/CUDA/renderer/MP4 chain before running an experiment."""

import argparse
from contextlib import closing
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

from .checkpoint import DEFAULT_PATH, verify_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--render", action="store_true", help="Render real simulation frames, encode, and decode an MP4")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--output", type=Path, default=Path("outputs/doctor"))
    args = parser.parse_args()
    (args.output / "doctor.json").unlink(missing_ok=True)
    print("Python:", sys.version.split()[0], "Platform:", platform.system())
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("Use the Python3.10 .venv: source .venv/bin/activate")
    import torch
    print("PyTorch:", torch.__version__, "Bundled CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable. See README.md for environment setup; use --device cpu to run without CUDA.")
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} / {memory:.2f} GiB")
        result = torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda")
        torch.cuda.synchronize()
        assert result[0, 0].item() == 32.0
        print("CUDA tensor calculation: PASS")
    expected = {"robomimic": "0.3.0", "robosuite": "1.4.1", "mujoco": "2.3.7", "numpy": "1.26.4"}
    for name, version in expected.items():
        installed = importlib.metadata.version(name)
        print(f"{name}: {installed}")
        if installed != version:
            raise SystemExit(f"Expected {name}=={version}; rerun bootstrap in an isolated .venv")
    print("Checkpoint:", verify_checkpoint(args.checkpoint)["sha256"])
    if not args.render:
        print("Basic checks PASS. Run again with --render to test simulation recording.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    backend = None
    try:
        from .backend import LiftBackend
        from .verify import inspect_video
        import imageio.v2 as imageio
        from PIL import Image
        backend = LiftBackend(args.checkpoint, args.device, "frontview", 640, 480)
        observation, _ = backend.reset(0, 0.0, 0.0)
        frame = backend.frame()
        if float(frame.std()) < 1.0:
            raise RuntimeError("Rendered image is nearly uniform; inspect renderer configuration")
        Image.fromarray(frame).save(args.output / "render.png")
        # This is a genuine short policy rollout, not a repeated still frame.
        with closing(imageio.get_writer(str(args.output / "render.mp4"), fps=backend.control_freq,
                                codec="libx264", pixelformat="yuv420p")) as writer:
            writer.append_data(frame)
            for _ in range(20):
                observation, _, _ = backend.step(observation)
                writer.append_data(backend.frame())
        result = inspect_video(args.output / "render.mp4")
        if result["frames"] != 21:
            raise RuntimeError(f"Expected21frames; got{result['frames']}")
        result.update(ok=True, device=args.device, mujoco_gl=os.environ.get("MUJOCO_GL"),
                      note="Rendering/inference/encoding check; this 1-second clip is not a task-success test.")
        (args.output / "doctor.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        print("Simulation camera -> PNG / H264 MP4 -> full decode: PASS")
        print("Files:", args.output)
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
