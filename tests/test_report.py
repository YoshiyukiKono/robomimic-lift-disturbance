"""Synthetic unit fixtures live only in temporary directories, not outputs/."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from lift_tutorial.report import FIELDS, load_inputs, summarize, validate_rows, wilson_interval, write_report
from lift_tutorial.verify import verify_run, video_file


def fixture(root, *, repeats=1, rows=None, grid_x=None):
    metadata = {"schema_version": 1, "recording_contract": "lift-tutorial-v1", "status": "completed", "config": {"grid_x": grid_x or [0.0], "grid_y": [0.0],
                "repeats": repeats, "control_freq": 20, "horizon": 100,
                "hold_seconds": .5, "lift_height_m": .08, "width": 64, "height": 64, "seed": 7}}
    default = {"point_id": "centre", "x_m": 0, "y_m": 0, "trial": 0, "seed": 7,
               "success": 1, "env_success": 1, "elapsed_s": 1,
               "time_to_success_s": 1, "max_height_m": .1, "video_path": "videos/centre_trial00.mp4",
               "steps": 20, "max_hold_s": .5}
    rows = rows if rows is not None else [default]
    metadata["episodes_completed"] = len(rows)
    (root / "run.json").write_text(json.dumps(metadata))
    with (root / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return metadata, default


def recording_fixture(root):
    """20 steps: first 10 on table, next 10 lifted; success at exactly 1 s."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    metadata, row = fixture(root)
    for directory in ("episodes", "videos", "frames"):
        (root / directory).mkdir(exist_ok=True)
    episode = "centre_trial00"
    (root / "episodes" / f"{episode}_initial.json").write_text(json.dumps({
        "cube_position_world_m": [0, 0, .8], "table_offset_m": [0, 0, .75]}))
    fields = ["step", "sim_time_s", "cube_x_m", "cube_y_m", "cube_z_m", "lift_height_m",
              "hold_s", "success", "env_success"] + [f"action_{i}" for i in range(7)]
    with (root / "episodes" / f"{episode}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in range(1, 21):
            height = .1 if step > 10 else 0
            writer.writerow({"step": step, "sim_time_s": step/20, "cube_x_m": 0,
                             "cube_y_m": 0, "cube_z_m": .8 + height, "lift_height_m": height,
                             "hold_s": max(0, step-10)/20, "success": int(step == 20),
                             "env_success": int(step > 10), **{f"action_{i}": 0 for i in range(7)}})
    for suffix in ("start", "end"):
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(root / "frames" / f"{episode}_{suffix}.png")
    path = root / row["video_path"]
    with imageio.get_writer(str(path), fps=20, codec="libx264") as writer:
        for frame_index in range(21):
            writer.append_data(np.full((64, 64, 3), frame_index * 10, dtype=np.uint8))
    return metadata, row, path


class ReportTests(unittest.TestCase):
    def test_failure_is_not_zero_second_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, row = fixture(root)
            row.update(success=0, env_success=0, time_to_success_s="", max_hold_s=.2)
            fixture(root, rows=[row])
            meta, rows = load_inputs(root)
            validate_rows(meta, rows, allow_partial=False)
            summary = summarize(rows)
            self.assertEqual(summary[0]["success_rate"], 0)
            self.assertIsNone(summary[0]["mean_success_time_s"])

    def test_missing_cells_are_not_counted_as_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, grid_x=[0, .05])
            meta, rows = load_inputs(root)
            validate_rows(meta, rows, allow_partial=True)
            with self.assertRaisesRegex(ValueError, "0/1"):
                validate_rows(meta, rows, allow_partial=False)
            self.assertEqual(summarize(rows)[0]["success_rate"], 1)
            result = write_report(root)
            self.assertFalse(result["complete"])
            self.assertEqual(result["completed_episodes"], 1)
            self.assertEqual(result["expected_episodes"], 2)
            self.assertTrue((root / "report.html").is_file())
            self.assertIn("data:image/png;base64", (root / "report.html").read_text())

    def test_duplicate_trial_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, row = fixture(root)
            fixture(root, repeats=2, rows=[row, row])
            meta, rows = load_inputs(root)
            with self.assertRaisesRegex(ValueError, "重複"):
                validate_rows(meta, rows, allow_partial=False)

    def test_nonfinite_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, row = fixture(root)
            row["max_height_m"] = "nan"
            fixture(root, rows=[row])
            with self.assertRaisesRegex(ValueError, "有限"):
                load_inputs(root)

    def test_false_success_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, row = fixture(root)
            row["max_hold_s"] = .1
            fixture(root, rows=[row])
            meta, rows = load_inputs(root)
            with self.assertRaisesRegex(ValueError, "保持時間"):
                validate_rows(meta, rows, allow_partial=False)

    def test_elapsed_uses_simulation_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, row = fixture(root)
            row["elapsed_s"] = 50
            fixture(root, rows=[row])
            meta, rows = load_inputs(root)
            with self.assertRaisesRegex(ValueError, "steps/control_freq"):
                validate_rows(meta, rows, allow_partial=False)

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                video_file(Path(directory), "../elsewhere.mp4")

    def test_trial_range_and_seed_pairing_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, row = fixture(root)
            row["trial"] = 1
            fixture(root, rows=[row])
            meta, rows = load_inputs(root)
            with self.assertRaisesRegex(ValueError, "範囲外"):
                validate_rows(meta, rows, allow_partial=False)
            row.update(trial=0, seed=12345)
            fixture(root, rows=[row])
            meta, rows = load_inputs(root)
            with self.assertRaisesRegex(ValueError, "seed"):
                validate_rows(meta, rows, allow_partial=False)

    def test_wilson_interval_has_uncertainty_at_extremes(self):
        low, high = wilson_interval(3, 3)
        self.assertLess(low, .5)
        self.assertAlmostEqual(high, 1)
        low, high = wilson_interval(0, 3)
        self.assertAlmostEqual(low, 0)
        self.assertGreater(high, .5)

    @unittest.skipUnless(importlib.util.find_spec("imageio") and importlib.util.find_spec("imageio_ffmpeg"), "video dependencies not installed")
    def test_video_complete_decode_and_truncation_detection(self):
        import imageio.v2 as imageio
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, path = recording_fixture(root)
            result = verify_run(root)
            self.assertEqual(result["videos"][0]["frames"], 21)
            self.assertTrue(result["videos"][0]["metrics_replayed"])
            # Even a single missing final frame must fail (previously tolerated).
            with imageio.get_writer(str(path), fps=20, codec="libx264") as writer:
                for _ in range(20):
                    writer.append_data(np.zeros((64, 64, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "フレーム"):
                verify_run(root)
            self.assertFalse((root / "verification.json").exists())
            path.write_bytes(b"not an mp4")
            with self.assertRaisesRegex(ValueError, "読み込めません"):
                verify_run(root)

    @unittest.skipUnless(importlib.util.find_spec("imageio") and importlib.util.find_spec("imageio_ffmpeg"), "video dependencies not installed")
    def test_failed_status_and_leftovers_are_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, _, _ = recording_fixture(root)
            metadata["status"] = "failed"
            (root / "run.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "completed"):
                verify_run(root)
            self.assertFalse(verify_run(root, allow_partial=True)["complete"])
            metadata["status"] = "completed"
            (root / "run.json").write_text(json.dumps(metadata))
            (root / "videos" / "unfinished.partial.mp4").write_bytes(b"unfinished")
            with self.assertRaisesRegex(ValueError, "途中保存"):
                verify_run(root)
            self.assertFalse(verify_run(root, allow_partial=True)["complete"])

    @unittest.skipUnless(importlib.util.find_spec("imageio") and importlib.util.find_spec("imageio_ffmpeg"), "video dependencies not installed")
    def test_success_time_must_match_height_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, row, _ = recording_fixture(root)
            # Aggregate fields look internally plausible but the trace has no
            # qualifying samples before 0.55 s, so 0.4 s success is impossible.
            row["time_to_success_s"] = .4
            fixture(root, rows=[row])
            with self.assertRaisesRegex(ValueError, "time_to_success_s"):
                verify_run(root)

    @unittest.skipUnless(importlib.util.find_spec("imageio") and importlib.util.find_spec("imageio_ffmpeg"), "video dependencies not installed")
    def test_actual_video_size_must_match_config(self):
        import imageio.v2 as imageio
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, path = recording_fixture(root)
            with imageio.get_writer(str(path), fps=20, codec="libx264") as writer:
                for _ in range(21):
                    writer.append_data(np.zeros((64, 80, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "寸法"):
                verify_run(root)


if __name__ == "__main__":
    unittest.main()
