"""Verify and compare matched seed trials; write CSV, plots and local HTML."""
import argparse
import csv
import html
import json
import math
from pathlib import Path

from .report import load_inputs, write_report
from .verify import verify_run
from .disturbance import pulse_steps

CONDITIONS = ("baseline", "push_x_1n", "push_x_1p5n", "push_minus_x_1n")


def episode_id(row):
    return f"{row['point_id']}_trial{row['trial']:02d}"


def read_trace(directory, row):
    with (directory / "episodes" / (episode_id(row) + ".csv")).open() as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def compare(root):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    for name in ("comparison.csv", "comparison.html"):
        (root / name).unlink(missing_ok=True)
    runs = {}
    for name in CONDITIONS:
        directory = root / name
        verify_run(directory)
        write_report(directory)
        meta, rows = load_inputs(directory)
        runs[name] = (meta, {(r["x_m"], r["y_m"], r["seed"]): r for r in rows})
    bm, baseline = runs["baseline"]
    if any(bm["config"]["disturbance"]["force_n"]):
        raise ValueError("Baseline must use zero force")
    expected_config = {k: v for k, v in bm["config"].items() if k != "disturbance"}
    records = []
    figures = []
    for number, (key, base_row) in enumerate(baseline.items()):
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        base_initial = json.loads((root / "baseline" / "episodes" / (episode_id(base_row) + "_initial.json")).read_text())
        base_trace = read_trace(root / "baseline", base_row)
        for name, (meta, indexed) in runs.items():
            if set(indexed) != set(baseline):
                raise ValueError("Position/seed pairs differ between conditions")
            if {k: v for k, v in meta["config"].items() if k != "disturbance"} != expected_config:
                raise ValueError("Only disturbance settings may differ")
            if meta["checkpoint"] != bm["checkpoint"]:
                raise ValueError("Checkpoint metadata differs")
            for field in ("device", "versions", "effective_environment_kwargs", "mujoco_gl"):
                if meta["environment"][field] != bm["environment"][field]:
                    raise ValueError(f"Environment differs: {field}")
            row = indexed[key]
            directory = root / name
            initial = json.loads((directory / "episodes" / (episode_id(row) + "_initial.json")).read_text())
            for field in ("qpos", "qvel", "cube_half_size_m", "table_offset_m"):
                a, b = np.asarray(initial[field]), np.asarray(base_initial[field])
                if a.shape != b.shape or not np.allclose(a, b, atol=1e-10, rtol=0):
                    raise ValueError(f"Initial state differs: {name}, seed={key[2]}, {field}")
            trace = read_trace(directory, row)
            pulse = meta["config"]["disturbance"]
            start, duration = pulse_steps(pulse, meta["config"]["control_freq"], meta["config"]["horizon"])
            if (pulse["start_s"], pulse["duration_s"]) != (bm["config"]["disturbance"]["start_s"], bm["config"]["disturbance"]["duration_s"]):
                raise ValueError("Pulse timing differs")
            position_fields = ["cube_x_m", "cube_y_m", "cube_z_m"]
            pre_fields = position_fields + [f"action_{i}" for i in range(7)]
            for a, b in zip(trace[:start], base_trace[:start]):
                if any(not math.isclose(a[k], b[k], abs_tol=1e-7, rel_tol=1e-7) for k in pre_fields):
                    raise ValueError(f"Pre-pulse trajectory differs: {name}, seed={key[2]}")
            initial_xyz = initial["cube_position_world_m"]
            times = [0] + [r["sim_time_s"] for r in trace]
            for axis, field, initial_value in zip(axes, position_fields[:2], initial_xyz[:2]):
                axis.plot(times, [0] + [100 * (r[field] - initial_value) for r in trace], label=name)
            pulse_complete = len(trace) >= start + duration
            end_delta = ""
            if pulse_complete and len(base_trace) >= start + duration:
                a, b = trace[start + duration - 1], base_trace[start + duration - 1]
                end_delta = math.hypot(a["cube_x_m"]-b["cube_x_m"], a["cube_y_m"]-b["cube_y_m"])
            nonzero_steps = sum(any(r[f"push_f{axis}_n"] for axis in "xyz") for r in trace)
            records.append(dict(condition=name, seed=key[2], x_m=key[0], y_m=key[1],
                baseline_success=base_row["success"], success=row["success"],
                time_to_success_s=row["time_to_success_s"],
                force_x_n=pulse["force_n"][0], force_y_n=pulse["force_n"][1],
                pulse_start_s=pulse["start_s"], pulse_duration_s=pulse["duration_s"],
                pulse_window_completed=int(pulse_complete), nonzero_force_steps=nonzero_steps,
                paired_xy_difference_at_pulse_end_m=end_delta,
                video_path=name + "/" + row["video_path"]))
        for axis, label in zip(axes, ["Cube X displacement (cm)", "Cube Y displacement (cm)"]):
            p = bm["config"]["disturbance"]
            axis.axvspan(p["start_s"], p["start_s"] + p["duration_s"], color="gray", alpha=.2)
            axis.set_ylabel(label)
            axis.grid(alpha=.25)
        axes[0].set_title(f"Matched seed {key[2]} / initial XY ({key[0]}, {key[1]}) m")
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("Simulation time (s)")
        fig.tight_layout()
        filename = f"comparison_pair{number:02d}_seed{key[2]}.png"
        fig.savefig(root / filename, dpi=140)
        plt.close(fig)
        figures.append(filename)
    with (root / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ['<!doctype html><meta charset="utf-8"><title>外乱比較</title>',
        '<style>body{font-family:sans-serif;max-width:1100px;margin:30px auto;padding:20px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #bbb}img{max-width:100%}video{width:480px;max-width:100%}</style>',
        '<h1>同じ初期状態・シードでの外乱比較</h1>',
        '<p>動画全フレーム・成功指標・外力記録を検証済み。初期状態と外力前の軌跡も比較しています。各条件2試行の探索的実験です。</p>',
        '<p>灰色は力を加える時間帯。変位は初期位置からの値で、把持後の動きも含みます。外乱なしでも失敗するシードは、成功例への外乱の評価から分けて扱ってください。外乱なしの成功を確認して選んだシードなので、一般的な成功率の推定には使えません。</p>',
        '<table><tr><th>条件</th><th>seed</th><th>力 Fx, Fy (N)</th><th>基準成功</th><th>成功</th><th>成功まで(s)</th><th>外力ステップ数</th><th>パルス完了</th><th>パルス終了時の基準との差(cm)</th></tr>']
    for r in records:
        delta = r['paired_xy_difference_at_pulse_end_m']
        values = [r['condition'], r['seed'], f"{r['force_x_n']}, {r['force_y_n']}", r['baseline_success'], r['success'], r['time_to_success_s'],
                  r['nonzero_force_steps'], r['pulse_window_completed'], '' if delta == '' else f'{delta*100:.3f}']
        lines.append('<tr>' + ''.join('<td>'+html.escape(str(v))+'</td>' for v in values) + '</tr>')
    lines.append('</table><p>パルス完了=0は、その時間帯までに試行が終了したケースです。外乱を完了した比較として数えません。</p>')
    lines += [f'<img src="{f}">' for f in figures]
    for r in records:
        lines.append(f'<h3>{r["condition"]} / seed {r["seed"]}</h3><video controls preload="metadata" src="{r["video_path"]}"></video>')
    (root / "comparison.html").write_text('\n'.join(lines), encoding="utf-8")
    print(f"Compared {len(records)} episodes. Open: {root / 'comparison.html'}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    compare(args.root)


if __name__ == "__main__":
    main()
