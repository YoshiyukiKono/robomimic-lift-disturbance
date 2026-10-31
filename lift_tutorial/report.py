"""Turn real rollout records into an offline, inspectable report.

This module does not run a policy or generate demonstration data. Missing
positions remain missing: an untested position is never counted as a failure.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


FIELDS = (
    "point_id", "x_m", "y_m", "trial", "seed", "success", "env_success",
    "elapsed_s", "time_to_success_s", "max_height_m", "video_path", "steps",
    "max_hold_s",
)
SUMMARY_FIELDS = (
    "point_id", "x_m", "y_m", "n_trials", "n_success", "success_rate",
    "wilson_low", "wilson_high", "mean_success_time_s", "mean_elapsed_s",
    "max_height_m",
)


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}: 数値ではありません: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label}: 有限の数値が必要です")
    return result


def load_inputs(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load schema v1 without importing graphics or simulation dependencies."""
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("run.json: schema_version=1 が必要です")
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("run.json: config が必要です")
    for axis in ("grid_x", "grid_y"):
        values = config.get(axis)
        if not isinstance(values, list) or not values:
            raise ValueError(f"config.{axis}: 空でない配列が必要です")
        converted = [finite_float(v, f"config.{axis}") for v in values]
        if len(converted) != len(set(converted)):
            raise ValueError(f"config.{axis}: 重複があります")
        config[axis] = converted
    repeats = config.get("repeats")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("config.repeats: 1以上の整数が必要です")
    for name in ("hold_seconds", "lift_height_m", "control_freq", "horizon"):
        config[name] = finite_float(config.get(name), f"config.{name}")
        if config[name] <= 0:
            raise ValueError(f"config.{name}: 正の値が必要です")

    rows = []
    with (run_dir / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"episodes.csv: 必須列が不足しています: {FIELDS}")
        for number, raw in enumerate(reader, start=2):
            if None in raw or any(raw.get(field) is None for field in FIELDS):
                raise ValueError(f"episodes.csv:{number}: 列数が一致しません")
            row: dict[str, Any] = dict(raw)
            for name in ("x_m", "y_m", "elapsed_s", "max_height_m", "max_hold_s"):
                row[name] = finite_float(raw[name], f"CSV行{number}.{name}")
            for name in ("trial", "seed", "steps"):
                try:
                    row[name] = int(raw[name])
                except ValueError as exc:
                    raise ValueError(f"CSV行{number}.{name}: 整数が必要です") from exc
            for name in ("success", "env_success"):
                if raw[name] not in ("0", "1"):
                    raise ValueError(f"CSV行{number}.{name}: 0 または 1 が必要です")
                row[name] = int(raw[name])
            row["time_to_success_s"] = (
                finite_float(raw["time_to_success_s"], f"CSV行{number}.time_to_success_s")
                if raw["time_to_success_s"].strip() else None
            )
            if not re.fullmatch(r"[A-Za-z0-9_-]+", row["point_id"]):
                raise ValueError(f"CSV行{number}: point_id は英数字・ハイフン・下線のみ使用できます")
            rows.append(row)
    if not rows:
        raise ValueError("episodes.csv: 実行結果がありません")
    return metadata, rows


def axis_match(value: float, values: list[float]) -> float:
    """Match floating point coordinates to their configured grid point."""
    matches = [candidate for candidate in values if math.isclose(value, candidate, abs_tol=1e-8, rel_tol=0)]
    if len(matches) != 1:
        raise ValueError(f"グリッドに一致しない座標: {value}")
    return matches[0]


def validate_rows(metadata: dict[str, Any], rows: list[dict[str, Any]], *, allow_partial: bool) -> None:
    """Check experimental bookkeeping, independently of simulator availability."""
    config = metadata["config"]
    counts: dict[tuple[float, float], int] = defaultdict(int)
    ids: dict[str, tuple[float, float]] = {}
    point_ids: dict[tuple[float, float], str] = {}
    seen = set()
    seen_seeds = set()
    seen_videos = set()
    dt = 1.0 / config["control_freq"]
    for number, row in enumerate(rows, start=2):
        prefix = f"CSV行{number}"
        point = (axis_match(row["x_m"], config["grid_x"]), axis_match(row["y_m"], config["grid_y"]))
        if row["point_id"] in ids and ids[row["point_id"]] != point:
            raise ValueError(f"{prefix}: 同じ point_id に複数の座標があります")
        if point in point_ids and point_ids[point] != row["point_id"]:
            raise ValueError(f"{prefix}: 同じ座標に複数の point_id があります")
        ids[row["point_id"]] = point
        point_ids[point] = row["point_id"]
        key = (*point, row["trial"])
        if key in seen:
            raise ValueError(f"{prefix}: 試行が重複しています")
        seen.add(key)
        seed_key = (*point, row["seed"])
        if seed_key in seen_seeds:
            raise ValueError(f"{prefix}: 同じ位置で seed が重複しています")
        seen_seeds.add(seed_key)
        counts[point] += 1
        if not 0 <= row["trial"] < config["repeats"] or row["seed"] < 0 or not 0 < row["steps"] <= config["horizon"]:
            raise ValueError(f"{prefix}: trial/seed/steps が範囲外です")
        if "seed" in config and row["seed"] != config["seed"] + row["trial"]:
            raise ValueError(f"{prefix}: seed が config.seed + trial と一致しません")
        expected_elapsed = row["steps"] * dt
        if not math.isclose(row["elapsed_s"], expected_elapsed, abs_tol=1e-5, rel_tol=1e-6):
            raise ValueError(f"{prefix}: elapsed_s が steps/control_freq と一致しません")
        if not 0 <= row["max_hold_s"] <= row["elapsed_s"] + 1e-6:
            raise ValueError(f"{prefix}: max_hold_s が範囲外です")
        success_time = row["time_to_success_s"]
        if row["success"]:
            if success_time is None or not 0 < success_time <= row["elapsed_s"] + 1e-6:
                raise ValueError(f"{prefix}: 成功時刻がありません、または範囲外です")
            if row["max_hold_s"] + 1e-6 < config["hold_seconds"]:
                raise ValueError(f"{prefix}: 成功ですが保持時間が不足しています")
            if row["max_height_m"] + 1e-6 < config["lift_height_m"]:
                raise ValueError(f"{prefix}: 成功ですが高さが不足しています")
        elif success_time is not None:
            raise ValueError(f"{prefix}: 失敗なのに成功時刻があります")
        elif row["max_hold_s"] + 1e-6 >= config["hold_seconds"]:
            raise ValueError(f"{prefix}: 保持時間を満たしていますが success=0 です")
        video = row["video_path"]
        if not video:
            raise ValueError(f"{prefix}: video_path が空です")
        if video in seen_videos:
            raise ValueError(f"{prefix}: 動画が別の試行と重複しています")
        seen_videos.add(video)
    for x in config["grid_x"]:
        for y in config["grid_y"]:
            count = counts[(x, y)]
            if count > config["repeats"] or (not allow_partial and count != config["repeats"]):
                raise ValueError(f"位置 x={x}, y={y}: {count}/{config['repeats']} 試行です")


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """95% Wilson score interval; descriptive uncertainty, not generalization."""
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["point_id"], row["x_m"], row["y_m"])].append(row)
    result = []
    for (point_id, x, y), group in sorted(grouped.items(), key=lambda item: (item[0][2], item[0][1])):
        successes = sum(row["success"] for row in group)
        times = [row["time_to_success_s"] for row in group if row["success"]]
        low, high = wilson_interval(successes, len(group))
        result.append({
            "point_id": point_id, "x_m": x, "y_m": y,
            "n_trials": len(group), "n_success": successes,
            "success_rate": successes / len(group), "wilson_low": low,
            "wilson_high": high,
            "mean_success_time_s": sum(times) / len(times) if times else None,
            "mean_elapsed_s": sum(row["elapsed_s"] for row in group) / len(group),
            "max_height_m": max(row["max_height_m"] for row in group),
        })
    return result


def render_maps(metadata: dict[str, Any], summary: list[dict[str, Any]], run_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    xs = sorted(metadata["config"]["grid_x"])
    ys = sorted(metadata["config"]["grid_y"])
    lookup = {(axis_match(row["x_m"], xs), axis_match(row["y_m"], ys)): row for row in summary}
    for kind in ("success", "time"):
        values = np.full((len(ys), len(xs)), np.nan)
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                row = lookup.get((x, y))
                if row:
                    if kind == "success":
                        values[yi, xi] = 100 * row["success_rate"]
                    elif row["mean_success_time_s"] is not None:
                        values[yi, xi] = row["mean_success_time_s"]
        fig, ax = plt.subplots(figsize=(7.8, 6.0), constrained_layout=True)
        cmap = plt.get_cmap("YlGnBu" if kind == "success" else "viridis").copy()
        cmap.set_bad("#e2e8f0")
        upper = 100 if kind == "success" else max(1, float(np.nanmax(values)) if np.isfinite(values).any() else 1)
        im = ax.imshow(np.ma.masked_invalid(values), origin="lower", vmin=0, vmax=upper, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(xs)), [f"{x:+.3f}" for x in xs])
        ax.set_yticks(range(len(ys)), [f"{y:+.3f}" for y in ys])
        ax.set_xlabel("Requested table-frame x [m]")
        ax.set_ylabel("Requested table-frame y [m]")
        ax.set_title("Lift success rate" if kind == "success" else "Time to success (successful trials only)", pad=14)
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                row = lookup.get((x, y))
                if row is None:
                    label = "NOT TESTED"
                elif kind == "success":
                    label = f"{100 * row['success_rate']:.0f}%\n{row['n_success']}/{row['n_trials']}"
                elif row["mean_success_time_s"] is None:
                    label = "NO SUCCESS"
                else:
                    label = f"{row['mean_success_time_s']:.2f} s\nn={row['n_success']}"
                val = values[yi, xi]
                colour = "white" if np.isfinite(val) and ((kind == "success" and val > 55) or (kind == "time" and val < upper * .4)) else "#14213d"
                ax.text(xi, yi, label, ha="center", va="center", color=colour, fontsize=10)
        fig.colorbar(im, ax=ax, label="Success [%]" if kind == "success" else "Simulation time [s]")
        fig.savefig(run_dir / f"{kind}_map.png", dpi=150)
        plt.close(fig)


def write_report(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    metadata, rows = load_inputs(run_dir)
    validate_rows(metadata, rows, allow_partial=True)
    summary = summarize(rows)
    expected = len(metadata["config"]["grid_x"]) * len(metadata["config"]["grid_y"]) * metadata["config"]["repeats"]
    total = {"schema_version": 1, "completed_episodes": len(rows), "expected_episodes": expected,
             "complete": len(rows) == expected and metadata.get("status") == "completed",
             "run_status": metadata.get("status", "unknown"),
             "successes": sum(row["success"] for row in rows),
             "points": summary}
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    (run_dir / "summary.json").write_text(json.dumps(total, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_maps(metadata, summary, run_dir)

    config = metadata["config"]
    multiple_positions = len(config["grid_x"]) * len(config["grid_y"]) > 1
    position_description = " / ".join(
        f"{axis.upper()} = {', '.join(f'{value:+.3f}' for value in config[f'grid_{axis}'])} m"
        for axis in ("x", "y")
    )
    seed_description = "、".join(str(seed) for seed in sorted({row["seed"] for row in rows}))
    disturbance = config.get("disturbance")
    if disturbance is not None and any(disturbance["force_n"]):
        force_description = (
            "世界座標での外力 (Fx, Fy, Fz) = ("
            + ", ".join(f"{force:+g}" for force in disturbance["force_n"])
            + f") N。開始 {disturbance['start_s']:.2f} 秒後から {disturbance['duration_s']:.2f} 秒間。"
        )
    else:
        force_description = "外乱なし（物体への外力の付与なし）。"
    success_heading = "初期位置別の集計" if multiple_positions else "この条件での成否"
    success_alt = "初期位置別の成功率。未実施はNOT TESTED。" if multiple_positions else "この条件での成功回数と成功率。"
    success_note = (
        "座標はテーブル基準の要求位置です。画面の左右とは必ずしも一致しません。各セルに成功回数／試行回数を表示しています。"
        if multiple_positions else
        "初期位置と外力設定を共通にし、シードを変えた試行の成功回数／試行回数を表示しています。図中の座標は固定した初期位置です。"
    )
    position_headers = "<th>位置ID</th><th>x [m]</th><th>y [m]</th>" if multiple_positions else "<th>集計対象</th>"

    def picture(filename: str, alt: str) -> str:
        data = base64.b64encode((run_dir / filename).read_bytes()).decode("ascii")
        return f'<img src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'

    table_rows = []
    for row in summary:
        mean = "—" if row["mean_success_time_s"] is None else f"{row['mean_success_time_s']:.2f}"
        position_cells = (
            f"<td>{html.escape(row['point_id'])}</td><td>{row['x_m']:+.3f}</td><td>{row['y_m']:+.3f}</td>"
            if multiple_positions else "<td>この条件</td>"
        )
        table_rows.append(f"<tr>{position_cells}"
                          f"<td>{row['n_success']}/{row['n_trials']}</td><td>{row['success_rate']:.1%}</td>"
                          f"<td>{row['wilson_low']:.1%}–{row['wilson_high']:.1%}</td><td>{mean}</td></tr>")
    if total["complete"]:
        warning = "全予定試行の結果です。"
    elif multiple_positions:
        warning = "未完了の実験です。未実施位置は灰色で表示し、成功率の分母に含めません。"
    else:
        warning = "未完了の実験です。集計は記録済みの試行のみを対象としています。"
    document = f'''<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Liftタスクの実行結果</title>
<style>body{{font:16px/1.8 system-ui,sans-serif;color:#183047;background:#f5f7fa;margin:0}}main{{max-width:1050px;margin:auto;padding:40px 24px}}h1{{line-height:1.4;font-size:30px}}h2{{margin-top:36px}}.card{{background:white;padding:24px;border-radius:12px;margin:20px 0}}.notice{{border-left:4px solid #dd9b20;padding:12px 18px;background:#fff6df}}.metrics{{font-size:22px;font-weight:600}}img{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:right;padding:10px;border-bottom:1px solid #dbe2ea}}th:first-child,td:first-child{{text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}.scroll{{overflow:auto}}code{{background:#edf1f6;padding:2px 5px}}</style>
<main><p>robomimic Lift / 学習済みモデルによる実験</p><h1>Liftタスクの実行結果</h1>
<p>この実験条件における各試行の成否と、成功までの時間を集計します。以下は <code>episodes.csv</code> の記録から計算した結果です。</p>
<div class="card"><h2>今回の実験条件</h2><p>{html.escape(force_description)}</p><p>テーブル基準の初期位置：{html.escape(position_description)}</p><p>記録済みのシード：{html.escape(seed_description)}</p></div>
<div class="card metrics">完了 {len(rows)} / {expected} 試行　・　成功 {total['successes']} / {len(rows)} 試行</div>
<p class="notice">{warning} 少数試行の結果から、一般的な性能や条件の違いによる因果関係を断定することはできません。</p>
<div class="card"><h2>今回の成功判定</h2><p>物体の中心が初期位置より <strong>{config['lift_height_m']:.3f} m</strong> 以上高い状態を、連続して <strong>{config['hold_seconds']:.2f} 秒</strong> 満たしたら成功です。これは本プロジェクト独自の判定で、環境標準の <code>env_success</code> とは別に記録します。</p>
<p>高さだけでは、指で確実に把持していることまでは証明できません。動画と合わせて解釈してください。時間はシミュレーション内の時間で、PC上の処理時間ではありません。</p></div>
<div class="card"><h2>{success_heading}</h2>{picture('success_map.png', success_alt)}<p>{success_note}</p></div>
<div class="card"><h2>成功までの時間</h2>{picture('time_map.png', '成功試行のみの平均成功時間。成功ゼロはNO SUCCESS。')}<p>成功した試行だけの平均です。失敗した試行は平均に含めていません。成功がない場合は「NO SUCCESS」と表示します。</p></div>
<div class="card"><h2>数値表</h2><div class="scroll"><table><thead><tr>{position_headers}<th>成功/試行</th><th>成功率</th><th>95% Wilson区間</th><th>成功時間 [s]</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div><p>区間は二項比率の不確かさの目安です。試行条件の偏りや、モデルの学習環境との違いを取り除くものではありません。</p></div>
<div class="card"><h2>動画と元データ</h2><p>試行ごとの動画は <code>episodes.csv</code> の <code>video_path</code> に記載されています。CSV、PNG、JSON、MP4を同じ実験フォルダに保存します。このHTMLの図は埋め込み済みで、インターネット接続は不要です。</p>
<p>次に考える問い：失敗は接近・把持・持ち上げのどの段階か。試行回数を増やしても傾向が続くか。基準の高さや保持時間を変えると結果はどう変わるか。</p></div>
<details class="card"><summary>実行設定・出典情報</summary><pre>{html.escape(json.dumps(metadata, ensure_ascii=False, indent=2))}</pre></details>
</main></html>'''
    (run_dir / "report.html").write_text(document, encoding="utf-8")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Liftタスクの実行結果を集計し、PNGとオフラインHTMLに保存")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = write_report(args.run_dir)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, f"集計失敗: {exc}\n")
    print(f"Report saved: {args.run_dir / 'report.html'}")
    print(f"Episodes: {result['completed_episodes']}/{result['expected_episodes']}; complete={result['complete']}")


if __name__ == "__main__":
    main()
