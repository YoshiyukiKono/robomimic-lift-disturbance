#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
device="${1:-cuda}"
if [[ "$#" -gt 1 || ( "$device" != cuda && "$device" != cpu ) ]]; then
  echo "Usage: bash scripts/run_disturbance.sh [cuda|cpu]" >&2; exit 2
fi
if [[ ! -x .venv/bin/python ]]; then echo 'Activate/install the original tutorial environment first.' >&2; exit 1; fi
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
root="outputs/disturbance-$(date +%Y%m%d-%H%M%S)-$$"
echo "Results: $root"
for condition in baseline push_x_1n push_x_1p5n push_minus_x_1n; do
  .venv/bin/python -u -m lift_tutorial.run \
    --config "configs/disturbance_${condition}.json" \
    --output "$root/$condition" --device "$device"
done
.venv/bin/python -m lift_tutorial.compare_disturbance --root "$root"
echo "Open comparison.html in: $root"
