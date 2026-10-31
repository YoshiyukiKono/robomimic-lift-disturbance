#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

mode="cuda"
if [[ "${1:-}" == "--cpu" ]]; then mode="cpu"; shift; fi
if [[ "$#" -ne 0 ]]; then echo "Usage: bash scripts/bootstrap.sh [--cpu]" >&2; exit 2; fi
if [[ "$(uname -s)" != Linux ]]; then echo "Run this script inside Ubuntu (WSL2), not Windows PowerShell." >&2; exit 1; fi
for program in git cmake cc; do
  if ! command -v "$program" >/dev/null; then
    echo "Missing $program. Install README's Ubuntu prerequisites first." >&2; exit 1
  fi
done

if command -v uv >/dev/null; then
  uv_bin="$(command -v uv)"
else
  # This helper venv uses Ubuntu's own Python only to install the environment manager.
  # No sudo pip, no system Python replacement, no downloaded shell script execution.
  if [[ ! -x .tools/uv-env/bin/uv ]]; then
    python3 -m venv .tools/uv-env
    .tools/uv-env/bin/python -m pip install 'uv==0.13.4'
  fi
  uv_bin="$PWD/.tools/uv-env/bin/uv"
fi
if [[ -e .venv && ! -x .venv/bin/python ]]; then
  echo ".venv exists but is not a valid environment; move it aside explicitly before retrying." >&2; exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  "$uv_bin" venv --python 3.10.22 --seed .venv
fi
.venv/bin/python - <<'PY'
import sys
if sys.version_info[:3] != (3, 10, 22):
    raise SystemExit('Existing .venv is not Python 3.10.22. Use a fresh checkout/environment.')
PY
torch_index="https://download.pytorch.org/whl/cu121"
torch_variant="cu121"
if [[ "$mode" == cpu ]]; then torch_index="https://download.pytorch.org/whl/cpu"; torch_variant="cpu"; fi
"$uv_bin" pip install --python .venv/bin/python \
  "torch==2.2.2+$torch_variant" "torchvision==0.17.2+$torch_variant" \
  -r requirements.lock \
  --index-url https://pypi.org/simple --extra-index-url "$torch_index" \
  --index-strategy unsafe-best-match
"$uv_bin" pip check --python .venv/bin/python
mkdir -p outputs
"$uv_bin" pip freeze --python .venv/bin/python > "outputs/installed-$mode.txt"
echo "Environment ready ($mode). Next: source .venv/bin/activate"
echo "Then: python -m lift_tutorial.checkpoint"
