#!/usr/bin/env bash
# One-time setup: fetch the pinned starter kit and install it into .venv.
#
#   ./setup.sh            # uses uv if available, else python3 -m venv + pip
#   source .venv/bin/activate
set -euo pipefail
cd "$(dirname "$0")"

git submodule update --init --recursive external/starter-kit

if command -v uv >/dev/null 2>&1; then
  uv venv -p 3.12 .venv
  VIRTUAL_ENV="$PWD/.venv" uv pip install -e external/starter-kit
else
  python3 -m venv .venv   # needs Python >= 3.12 (the kit's requirement)
  .venv/bin/pip install -U pip
  .venv/bin/pip install -e external/starter-kit
fi

echo "Done. Activate with: source .venv/bin/activate"
