#!/usr/bin/env bash
# Rebuild the whole data tree from configs/. Deterministic; safe to re-run
# (existing scenarios and rollouts are skipped).
#
#   scripts/prepare_all.sh              # weather, buildings, splits, scenarios
#   ROLLOUTS=1 scripts/prepare_all.sh   # ... plus baseline rollouts on val (hours)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
WORKERS=${WORKERS:-$(nproc)}

$PY scripts/fetch_weather.py          # skips stations already downloaded
$PY scripts/make_weather_strips.py
$PY scripts/make_buildings.py
$PY scripts/make_scenarios.py --workers "$WORKERS"

if [[ "${ROLLOUTS:-0}" == "1" ]]; then
  # Reference scores on the test-shaped split, for every baseline.
  $PY scripts/collect_rollouts.py --split val --workers "$WORKERS" \
      --policies pid_water pid fixed_setpoint bang_bang
fi
