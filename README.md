# HVAC control system — Smart Buildings Challenge

Work toward a transferable HVAC control policy for the
[Smart Buildings Challenge](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/),
built on the official [starter kit](https://github.com/smart-buildings-2026/starter-kit)
(pinned as a submodule in `external/starter-kit`).

This first stage is **data preparation**: weather, buildings, train/val
splits, runnable scenarios, and recorded rollouts that a learned or tuned
policy can train on.

## Setup

```bash
git clone --recurse-submodules <this repo>
./setup.sh                 # pinned starter kit + .venv (Python 3.12)
source .venv/bin/activate
scripts/prepare_all.sh     # weather -> buildings -> splits -> scenarios (a few minutes)
```

## What gets built

| step | script | output |
|---|---|---|
| 1. extra climates | `scripts/fetch_weather.py` | `data/weather/csv/<station>.csv` (committed) |
| 2. weather strips | `scripts/make_weather_strips.py` | `data/weather/strips/*.yaml`, `data/weather/strips.csv` |
| 3. buildings | `scripts/make_buildings.py` | `data/buildings/building_<seed>.yaml`, `buildings.csv` (committed) |
| 4. splits + scenarios | `scripts/make_scenarios.py` | `data/splits.yaml` (committed), `data/scenarios/<building>/<strip>/` |
| 5. rollouts | `scripts/collect_rollouts.py` | `data/rollouts/<tag>/<building>/<strip>/<policy>.npz/.json`, `index.csv` |

Settings live in `configs/data.yaml` and `configs/stations.yaml`. Everything is
deterministic, so `data/scenarios`, `data/rollouts` and `data/weather/strips`
are gitignored and rebuilt on demand.

### Weather: 26 stations, 624 strips

The kit ships 6 recorded stations (5 used by its 30 scenarios). The hidden test
uses **unseen weather**, so we add 20 more climates from the Open-Meteo
historical archive (ERA5 reanalysis, 2023, local time, same CSV format the
simulator reads): marine, monsoon, hot-humid, very cold continental, high
altitude, southern hemisphere. Every station is cut into a 30-day strip every
14 days, covering all seasons.

The outside-surface convection coefficient for new stations is set to
6 × mean wind speed (m/s), which matches how the shipped values track wind
(ratio 5.1–6.9 across the six).

Weather data from [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).

### Buildings: 40 train + 6 val

Sampled with the kit's own generator and its envelope-material sampler. The
shipped buildings are smaller than the generator's `DEFAULT_RANGES`, and the
task text doesn't say which ranges the hidden set uses, so 75% are drawn
shipped-sized and 25% generator-sized. Seeds whose floor plan the kit would
reject (zones under 20 m², which diverge in simulation) are skipped at
generation time. Result: 19–192 zones and 4–20 AHUs, against 30–85 zones and
3–9 AHUs in the shipped six.

### Splits

| split | scenarios | buildings | stations | purpose |
|---|---|---|---|---|
| `kit` | 30 | the shipped 6 | 5 shipped | the official local benchmark; matches `baseline_returns.json` |
| `train` | 320 | 40 generated | 21 | training data; 8 strips per building, stations spread out |
| `val` | 30 | 6 generated, unseen | 5 unseen: minneapolis, houston, las_vegas, anchorage, london | **mirrors the hidden test**: 6 unseen buildings × 5 unseen stations |

Use `val` for model selection. A gain that appears on `kit`/`train` but not
`val` is overfitting to buildings or climates.

## Rollouts

```bash
# Baseline reference scores on val (the number to beat)
python scripts/collect_rollouts.py --split val --policies pid_water pid fixed_setpoint bang_bang

# Training data from the best baseline, plus exploratory variants
python scripts/collect_rollouts.py --split train --policies pid_water
python scripts/collect_rollouts.py --split train --policies pid_water --sigma 0.3 --p-flip 0.05

# Your own controller
python scripts/collect_rollouts.py --split val --policies mypkg.policy:MyPolicy
```

Each rollout stores `obs` (T+1 × D), the commanded `continuous_action` and
`discrete_action`, `reward`, every reward component (`rc_*`), raw zone
temperatures in kelvin, outside air temperature and timestamps, plus a JSON with
the building metadata the policy receives at runtime. Exploration noise is
held for `--hold` steps (default 1 h), because the building responds far more
slowly than one 5-minute step.

The `score` in each summary is the leaderboard value for that scenario: the
mean per-step loss, `-total_reward / 8640`. Only full 8640-step episodes are
comparable; shorter runs skip setback periods and look better than they are.

**Cost.** One step takes about 0.05 s (30 zones) to 0.17 s (100 zones) per core,
so a full episode is 7–25 minutes. The val split with one policy is about
3–4 hours on 4 cores; all 320 train scenarios is about a day. Use `--limit`,
`--buildings` and `--stations` to take slices.

### Per-zone tables

Observation and action lengths change with the building, so
`hvacdata.dataset.zone_table` reshapes a rollout into one row per
(step, zone): the zone's sensors and commands, its AHU's, the hot-water loop's,
the building-wide signals, the zone's next temperature, and the reward. A
per-zone model with shared weights trains across all buildings and runs on any
building size.

```python
import pandas as pd
from hvacdata.dataset import load_rollout, zone_table
r = load_rollout("data/rollouts/val/building_09001/houston_d0924/pid_water.npz")
df = pd.DataFrame(zone_table(r))
```

## Things worth knowing about the simulator

Some of these are in the kit's docs and some were found while building this:

- `collect_rollouts.py` reproduces the kit's own `evaluate.py` **bit for bit**
  (`pid_water` on `building_train_8/fairbanks_8`: −668.7475258982959 from both).
- The kit's shipped `baseline_returns.json` does **not** match the current kit
  code: it lists −667.28 for that scenario. Compare against baselines you ran
  yourself (`score_leaderboard.py --baselines self`, or the `kit` split here),
  not against the shipped file.
- `reward = 0.6·normalized_productivity_regret − 0.2·normalized_energy_cost − 0.2·normalized_carbon_emission`,
  exactly (checked against recorded components).
- The pump's `differential_pressure_setpoint` is listed in
  `meta["pressure_indices"]` (as entry K, after the K AHUs), not in
  `water_indices`, which holds only the supply-water temperature.
- The kit runs all simulator code relative to its own root; `hvacdata.paths.in_kit()` handles that.
- Weather CSV `Time` is local time but treated as UTC by the simulator; fetched
  stations follow the same convention so the occupancy schedule lines up with
  daylight.
