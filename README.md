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

## Feature encoding

The raw observation is not usable as model input as delivered (details in
`hvacdata/features.py` and the data audit report):

- **Mixed units.** The env's normaliser matches field names *exactly* against a
  13-entry table and passes everything else through raw. Zone temperature
  arrives z-scored while the zone's own setpoints arrive in kelvin; static
  pressure is 0–2000 Pa, occupancy a head count, request counts equal the
  number of zones served.
- **Redundancy.** Hour-of-day is 12 phase-shifted sin/cos pairs of the same
  24 h cycle (24 columns, rank 2), likewise day-of-week; AHU sensors equal their
  setpoints; three fan flags are identical; outside air is repeated per AHU; the
  per-zone airflow setpoint is constant.
- **Variable shape.** Length and positions change per building.

`hvacdata.features.BuildingEncoder(meta)` (version 2) turns each observation
into `zone` (V × 31) and `glob` (25) arrays. Every column is declared once in
`ZONE_SPEC` / `GLOBAL_SPEC` with its unit, transform, clip and reason;
**[docs/FEATURES.md](docs/FEATURES.md)** is generated from that table plus
statistics measured on the training data (`scripts/feature_catalog.py`).
The practices applied, column by column:

- **Fixed physical scales, not fitted ones** (5 K = the reward's comfort knee,
  2000 Pa = the fan's range), so an unseen building or climate cannot shift
  them. `FeatureScaler` adds a robust (median/IQR) standardisation fitted on
  the train split only, saved as `configs/feature_scaler.json`, for models that
  need unit variance.
- **Tails:** `asinh` for heavy-tailed signed values (5-minute temperature
  change spans ±110 K/h), `log1p` plus a 0/1 indicator for zero-inflated
  positives (the quadratic comfort shortfall, skew 21); every column clipped
  to a stated range.
- **Gating:** readings that mean nothing while equipment is off (supply-air
  temperature of a stopped AHU, pump pressure of a stopped pump) are zeroed
  by their run flag.
- **Dropping:** constant and duplicate fields (cooling-request fraction is
  always 1; static pressure tracks airflow at r=0.96); two-valued columns
  become flags.
- **Interactions from physics:** heat the supply air can deliver
  (airflow × damper × ΔT) and heat the reheat coil can deliver.
- **History without a sequence model:** 5-minute and 1-hour changes,
  6-hour moving averages of zone temperature and of outdoor temperature,
  a 24-hour one of outdoor temperature, hours out of band, hours since the AHU
  switched.
- **Runtime-safe:** only the observation and the metadata a policy receives
  (0.15 ms per step), so the same class builds training data and runs inside a
  submission. `hvacdata/features_v1.py` is the first version, kept for
  comparisons.

```bash
python scripts/build_features.py --tag diverse              # data/features/diverse/{zone,step,zones_static}
python scripts/build_features.py --tag analysis --version 1 --out-tag analysis_v1
python scripts/feature_catalog.py --tags diverse analysis   # docs/FEATURES.md + configs/feature_scaler.json
```

Zone rows also carry the full commanded action at t in scaled units (zone
damper and reheat, its AHU's run / supply temperature / pressure /
recirculation, water temperature / pressure / pump) and targets
(`y_dtemp`, `y_dtemp_1h`, `y_shortfall`); step rows carry the reward split into
comfort, cost and carbon terms.

## Collecting varied training data

A model only learns the effect of actions it sees, and `pid_water` alone pins
recirculation at 0.7 and rarely moves the plant. `configs/collection.yaml`
mixes six behaviours (`hvacdata/rollouts.py: BEHAVIOURS`):

| behaviour | what it adds |
|---|---|
| `pid_water` | the best baseline, as is |
| `pid_water_explore` | small noise held for an hour; AHUs flipped rarely, the pump never (flipping it tripled the loss in the cold) |
| `pid_water_plant_random` | half the hours, plant setpoints redrawn at random within sane ranges while zone loops stay closed |
| `pid`, `fixed_setpoint`, `bang_bang` | baselines that fail differently (cold vs heat) |

```bash
python scripts/plan_collection.py                       # data/collection_plan.csv (160 x 7-day episodes)
python scripts/collect_plan.py                          # resumable; all cores
python scripts/plan_collection.py --config configs/collection_val.yaml --out data/collection_plan_val.csv
python scripts/collect_plan.py --plan data/collection_plan_val.csv
```

What was collected (`data/collection_plan.csv`): 117 train episodes on all 40
train buildings and 20 climates, 9.3 M zone-steps, plus 12 validation episodes
with varied controllers (`data/collection_plan_val.csv`). Simulation cost
grows with floor area, so buildings over 90 zones got 2 weather windows and
the rest 3, and episodes are 3.5 days (the first 8 ran 7). That took about
1.5 hours on 4 cores; raise `episodes_per_building` or `steps` on a bigger
machine. `data/collection_plan_full.csv` is the original 160-episode plan.

`analysis/feature_eval.py` measures what each choice buys, scored on unseen
buildings in unseen climates run both by `pid_water` and by varied
controllers:

- **Varied data scales; shipped-only data levels off.** Next-hour temperature
  R² on varied-run buildings: 0.535 → 0.555 → 0.567 for shipped `pid_water`
  data at 20k / 80k / 320k rows, against 0.563 → 0.593 → 0.610 for the varied
  collection.
- **Encoding v2 helps the neural network** (+0.015 to +0.045 R², largest for
  comfort penalty under varied controllers, 0.571 → 0.616); gradient boosting
  is indifferent to it, as trees are to monotone transforms and scaling.
- **Feature groups:** recent dynamics matter most; the action columns matter
  on varied-run buildings; static building descriptors lean slightly negative
  (possible memorisation of train buildings). A refit with another seed moves
  R² by about 0.005, so differences under 0.01 are noise.

## Analysis

`analysis/static_analysis.py` (weather and buildings), then
`analysis/rollout_analysis.py <tag>` (encoding audit, reward anatomy, where
comfort is lost, a next-hour temperature model raw vs re-encoded, shift between
splits), then `analysis/export_dashboard.py` writes the report's data.

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
