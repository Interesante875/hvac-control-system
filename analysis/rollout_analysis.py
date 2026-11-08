"""Analysis of recorded rollouts and of the feature encoding.

Reads data/rollouts/<tag> (raw) and data/features/<tag> (encoded, built by
scripts/build_features.py) and writes analysis/out/rollouts.json for the report.

Sections
  audit       raw observation fields: scale, how the env encodes them, what we do
  reward      comfort / cost / carbon by station and policy; occupied vs not
  comfort     where comfort is lost: hour of day, worst zones, facade exposure
  recon       per-zone comfort reconstruction vs the reward's recorded term
  model       does the encoding help? next-hour zone temperature, train on the
              kit buildings, test on unseen val buildings in unseen climates
  shift       kit vs val distribution shift per encoded feature
"""

from __future__ import annotations

import builtins
import functools
import json
import re
import sys
from pathlib import Path

print = functools.partial(builtins.print, flush=True)  # noqa: A001  (progress shows when piped)

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hvacdata import paths  # noqa: E402
from hvacdata.dataset import load_rollout  # noqa: E402
from hvacdata.features import GLOBAL_FEATURES, ZONE_FEATURES  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "analysis"
FEAT = paths.DATA_ROOT / "features" / TAG
OUT = ROOT / "analysis" / "out"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
res: dict = {}

step = pd.read_parquet(FEAT / "step")
step["noisy"] = step["sigma"] > 0
step["run"] = step["policy"] + np.where(step["noisy"], "+noise", "")
print("step rows", len(step), step.groupby(["split", "run"]).building.nunique().to_dict())

# ------------------------------------------------------------------ audit
def kind(n: str) -> str:
    n = re.sub(r"^VAV_\d+_", "zone: ", n)
    n = re.sub(r"^ahs_ahu_\d+_", "ahu: ", n)
    n = re.sub(r"^hws_", "water: ", n)
    return re.sub(r"_\d{3}$", "_*", n)

raw_paths = sorted((paths.ROLLOUTS_DIR / TAG).rglob("pid_water.npz"))
groups: dict[str, list[np.ndarray]] = {}
dims = []
for p in raw_paths:
    r = load_rollout(p)
    names = r["meta"]["observation_field_names"]
    dims.append({"building": r["summary"]["building"], "obs_dim": len(names),
                 "n_zones": r["summary"]["n_zones"], "n_ahus": r["summary"]["n_ahus"]})
    O = r["obs"][:: 6]
    for k in set(map(kind, names)):
        idx = [i for i, n in enumerate(names) if kind(n) == k]
        groups.setdefault(k, []).append(O[:, idx].ravel())

NORMALISED = {"zone: reheat_valve_percentage_command", "zone: supply_air_damper_percentage_command",
              "zone: zone_air_temperature_sensor", "water: differential_pressure_setpoint",
              "water: run_status", "water: supervisor_run_command"}
DECISION = {  # what the encoder does with each raw group
    "zone: zone_air_temperature_sensor": ("re-encoded", "de-normalised to K; expressed vs the zone's own band"),
    "zone: zone_air_heating_temperature_setpoint_sensor": ("re-encoded", "raw K; becomes band geometry"),
    "zone: zone_air_cooling_temperature_setpoint_sensor": ("re-encoded", "raw K; becomes band geometry"),
    "zone: supply_air_damper_percentage_command": ("re-encoded", "to fraction 0-1"),
    "zone: reheat_valve_percentage_command": ("re-encoded", "to fraction 0-1"),
    "zone: supply_air_flowrate_setpoint": ("dropped", "constant"),
    "ahu: supply_air_temperature_sensor": ("re-encoded", "raw K; vs 20 C and vs each zone"),
    "ahu: supply_air_temperature_setpoint": ("dropped", "identical to sensor"),
    "ahu: supply_air_static_pressure_sensor": ("re-encoded", "raw Pa; /2000"),
    "ahu: supply_air_static_pressure_setpoint": ("dropped", "identical to sensor"),
    "ahu: supply_air_flowrate_sensor": ("re-encoded", "raw m3/s; per zone served"),
    "ahu: supply_air_flowrate_setpoint": ("dropped", "identical to sensor"),
    "ahu: outside_air_flowrate_sensor": ("dropped", "follows flow x (1 - recirculation)"),
    "ahu: outside_air_temperature_sensor": ("re-encoded", "raw K, repeated per AHU; kept once"),
    "ahu: recirculation_percentage_command": ("kept", "already a fraction"),
    "ahu: supply_fan_run_command": ("kept", "0/1"),
    "ahu: supervisor_run_command": ("dropped", "identical to supply fan run"),
    "ahu: exhaust_fan_run_command": ("dropped", "identical to supply fan run"),
    "ahu: supply_fan_speed_percentage_command": ("dropped", "follows static pressure"),
    "ahu: discharge_fan_speed_percentage_command": ("dropped", "identical to supply fan speed"),
    "ahu: cooling_request_count": ("re-encoded", "equals zones served when on; per zone served"),
    "water: supply_water_temperature_sensor": ("re-encoded", "raw K; (K-330)/20"),
    "water: supply_water_temperature_setpoint": ("dropped", "tracked by sensor"),
    "water: differential_pressure_sensor": ("dropped", "raw; setpoint kept"),
    "water: differential_pressure_setpoint": ("re-encoded", "de-normalised, /20"),
    "water: heating_request_count": ("dropped", "equals number of zones"),
    "water: run_status": ("re-encoded", "de-normalised to 0/1"),
    "water: supervisor_run_command": ("dropped", "identical to run status"),
    "hod_cos_*": ("re-encoded", "12 phase-shifted copies of one 24 h cycle; keep 1 pair"),
    "hod_sin_*": ("re-encoded", "12 phase-shifted copies of one 24 h cycle; keep 1 pair"),
    "dow_cos_*": ("re-encoded", "12 phase-shifted copies of one weekly cycle; keep 1 pair"),
    "dow_sin_*": ("re-encoded", "12 phase-shifted copies of one weekly cycle; keep 1 pair"),
    "comfort_mode_now": ("kept", "0/1"),
    "comfort_mode_soon": ("kept", "0/1; plus hours until comfort starts"),
    "num_occupants": ("re-encoded", "raw count grows with building; per zone"),
}
audit = []
for k, vs in sorted(groups.items()):
    v = np.concatenate(vs)
    dec, why = DECISION.get(k, ("kept", ""))
    audit.append({"field": k, "env_encoding": "z-scored" if k in NORMALISED else "raw (passed through)",
                  "min": float(v.min()), "max": float(v.max()), "std": float(v.std()),
                  "n_unique_le2": bool(len(np.unique(v[:5000])) <= 2), "decision": dec, "why": why})
res["audit"] = audit
res["dims"] = dims
res["n_encoded"] = {"zone": len(ZONE_FEATURES), "global": len(GLOBAL_FEATURES)}

# ------------------------------------------------------------------ reward
clean = step[~step["noisy"]]
agg = (step.groupby(["run", "split", "station"])
       [["comfort_term", "cost_term", "carbon_term", "reward"]].mean().reset_index())
res["reward_by_station"] = agg.round(5).to_dict("records")
occ = step.assign(occupied=step["g_occupied"] > 0.5).groupby(["run", "occupied"])[
    ["comfort_term", "cost_term", "carbon_term"]].sum()
res["reward_by_occupancy"] = {f"{r}|{o}": v for (r, o), v in occ.round(2).T.to_dict().items()}
res["occupied_share_of_steps"] = float((clean["g_occupied"] > 0.5).mean())
per_run = (step.groupby(["run", "split", "building", "weather", "station"])
           [["reward", "comfort_term", "cost_term", "carbon_term"]].mean().reset_index())
per_run["score"] = -per_run["reward"]
res["per_rollout"] = per_run.round(5).to_dict("records")

# ------------------------------------------------------------------ comfort
pw = step[step["run"] == "pid_water"]
hod = (pw.assign(hour=pd.to_datetime(pw["timestamp"], unit="s").dt.hour)
       .groupby(["station", "hour"])[["comfort_term", "cost_term", "carbon_term"]].mean().reset_index())
res["by_hour"] = hod.round(5).to_dict("records")

zcols = ["building", "weather", "station", "split", "zone", "t", "f_comfort_shortfall",
         "f_exterior_fraction", "f_log_area", "y_shortfall"]
zone = pd.read_parquet(FEAT / "zone", columns=zcols + ["policy", "sigma"],
                       filters=[("policy", "==", "pid_water"), ("sigma", "==", 0.0)])
occ_t = pw[["building", "weather", "t", "g_occupied"]]
zone = zone.merge(occ_t, on=["building", "weather", "t"])
zo = zone[zone["g_occupied"] > 0.5]
per_zone = zo.groupby(["building", "weather", "station", "split", "zone"]).agg(
    shortfall=("y_shortfall", "mean"), ext=("f_exterior_fraction", "first"),
    log_area=("f_log_area", "first")).reset_index()
conc = []
for (b, w), g in per_zone.groupby(["building", "weather"]):
    s = np.sort(g["shortfall"].values)[::-1]
    tot = s.sum()
    if tot <= 0:
        continue
    n10 = max(1, int(np.ceil(0.1 * len(s))))
    conc.append({"building": b, "weather": w, "station": g["station"].iloc[0],
                 "top10_share": float(s[:n10].sum() / tot), "n_zones": len(s)})
res["concentration"] = conc
per_zone["ext_bin"] = pd.cut(per_zone["ext"], [-0.01, 0.0, 0.2, 0.35, 0.6],
                             labels=["interior", "0-0.2", "0.2-0.35", ">0.35"])
res["by_exposure"] = (per_zone.groupby(["station", "ext_bin"], observed=True)["shortfall"]
                      .agg(["mean", "count"]).reset_index().round(5).to_dict("records"))
res["zone_scatter"] = per_zone.sample(min(1500, len(per_zone)), random_state=0)[
    ["station", "ext", "log_area", "shortfall"]].round(4).to_dict("records")

# ------------------------------------------------------------------ recon
rec = []
for (b, w), g in zone.groupby(["building", "weather"]):
    est = -(g.groupby("t")["y_shortfall"].mean() * g.groupby("t")["g_occupied"].first())
    true = pw[(pw.building == b) & (pw.weather == w)].set_index("t")["comfort_term"] / 0.6
    j = pd.concat([est.rename("est"), true.rename("true")], axis=1).dropna()
    if j["true"].abs().sum() == 0:
        continue
    rec.append({"building": b, "weather": w, "station": g["station"].iloc[0],
                "corr": float(j.corr().iloc[0, 1]), "ratio": float(j["est"].sum() / j["true"].sum())})
res["recon"] = rec
del zone, zo

import os  # noqa: E402
DEBUG = os.environ.get("HV_DEBUG_HOLDOUT")  # e.g. building_train_8: test on a held-out kit building
if not DEBUG and not (step["split"] == "val").any():
    (OUT / "rollouts.json").write_text(json.dumps(res, default=float))
    sys.exit("no val rollouts yet: wrote the non-model sections only")

# ------------------------------------------------------------------ model
# Target: zone temperature change over the next hour. Train: kit buildings
# (pid_water, clean + noisy). Test: val buildings in val climates (pid_water).
FCOLS = [f"f_{c}" for c in ZONE_FEATURES]
ACOLS = ["a_damper", "a_reheat", "a_ahu_run", "a_ahu_sat_k"]
GCOLS = [f"g_{c}" for c in GLOBAL_FEATURES]


def load_zone(split: str, n: int) -> pd.DataFrame:
    filt = [("policy", "==", "pid_water")]
    if DEBUG:
        filt += [("split", "==", "kit"), ("building", "==" if split == "val" else "!=", DEBUG)]
        if split == "val":
            filt += [("sigma", "==", 0.0)]
    else:
        filt += [("split", "==", split)] + ([("sigma", "==", 0.0)] if split == "val" else [])
    z = pd.read_parquet(FEAT / "zone", filters=filt,
                        columns=["building", "weather", "t", "zone", "sigma", *FCOLS, *ACOLS,
                                 "y_dtemp_1h", "y_dtemp"])
    z = z[z["t"] >= 12]  # need an hour of history for the lag features
    z = z[z["t"] < z.groupby(["building", "weather", "sigma"])["t"].transform("max") - 12]
    z = z.sample(min(n, len(z)), random_state=0)
    g = step[["building", "weather", "sigma", "t", *GCOLS]]
    return z.merge(g, on=["building", "weather", "sigma", "t"])


def raw_rows(z: pd.DataFrame) -> np.ndarray:
    """The same rows, described by the env's raw (un-re-encoded) fields."""
    out = np.zeros((len(z), 6 + 15 + 7 + 51), np.float32)  # zone, AHU, water, building-wide
    for (b, w, s), g in z.groupby(["building", "weather", "sigma"]):
        stem = "pid_water" + (f"_s{s:g}_f0.05" if s > 0 else "")
        r = load_rollout(paths.ROLLOUTS_DIR / TAG / b / w / f"{stem}.npz")
        names = r["meta"]["observation_field_names"]
        ix = {n: i for i, n in enumerate(names)}
        zids = list(r["zone_ids"])
        z2a = {room.replace("room_", "VAV_"): k for k, rooms in r["meta"]["ahu_zone_assignments"].items()
               for room in rooms}
        glob_idx = [i for i, n in enumerate(names) if not n.startswith(("VAV_", "ahs_", "hws_"))]
        hws_idx = [i for i, n in enumerate(names) if n.startswith("hws_")]
        rows = np.flatnonzero(((z["building"] == b) & (z["weather"] == w) & (z["sigma"] == s)).values)
        for row, (t, zi) in zip(rows, g[["t", "zone"]].values):
            zid = zids[zi]
            vi = sorted(i for n, i in ix.items() if n.startswith(zid + "_"))
            ai = sorted(i for n, i in ix.items() if n.startswith(f"ahs_{z2a[zid]}_"))
            o = r["obs"][t]
            out[row] = np.concatenate([o[vi], o[ai], o[hws_idx], o[glob_idx]])
    return out


tr = load_zone("kit", 80_000).reset_index(drop=True)
te = load_zone("val", 40_000).reset_index(drop=True)
print("model rows", len(tr), len(te))
Xr_tr, Xr_te = raw_rows(tr), raw_rows(te)
act_tr, act_te = tr[ACOLS].values, te[ACOLS].values
Xe_tr = tr[FCOLS + GCOLS + ACOLS].values
Xe_te = te[FCOLS + GCOLS + ACOLS].values
Xr_tr = np.hstack([Xr_tr, act_tr]); Xr_te = np.hstack([Xr_te, act_te])
y_tr, y_te = tr["y_dtemp_1h"].values, te["y_dtemp_1h"].values

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

models = {
    "Ridge": lambda: Ridge(alpha=1.0),
    "MLP": lambda: MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=40, early_stopping=True, random_state=0),
    "Gradient boosting": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0),
}
mres = []
fitted = {}
for fs, (Xa, Xb) in {"raw obs (as delivered)": (Xr_tr, Xr_te), "re-encoded": (Xe_tr, Xe_te)}.items():
    for mn, mk in models.items():
        m = mk().fit(Xa, y_tr)
        p = m.predict(Xb)
        # also in-distribution: held-out kit rows are not available separately,
        # so report train fit for reference only
        mres.append({"features": fs, "model": mn, "r2_val": float(r2_score(y_te, p)),
                     "mae_val": float(mean_absolute_error(y_te, p)),
                     "r2_train": float(r2_score(y_tr, m.predict(Xa)))})
        fitted[(fs, mn)] = m
        print(mres[-1])
res["model"] = mres
res["model_baseline_mae"] = float(np.abs(y_te - y_te.mean()).mean())
res["model_persist_mae"] = float(np.abs(y_te - te["f_dtemp_1h"].values).mean())  # "next hour = last hour"

gb = fitted[("re-encoded", "Gradient boosting")]
sub = rng.choice(len(Xe_te), 8000, replace=False)
pi = permutation_importance(gb, Xe_te[sub], y_te[sub], n_repeats=3, random_state=0, scoring="r2")
names = FCOLS + GCOLS + ACOLS
imp = sorted(zip(names, pi.importances_mean), key=lambda x: -x[1])
res["importance"] = [{"feature": n, "drop_r2": float(v)} for n, v in imp[:15]]

# ------------------------------------------------------------------ shift
sh = []
for c in FCOLS + GCOLS:
    a, b = tr[c].values, te[c].values
    sd = np.sqrt((a.var() + b.var()) / 2) or 1.0
    sh.append({"feature": c, "smd": float((b.mean() - a.mean()) / sd), "kit_mean": float(a.mean()),
               "val_mean": float(b.mean())})
res["shift"] = sorted(sh, key=lambda x: -abs(x["smd"]))

(OUT / ("rollouts_debug.json" if DEBUG else "rollouts.json")).write_text(json.dumps(res, default=float))
print("wrote", OUT / "rollouts.json")
