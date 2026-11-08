"""Datasets for the report's data-collection and feature-engineering sections.

Reads data/features/{diverse,diverse_val}, analysis/out/feature_eval.json and
analysis/out/feature_catalog.json; writes analysis/out/datasets/<id>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hvacdata import features as F2  # noqa: E402
from hvacdata import paths  # noqa: E402

OUT = ROOT / "analysis" / "out" / "datasets"
OUT.mkdir(parents=True, exist_ok=True)
FEAT = paths.DATA_ROOT / "features"
E = json.loads((ROOT / "analysis/out/feature_eval.json").read_text())
CAT = json.loads((ROOT / "analysis/out/feature_catalog.json").read_text())


def put(name, rows):
    rows = json.loads(pd.DataFrame(rows).to_json(orient="records", double_precision=4))
    (OUT / f"{name}.json").write_text(json.dumps(rows))
    print(f"{name:20s} {len(rows):5d} rows  {(OUT / f'{name}.json').stat().st_size / 1024:.1f} KB")


BNAMES = {"pid_water": "pid_water", "pid_water_explore": "pid_water + explore",
          "pid_water_plant_random": "pid_water + random plant", "pid": "pid",
          "fixed_setpoint": "fixed_setpoint", "bang_bang": "bang_bang"}

# ---- what was collected ------------------------------------------------------------
steps = []
for tag in ("diverse", "diverse_val"):
    for f in sorted((FEAT / tag / "step").glob("*.parquet")):
        s = pd.read_parquet(f, columns=["building", "weather", "station", "behaviour", "n_zones", "reward",
                                         "comfort_term", "cost_term", "carbon_term"])
        steps.append({"set": "train" if tag == "diverse" else "validation",
                      "building": s.building.iloc[0], "weather": s.weather.iloc[0],
                      "station": s.station.iloc[0], "behaviour": s.behaviour.iloc[0],
                      "n_zones": int(s.n_zones.iloc[0]), "steps": len(s),
                      "zone_rows": int(len(s) * s.n_zones.iloc[0]), "loss": float(-s.reward.mean())})
ep = pd.DataFrame(steps)
tr = ep[ep.set == "train"]
put("collection_headline", [{
    "episodes": int(len(tr)), "buildings": int(tr.building.nunique()), "climates": int(tr.station.nunique()),
    "behaviours": int(tr.behaviour.nunique()), "zone_rows_m": float(tr.zone_rows.sum() / 1e6),
    "days": float(tr.steps.sum() / 288), "val_episodes": int((ep.set == "validation").sum()),
    "zone_features": len(F2.ZONE_FEATURES), "global_features": len(F2.GLOBAL_FEATURES),
}])
summ = (tr.groupby("behaviour").agg(episodes=("building", "size"), buildings=("building", "nunique"),
                                     climates=("station", "nunique"), zone_rows=("zone_rows", "sum"))
        .reset_index())
summ["zone_rows_m"] = summ.zone_rows / 1e6
summ["label"] = summ.behaviour.map(BNAMES)
put("collection_behaviours", summ[["behaviour", "label", "episodes", "buildings", "climates", "zone_rows_m"]])
mat = tr.pivot_table(index="station", columns="behaviour", values="building", aggfunc="size", fill_value=0).reset_index()
mat.columns.name = None
for b in BNAMES:
    if b not in mat:
        mat[b] = 0
put("collection_matrix", mat[["station", *BNAMES]])

# ---- action coverage ------------------------------------------------------------------
ANAMES = {"a_ahu_sat": "Supply air temperature", "a_ahu_pressure": "Fan static pressure",
          "a_ahu_recirc": "Recirculation", "a_water_temp": "Supply water temperature",
          "a_water_dp": "Pump pressure", "a_damper": "Zone damper", "a_reheat": "Zone reheat",
          "a_ahu_run": "AHU on", "a_pump_run": "Pump on"}
cov = pd.DataFrame(E["coverage"])
cov = cov[cov.action.isin(["a_ahu_sat", "a_ahu_pressure", "a_ahu_recirc", "a_water_temp", "a_water_dp",
                           "a_damper", "a_reheat"])]
cov["label"] = cov.action.map(ANAMES)
cov["key"] = cov.action + "|" + cov.source
put("action_coverage", cov[["key", "action", "label", "source", "p05", "p25", "p50", "p75", "p95", "std"]])

# ---- evaluations --------------------------------------------------------------------------
TNAME = {"y_dtemp_1h": "Temperature in 1 h", "y_shortfall_1h": "Comfort penalty in 1 h"}
TEST = {"pid_water": "Run by pid_water", "varied": "Run by varied controllers"}
enc = pd.DataFrame(E["encoding"])
enc["target_label"], enc["test_label"] = enc.target.map(TNAME), enc.test.map(TEST)
enc["key"] = enc.encoding + "|" + enc.target + "|" + enc.test
put("encoding_eval", enc[["key", "encoding", "target", "target_label", "test", "test_label", "gb", "mlp", "n_features"]])
abl = pd.DataFrame(E["ablation"])
abl = abl[abl.target == "y_dtemp_1h"].pivot_table(index=["group", "n"], columns="test", values="drop_r2").reset_index()
abl.columns.name = None
abl = abl.rename(columns={"pid_water": "drop_pid_water", "varied": "drop_varied"}).sort_values("drop_varied", ascending=False)
put("ablation_eval", abl)
dat = pd.DataFrame(E["data"])
dat["target_label"], dat["test_label"] = dat.target.map(TNAME), dat.test.map(TEST)
dat["key"] = dat.train + "|" + dat.target + "|" + dat.test
put("data_eval", dat[["key", "train", "target", "target_label", "test", "test_label", "r2", "rows", "buildings", "behaviours"]])

# ---- the catalog ------------------------------------------------------------------------------
cat = pd.DataFrame(CAT)
cat["column"] = cat["column"].astype(str)
cat["name"] = cat["column"].str.replace(r"^[fga]_", "", regex=True).where(~cat["column"].str.startswith("y_"), cat["column"])
put("feature_catalog", cat[["column", "name", "level", "group", "unit", "transform", "clip", "p01", "p50", "p99",
                            "zero", "flags", "why"]])

# ---- how each data source scales -----------------------------------------------------------
sc = pd.DataFrame(json.loads((ROOT / "analysis/out/scale_eval.json").read_text()))
sc["key"] = sc["train"] + "|" + sc["rows"].astype(str) + "|" + sc["test"]
put("scale_eval", sc[["key", "train", "rows", "test", "r2"]])
