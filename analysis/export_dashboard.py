"""Turn analysis/out/{static,rollouts}.json into row datasets for the report.

Writes analysis/out/datasets/<id>.json (each a JSON array of row objects).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hvacdata import paths  # noqa: E402
from hvacdata.dataset import load_rollout  # noqa: E402

OUT = ROOT / "analysis" / "out" / "datasets"
OUT.mkdir(parents=True, exist_ok=True)
S = json.loads((ROOT / "analysis/out/static.json").read_text())
R = json.loads((ROOT / "analysis/out/rollouts.json").read_text())


def put(name, rows):
    rows = json.loads(pd.DataFrame(rows).to_json(orient="records", double_precision=5))
    (OUT / f"{name}.json").write_text(json.dumps(rows))
    print(f"{name:18s} {len(rows):5d} rows  {(OUT / f'{name}.json').stat().st_size / 1024:.1f} KB")


# ---- headline figures --------------------------------------------------------
audit = pd.DataFrame(R["audit"])
model = pd.DataFrame(R["model"])
gb = model[model.model == "Gradient boosting"].set_index("features")["r2_val"]
mlp = model[model.model == "MLP"].set_index("features")["r2_val"]
dims = pd.DataFrame(R["dims"]).drop_duplicates("building")
recon = pd.DataFrame(R["recon"])
headline = [{
    "field_types": len(audit),
    "field_types_raw": int((audit.env_encoding != "z-scored").sum()),
    "field_types_dropped": int((audit.decision == "dropped").sum()),
    "obs_dim_min": int(dims.obs_dim.min()), "obs_dim_max": int(dims.obs_dim.max()),
    "zone_features": R["n_encoded"]["zone"], "global_features": R["n_encoded"]["global"],
    "r2_gb_raw": float(gb["raw obs (as delivered)"]), "r2_gb_enc": float(gb["re-encoded"]),
    "r2_mlp_raw": float(mlp["raw obs (as delivered)"]), "r2_mlp_enc": float(mlp["re-encoded"]),
    "recon_ratio_median": float(recon.ratio.median()), "recon_corr_median": float(recon["corr"].median()),
    "occupied_share": R["occupied_share_of_steps"],
}]
put("headline", headline)

# ---- encoding audit ------------------------------------------------------------
put("obs_audit", audit[["field", "env_encoding", "min", "max", "std", "decision", "why"]])

# ---- weather ------------------------------------------------------------------
b = S["bins"]
put("weather_hist", [{"temp_c": (b[i] + b[i + 1]) / 2, "kit": S["hist"]["kit"][i],
                      "train": S["hist"]["train"][i], "val": S["hist"]["val"][i]}
                     for i in range(len(b) - 1)])
st = pd.DataFrame(S["strips"])
st = st[st.used]
ks = pd.DataFrame(S["kit_strips"])
put("weather_strips", pd.concat([st, ks])[["name", "station", "split", "mean_c", "min_c", "max_c", "hdh", "cdh"]])

# ---- buildings -----------------------------------------------------------------
put("buildings", S["buildings"])

# ---- reward anatomy ---------------------------------------------------------------
rs = pd.DataFrame(R["reward_by_station"])
rs = rs[rs.run == "pid_water"]
put("reward_station", rs[["split", "station", "comfort_term", "cost_term", "carbon_term", "reward"]]
    .assign(loss=lambda d: -d.reward).sort_values("loss", ascending=False))
occ = [{"occupied": o == "True", "term": term, "total": v}
       for k, d in R["reward_by_occupancy"].items() if k.startswith("pid_water|")
       for o in [k.split("|")[1]] for term, v in d.items()]
put("reward_occupancy", occ)
kit_stations = {"atlanta", "denver", "fairbanks", "phoenix", "singapore"}
put("comfort_by_hour", [d for d in R["by_hour"] if d["station"] in kit_stations])

# ---- where comfort is lost -----------------------------------------------------------
ex = pd.DataFrame(R["by_exposure"]).pivot(index="station", columns="ext_bin", values="mean").reset_index()
ex.columns.name = None
ex["split"] = np.where(ex["station"].isin(kit_stations), "kit", "val")
put("exposure", ex)
put("concentration", R["concentration"])
put("recon", R["recon"])

# ---- modelling + shift -------------------------------------------------------------
put("model", model.assign(mae_persist=R["model_persist_mae"], mae_mean=R["model_baseline_mae"]))
put("importance", R["importance"])
put("shift", R["shift"][:12])

# ---- one day of zone air temperature around AHU switching ---------------------------
r = load_rollout(paths.ROLLOUTS_DIR / "verify/building_train_8/fairbanks_8/pid_water.npz")
names = r["meta"]["observation_field_names"]
ix = {n: i for i, n in enumerate(names)}
zid = "VAV_28"
ahu = [k for k, rooms in r["meta"]["ahu_zone_assignments"].items() if "room_28" in rooms][0]
k = list(r["meta"]["ahu_zone_assignments"]).index(ahu)
t0, t1 = 0, 288
trace = []
for t in range(t0, t1):
    o = r["obs"][t + 1]
    trace.append({"time": pd.Timestamp(int(r["timestamp"][t]), unit="s").strftime("%Y-%m-%dT%H:%M"),
                  "zone_temp_c": float(o[ix[f"{zid}_zone_air_temperature_sensor"]] * 5 + 295 - 273.15),
                  "heat_sp_c": float(o[ix[f"{zid}_zone_air_heating_temperature_setpoint_sensor"]] - 273.15),
                  "cool_sp_c": float(o[ix[f"{zid}_zone_air_cooling_temperature_setpoint_sensor"]] - 273.15),
                  "ahu_on": int(r["discrete_action"][t, k])})
put("temp_trace", trace)
T = np.array([r["obs"][t][[ix[f"VAV_{i}_zone_air_temperature_sensor"] for i in range(1, 31)]] * 5
              for t in range(len(r["obs"]))])
dT = np.abs(np.diff(T, axis=0))
put("step_change", [{"p50": float(np.percentile(dT, 50)), "p90": float(np.percentile(dT, 90)),
                     "p99": float(np.percentile(dT, 99)), "max": float(dT.max())}])
