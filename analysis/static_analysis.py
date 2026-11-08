"""Weather + building population analysis (no simulation needed).

Writes analysis/out/static.json, consumed by the report.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hvacdata import paths  # noqa: E402

OUT = ROOT / "analysis" / "out"; OUT.mkdir(parents=True, exist_ok=True)
strips = pd.read_csv(paths.DATA_ROOT / "weather" / "strips.csv")
splits = __import__("yaml").safe_load(open(paths.SPLITS_FILE))

# ---- weather: hourly temps per station, and per-strip degree-hours ----------------
def station_csv(st):
    p = paths.WEATHER_CSV_DIR / f"{st}.csv"
    return p if p.exists() else paths.KIT_WEATHER / "csv" / f"{st}.csv"

hourly = {}
for st in strips.station.unique():
    df = pd.read_csv(station_csv(st))
    t = pd.to_datetime(df["Time"], format="%Y%m%d-%H%M")
    hourly[st] = pd.Series(((df["TempF"] - 32) * 5 / 9).values, index=t)

def strip_metrics(st, start):
    s = hourly[st]; t0 = pd.Timestamp(start); x = s[(s.index >= t0) & (s.index < t0 + pd.Timedelta(days=30))]
    return {"hdh": float(np.clip(18 - x, 0, None).sum() / 1000),   # heating degree-kilohours (base 18 C)
            "cdh": float(np.clip(x - 24, 0, None).sum() / 1000),   # cooling degree-kilohours (base 24 C)
            "sat_inert_frac": float((x < -6.8).mean())}
m = pd.DataFrame([strip_metrics(r.station, r.start) for r in strips.itertuples()])
strips = pd.concat([strips, m], axis=1)

used = {"kit": {e["weather"] for e in splits["kit"]}, "train": {e["weather"] for e in splits["train"]},
        "val": {e["weather"] for e in splits["val"]}}
kit_strips = []
for e in splits["kit"]:
    import yaml
    w = yaml.safe_load(open(paths.KIT_WEATHER / f"{e['weather']}.yaml"))
    st = e["station"]; mm = strip_metrics(st, w["start_timestamp"])
    s = hourly[st]; t0 = pd.Timestamp(w["start_timestamp"]); x = s[(s.index >= t0) & (s.index < t0 + pd.Timedelta(days=30))]
    kit_strips.append({"name": e["weather"], "station": st, "split": "kit", "mean_c": float(x.mean()),
                       "min_c": float(x.min()), "max_c": float(x.max()), **mm})
strip_pts = [{"name": r.name, "station": r.station, "split": ("val" if r.name in used["val"] else "train"),
              "used": bool(r.name in used["train"] or r.name in used["val"]),
              "mean_c": r.mean_c, "min_c": r.min_c, "max_c": r.max_c, "hdh": r.hdh, "cdh": r.cdh,
              "sat_inert_frac": r.sat_inert_frac} for r in strips.itertuples()]

# hourly temperature histogram of the scenarios actually used, per split
bins = np.arange(-45, 51, 2.5)
def hist_for(entries, kit=False):
    vals = []
    for e in entries:
        st = e["station"]
        start = (yaml.safe_load(open(paths.KIT_WEATHER / f"{e['weather']}.yaml"))["start_timestamp"] if kit
                 else strips.set_index("name").loc[e["weather"], "start"])
        s = hourly[st]; t0 = pd.Timestamp(start)
        vals.append(s[(s.index >= t0) & (s.index < t0 + pd.Timedelta(days=30))].values)
    v = np.concatenate(vals)
    h, _ = np.histogram(v, bins=bins)
    return (h / h.sum()).tolist(), v
hk, vk = hist_for(splits["kit"], kit=True); ht, vt = hist_for(splits["train"]); hv, vv = hist_for(splits["val"])
coverage = {"val_in_train_range": float(((vv >= vt.min()) & (vv <= vt.max())).mean()),
            "val_in_kit_range": float(((vv >= vk.min()) & (vv <= vk.max())).mean()),
            "train_p01_p99": [float(np.percentile(vt, 1)), float(np.percentile(vt, 99))],
            "kit_p01_p99": [float(np.percentile(vk, 1)), float(np.percentile(vk, 99))],
            "val_p01_p99": [float(np.percentile(vv, 1)), float(np.percentile(vv, 99))]}

# monthly mean per station
monthly = {st: s.groupby(s.index.month).mean().round(2).tolist() for st, s in hourly.items()}

# ---- buildings ------------------------------------------------------------------
from hvacdata.buildings import check_floor_plan  # noqa: E402
from smart_buildings.smart_control.utils.materials import envelope_capacity, facade_u  # noqa: E402
b = pd.read_csv(paths.BUILDINGS_DIR / "buildings.csv")
brecs = b[["name", "split", "profile", "n_zones", "n_ahus", "num_floors", "width", "height", "facade_u", "envelope_c_mj"]].to_dict("records")
for p in sorted(paths.KIT_BUILDINGS.glob("*.yaml")):
    cfg = yaml.safe_load(open(p)); nz, na = check_floor_plan(p); fp = cfg["floor_plan"]
    brecs.append({"name": p.stem, "split": "kit", "profile": "shipped", "n_zones": nz, "n_ahus": na,
                  "num_floors": fp["num_floors"], "width": fp["width"], "height": fp["height"],
                  "facade_u": round(facade_u(cfg["materials"]), 3),
                  "envelope_c_mj": round(envelope_capacity(cfg["materials"]) / 1e6, 3)})

out = {"bins": bins.tolist(), "hist": {"kit": hk, "train": ht, "val": hv}, "coverage": coverage,
       "strips": strip_pts, "kit_strips": kit_strips, "monthly": monthly,
       "station_split": {r.station: r.split for r in strips.drop_duplicates("station").itertuples()},
       "buildings": brecs,
       "counts": {k: len(v) for k, v in splits.items()}}
(OUT / "static.json").write_text(json.dumps(out))
print(json.dumps(coverage, indent=1))
bd = pd.DataFrame(brecs); print(bd.groupby("split")[["n_zones", "n_ahus", "facade_u", "envelope_c_mj"]].agg(["min", "median", "max"]).round(2).to_string())
