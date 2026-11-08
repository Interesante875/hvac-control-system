"""Do the feature-engineering choices and the varied data actually help?

Task: predict each zone's temperature change over the next hour (y_dtemp_1h)
and its comfort shortfall in one hour, on the validation split (unseen
buildings in unseen climates). Every comparison uses the same target rows.

Two test sets, both unseen buildings in unseen climates: the validation
split run by pid_water, and the same buildings run by varied controllers
(diverse_val), which asks whether a model predicts the effect of actions the
baseline would not take: what a controller built on it needs.

  1. Encoding:  v1 vs v2 features, both trained on the varied collection.
  2. Groups:    v2 with one feature group removed at a time (ablation).
  3. Data:      v2 trained on shipped pid_water data vs the varied collection.

Writes analysis/out/feature_eval.json. Models: gradient boosting, and a neural
network on FeatureScaler-standardised inputs (fitted on train rows only).

    python analysis/feature_eval.py [--diverse-tag diverse]
"""

from __future__ import annotations

import argparse
import builtins
import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hvacdata import features as F2  # noqa: E402
from hvacdata import features_v1 as F1  # noqa: E402
from hvacdata import paths  # noqa: E402

print = functools.partial(builtins.print, flush=True)  # noqa: A001
FEAT = paths.DATA_ROOT / "features"
OUT = ROOT / "analysis" / "out"
OUT.mkdir(parents=True, exist_ok=True)
KEY = ["building", "weather", "behaviour", "t", "zone"]
A_V2 = ["a_damper", "a_reheat", "a_ahu_run", "a_ahu_sat", "a_ahu_pressure", "a_ahu_recirc",
        "a_water_temp", "a_water_dp", "a_pump_run"]
A_V1 = A_V2  # both encodings are built with the same action columns; only the state encoding differs
Y = ["y_dtemp_1h", "y_shortfall_1h"]


def _one(tag: str, stem: str) -> pd.DataFrame:
    z = pd.read_parquet(FEAT / tag / "zone" / f"{stem}.parquet")
    st = pd.read_parquet(FEAT / tag / "step" / f"{stem}.parquet")
    if "behaviour" not in z.columns:  # v1 tables predate the column
        z["behaviour"] = z["policy"] + np.where(z["sigma"] > 0, "_noise", "")
    gcols = [c for c in st.columns if c.startswith("g_")]
    z = z.merge(st[["t", *gcols]], on="t")
    # 1-hour-ahead shortfall, from the same zone 12 steps later (y_shortfall is already t+1)
    z = z.sort_values(["zone", "t"])
    z["y_shortfall_1h"] = np.log1p(z.groupby("zone")["y_shortfall"].shift(-11))
    return z[z["t"] >= 12].dropna(subset=["y_shortfall_1h"])


def load(tag: str, n: int, split: str | None = None, behaviours: list[str] | None = None,
         seed: int = 0, align: str | None = None) -> pd.DataFrame:
    """About ``n`` rows spread evenly over the tag's episodes, loaded one file at a time.

    With ``align`` (another tag), the same (t, zone) rows are taken from it too,
    and both frames come back as a pair.
    """
    stems = sorted(p.stem for p in (FEAT / tag / "step").glob("*.parquet"))
    keep = []
    for stem in stems:
        head = pd.read_parquet(FEAT / tag / "step" / f"{stem}.parquet", columns=["split", "behaviour"]).iloc[0]
        if split and head["split"] != split:
            continue
        if behaviours and head["behaviour"] not in behaviours:
            continue
        keep.append(stem)
    per = max(1, n // max(1, len(keep)))
    out, out_b = [], []
    for i, stem in enumerate(keep):
        z = _one(tag, stem)
        z = z.sample(min(per, len(z)), random_state=seed + i)
        out.append(z)
        if align:
            zb = _one(align, stem)
            out_b.append(zb.merge(z[["t", "zone"]], on=["t", "zone"]).set_index(["t", "zone"])
                         .loc[list(zip(z["t"], z["zone"]))].reset_index())
    a = pd.concat(out, ignore_index=True)
    return (a, pd.concat(out_b, ignore_index=True)) if align else a


def sample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    return df.sample(min(n, len(df)), random_state=seed)


def fit(X, y, kind: str):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    if kind == "gb":
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0)
    else:
        m = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=40, early_stopping=True, random_state=0)
    return m.fit(X, y)


def r2(m, X, y) -> float:
    from sklearn.metrics import r2_score
    return float(r2_score(y, m.predict(X)))


def scaled(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]):
    sc = F2.FeatureScaler().fit(tr, cols)
    return sc.transform(tr, cols)[cols].to_numpy(), sc.transform(te, cols)[cols].to_numpy(), sc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=80_000)
    ap.add_argument("--n-test", type=int, default=30_000)
    args = ap.parse_args()
    res: dict = {"targets": Y}
    v2_cols = [f"f_{c}" for c in F2.ZONE_FEATURES] + [f"g_{c}" for c in F2.GLOBAL_FEATURES] + A_V2
    v1_cols = [f"f_{c}" for c in F1.ZONE_FEATURES] + [f"g_{c}" for c in F1.GLOBAL_FEATURES] + A_V1

    div2, div1 = load("diverse", args.n_train, seed=2, align="diverse_v1")
    kit2 = load("analysis", args.n_train, "kit", ["pid_water", "pid_water_noise"], 0)
    tests = {"pid_water": load("analysis", args.n_test, "val", ["pid_water"], 1, align="analysis_v1"),
             "varied": load("diverse_val", args.n_test, seed=5, align="diverse_val_v1")}
    res["rows"] = {"train_varied": len(div2), "train_shipped": len(kit2),
                   **{f"test_{k}": len(v[0]) for k, v in tests.items()}}
    res["episodes"] = {t: len(list((FEAT / t / "step").glob("*.parquet"))) for t in ("diverse", "diverse_val")}
    print(res["rows"], res["episodes"])

    # ---------------------------------------------------------------- 1. encoding
    enc = []
    for name, tr, cols, ti in (("v1", div1, v1_cols, 1), ("v2", div2, v2_cols, 0)):
        for y in Y:
            yt = div2[y].to_numpy()  # same target rows for both encodings
            gb = fit(tr[cols].to_numpy(), yt, "gb")
            sc = F2.FeatureScaler().fit(tr, cols)
            nn = fit(sc.transform(tr, cols)[cols].to_numpy(), yt, "mlp")
            for tname, pair in tests.items():
                te, ye = pair[ti], pair[0][y].to_numpy()
                row = {"encoding": name, "target": y, "test": tname, "n_features": len(cols),
                       "gb": r2(gb, te[cols].to_numpy(), ye),
                       "mlp": r2(nn, sc.transform(te, cols)[cols].to_numpy(), ye)}
                enc.append(row)
                print(row)
    res["encoding"] = enc

    # ---------------------------------------------------------------- 2. ablation
    groups: dict[str, list[str]] = {}
    for f in F2.ZONE_SPEC:
        groups.setdefault(f.group, []).append(f"f_{f.name}")
    for f in F2.GLOBAL_SPEC:
        groups.setdefault(f.group, []).append(f"g_{f.name}")
    groups["actions"] = A_V2
    base = {(e["target"], e["test"]): e["gb"] for e in enc if e["encoding"] == "v2"}
    abl = []
    for gname, gcols in groups.items():
        cols = [c for c in v2_cols if c not in gcols]
        for y in Y:
            m = fit(div2[cols].to_numpy(), div2[y].to_numpy(), "gb")
            for tname, pair in tests.items():
                te = pair[0]
                abl.append({"group": gname, "n": len(gcols), "target": y, "test": tname,
                            "drop_r2": base[(y, tname)] - r2(m, te[cols].to_numpy(), te[y].to_numpy())})
        print(gname, [round(a["drop_r2"], 4) for a in abl[-4:]])
    res["ablation"] = abl

    # ---------------------------------------------------------------- 3. data
    data = []
    for name, d in (("shipped, pid_water", kit2), ("varied collection", div2)):
        for y in Y:
            m = fit(d[v2_cols].to_numpy(), d[y].to_numpy(), "gb")
            for tname, pair in tests.items():
                te = pair[0]
                data.append({"train": name, "target": y, "test": tname, "rows": len(d),
                             "buildings": int(d.building.nunique()), "behaviours": int(d.behaviour.nunique()),
                             "r2": r2(m, te[v2_cols].to_numpy(), te[y].to_numpy())})
                print(data[-1])
    res["data"] = data

    # what each source covers of every action
    cov = []
    for c in A_V2:
        for name, d in (("shipped, pid_water", kit2), ("varied collection", div2)):
            v = d[c].to_numpy(np.float64)
            cov.append({"action": c, "source": name, "p05": float(np.percentile(v, 5)),
                        "p25": float(np.percentile(v, 25)), "p50": float(np.percentile(v, 50)),
                        "p75": float(np.percentile(v, 75)), "p95": float(np.percentile(v, 95)),
                        "std": float(v.std())})
    res["coverage"] = cov
    (OUT / "feature_eval.json").write_text(json.dumps(res, default=float))
    print("wrote", OUT / "feature_eval.json")


if __name__ == "__main__":
    main()
