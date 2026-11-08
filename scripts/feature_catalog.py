#!/usr/bin/env python3
"""Write docs/FEATURES.md: every engineered column, its treatment and its measured behaviour.

The definitions come from hvacdata/features.py (ZONE_SPEC, GLOBAL_SPEC,
DROPPED); the statistics are measured on encoded training data, so the
document shows both what each column should be and what it actually is.
Also fits the robust FeatureScaler on the same rows and saves it.

    python scripts/feature_catalog.py --tags diverse analysis
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvacdata import paths  # noqa: E402
from hvacdata.features import (DROPPED, FEATURE_VERSION, GLOBAL_SPEC, ZONE_SPEC,  # noqa: E402
                               FeatureScaler)

ACTIONS = [
    ("a_damper", "fraction above floor", "(cmd - 0.1) / 0.9, clip 0-1", "Commanded damper; the simulator floors it at 10%."),
    ("a_reheat", "fraction", "cmd", "Commanded reheat valve."),
    ("a_ahu_run", "0/1", "cmd", "Commanded AHU run."),
    ("a_ahu_sat", "5 K", "(K - 20 C) / 5", "Commanded supply air temperature, 285-305 K."),
    ("a_ahu_pressure", "fraction", "Pa / 2000", "Commanded static pressure; the simulator holds a floor while running."),
    ("a_ahu_recirc", "fraction of cap", "min(cmd, 0.7) / 0.7", "Commanded recirculation; above 0.7 has no effect."),
    ("a_water_temp", "20 K", "(K - 330) / 20", "Commanded supply water temperature, 310-350 K."),
    ("a_water_dp", "fraction", "dp / 20", "Commanded pump differential pressure."),
    ("a_pump_run", "0/1", "cmd", "Commanded pump run."),
]
TARGETS = [
    ("y_dtemp", "K", "T[t+1] - T[t]", "Next-step temperature change (5 min)."),
    ("y_dtemp_1h", "K", "T[t+12] - T[t]", "Temperature change over the next hour."),
    ("y_shortfall", "reward units", "shortfall at t+1", "The reward's per-zone comfort penalty one step ahead; log1p it as a model target."),
]


def stats(v: np.ndarray, clip, unit: str = "") -> dict:
    v = v[np.isfinite(v)]
    vals, counts = np.unique(np.round(v[:200_000], 6), return_counts=True)
    out = {"p01": float(np.percentile(v, 1)), "p50": float(np.percentile(v, 50)),
           "p99": float(np.percentile(v, 99)), "std": float(v.std()),
           "zero": float((v == 0).mean()), "top_share": float(counts.max() / counts.sum()),
           "n_unique": int(len(vals))}
    if clip:
        # Only bounds that saturate a value count: 0, and 1 for a fraction, are the
        # column's natural limits (a hinge at zero, a fully open damper), not clipping.
        natural = {0.0} | ({1.0} if "fraction" in unit else set())
        hit = np.zeros(len(v), bool)
        if clip[0] not in natural:
            hit |= v <= clip[0] + 1e-6
        if clip[1] not in natural:
            hit |= v >= clip[1] - 1e-6
        out["at_clip"] = float(hit.mean())
    return out


def flag(st: dict, corr_partner: str | None) -> str:
    notes = []
    if st["std"] < 1e-3 or st["top_share"] > 0.995:
        notes.append("near-constant here")
    if st.get("at_clip", 0) > 0.005:
        notes.append(f"{st['at_clip']:.0%} at clip")
    if corr_partner:
        notes.append(f"r>0.95 with {corr_partner}")
    return "; ".join(notes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", default=["diverse", "analysis"])
    ap.add_argument("--rows", type=int, default=400_000)
    args = ap.parse_args()

    zs, ss = [], []
    for tag in args.tags:
        d = paths.DATA_ROOT / "features" / tag
        if not (d / "zone").exists():
            continue
        z = pd.read_parquet(d / "zone", filters=[("split", "!=", "val")])
        zs.append(z.sample(min(args.rows, len(z)), random_state=0))
        ss.append(pd.read_parquet(d / "step", filters=[("split", "!=", "val")]))
    if not zs:
        sys.exit("no encoded training data found")
    z, s = pd.concat(zs), pd.concat(ss)
    print(f"measuring on {len(z):,} zone rows and {len(s):,} step rows from {args.tags}")

    zcols = [f"f_{f.name}" for f in ZONE_SPEC]
    gcols = [f"g_{f.name}" for f in GLOBAL_SPEC]
    acols = [a[0] for a in ACTIONS]
    corr_z = z[zcols + acols].corr().abs()
    corr_g = s[gcols].corr().abs()

    def partner(c, C):
        if c not in C:
            return None
        r = C[c].drop(c).dropna()
        return r.idxmax().split("_", 1)[1] if len(r) and r.max() > 0.95 else None

    rows = []
    for spec, cols, df, C in ((ZONE_SPEC, zcols, z, corr_z), (GLOBAL_SPEC, gcols, s, corr_g)):
        for f, c in zip(spec, cols):
            st = stats(df[c].to_numpy(np.float64), f.clip, f.unit)
            rows.append({"column": c, "level": "zone" if c.startswith("f_") else "building", "group": f.group,
                         "unit": f.unit, "transform": f.transform,
                         "clip": f"{f.clip[0]:g} to {f.clip[1]:g}" if f.clip else "none (0/1 or bounded)",
                         "why": f.why, "source": f.source, **st, "flags": flag(st, partner(c, C))})
    for c, unit, tr, why in ACTIONS:
        st = stats(z[c].to_numpy(np.float64), None)
        rows.append({"column": c, "level": "action", "group": "action (commanded)", "unit": unit, "transform": tr,
                     "clip": "", "why": why, "source": "commanded action", **st, "flags": flag(st, partner(c, corr_z))})
    for c, unit, tr, why in TARGETS:
        st = stats(z[c].to_numpy(np.float64), None)
        rows.append({"column": c, "level": "target", "group": "target (t+1)", "unit": unit, "transform": tr,
                     "clip": "", "why": why, "source": "future zone temperature", **st, "flags": ""})
    cat = pd.DataFrame(rows)

    out_json = paths.REPO_ROOT / "analysis" / "out" / "feature_catalog.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(cat.to_json(orient="records", double_precision=4))

    scaler = FeatureScaler().fit(pd.concat([z[zcols + acols].reset_index(drop=True)], axis=1), zcols + acols)
    scaler_g = FeatureScaler().fit(s, gcols)
    sc = {"feature_version": FEATURE_VERSION, "fitted_on": args.tags, "split": "train (no val)",
          "center": {**scaler.center, **scaler_g.center}, "scale": {**scaler.scale, **scaler_g.scale}}
    (paths.REPO_ROOT / "configs" / "feature_scaler.json").write_text(json.dumps(sc, indent=1))

    md = [f"# Feature catalog (version {FEATURE_VERSION})", "",
          "Generated by `scripts/feature_catalog.py` from the definitions in `hvacdata/features.py`",
          f"and measured on {len(z):,} training zone rows / {len(s):,} step rows ({', '.join(args.tags)}; validation excluded).",
          "", "Statistics are after the transform and clip. `zero` is the share of exact zeros;",
          "`flags` marks what to watch: near-constant in this data, often at its clip, or nearly a duplicate.",
          "The robust scaler fitted on these rows is `configs/feature_scaler.json`.", ""]
    for level, title in (("zone", "Per-zone features"), ("building", "Building-wide features"),
                         ("action", "Actions (commanded at t)"), ("target", "Targets")):
        md += [f"## {title}", "", "| column | group | transform | clip | p1 / p50 / p99 | zero | flags | why |",
               "|---|---|---|---|---|---|---|---|"]
        for r in cat[cat.level == level].itertuples():
            md.append(f"| `{r.column}` | {r.group} | {r.transform} | {r.clip} | "
                      f"{r.p01:.3g} / {r.p50:.3g} / {r.p99:.3g} | {r.zero:.0%} | {r.flags} | {r.why} |")
        md.append("")
    md += ["## Raw fields not used", "", "| raw field | why |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in DROPPED.items()]
    (paths.REPO_ROOT / "docs").mkdir(exist_ok=True)
    (paths.REPO_ROOT / "docs" / "FEATURES.md").write_text("\n".join(md) + "\n")
    print("wrote docs/FEATURES.md, configs/feature_scaler.json, analysis/out/feature_catalog.json")
    print(cat[cat["flags"] != ""][["column", "flags"]].to_string())


if __name__ == "__main__":
    main()
