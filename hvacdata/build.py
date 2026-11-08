"""Turn recorded rollouts into encoded, analysis-ready Parquet tables.

For every rollout ``data/rollouts/<tag>/<b>/<w>/<policy>.npz`` this writes

    data/features/<tag>/zone/<b>__<w>__<policy>.parquet   one row per (t, zone)
    data/features/<tag>/step/<b>__<w>__<policy>.parquet   one row per t

Zone rows hold the encoded state at t (``f_*``, see features.ZONE_FEATURES),
the zone's commanded action at t in physical units, and targets observed at
t+1 (temperature change, comfort shortfall). Step rows hold the global
features (``g_*``), the plant commands in native units, the reward and its
components, and building aggregates of the zone state. Identifiers (building,
weather, station, split, policy, noise) are columns on both, so tables from
many rollouts concatenate directly.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from hvacdata import paths
from hvacdata.dataset import load_rollout
from hvacdata.features import (FEATURE_VERSION, GLOBAL_FEATURES, ZONE_FEATURES, BuildingEncoder,
                               decode_plant_actions, decode_zone_actions,
                               zone_geometry)

FEATURES_DIR = paths.DATA_ROOT / "features"


def _split_of(scenario_dir: str) -> str:
    if "training_scenarios" in scenario_dir:
        return "kit"
    b = Path(scenario_dir).parent.name
    return "val" if b.startswith("building_09") else "train"


def _encoder_module(version: int):
    """The feature encoder for a version: 2 is current, 1 kept for comparisons."""
    import importlib
    return importlib.import_module("hvacdata.features" if version == 2 else "hvacdata.features_v1")


def encode_rollout(npz_path: str, out_root: str, version: int = FEATURE_VERSION) -> dict:
    F = _encoder_module(version)
    ZONE_FEATURES, GLOBAL_FEATURES = F.ZONE_FEATURES, F.GLOBAL_FEATURES
    paths.ensure_kit_importable()
    r = load_rollout(npz_path)
    s = r["summary"]
    meta = dict(r["meta"])
    scen = paths.REPO_ROOT / s["scenario_dir"]
    meta["floor_plan"] = np.load(scen / "floor_plan.npy")
    meta["zone_map"] = np.load(scen / "zone_map.npy")
    zone_ids = list(r["zone_ids"])
    geo = zone_geometry(meta, zone_ids)
    enc = F.BuildingEncoder(meta, zone_ids=zone_ids, geometry=geo)

    obs, T = r["obs"], len(r["reward"])
    Z = np.empty((T + 1, enc.V, len(ZONE_FEATURES)), np.float32)
    G = np.empty((T + 1, len(GLOBAL_FEATURES)), np.float32)
    TK = np.empty((T + 1, enc.V), np.float32)
    SH = np.empty((T + 1, enc.V), np.float32)
    for t in range(T + 1):
        # obs[t] is the state after step t-1, stamped with that step's end time
        e = (enc.encode(obs[t], float(r["timestamp"][t - 1]) if t > 0 else None) if version == 2
             else enc.encode(obs[t]))
        Z[t], G[t], TK[t], SH[t] = e["zone"], e["glob"], e["temp_k"], e["shortfall"]

    station = s["weather"].rsplit("_", 1)[0]
    ids = {"building": s["building"], "weather": s["weather"], "station": station,
           "split": _split_of(s["scenario_dir"]), "policy": s["policy"],
           "sigma": s["sigma"], "p_flip": s["p_flip"],
           # behaviour: the data-collection controller; older rollouts predate the field
           "behaviour": s.get("behaviour") or (s["policy"] + ("_noise" if s["sigma"] or s["p_flip"] else "")),
           "base_policy": s.get("base_policy", s["policy"]),
           "feature_version": version}
    stem = f"{s['building']}__{s['weather']}__{Path(npz_path).stem}"
    out_root = Path(out_root)

    # ---- zone table ------------------------------------------------------
    za = decode_zone_actions(meta, zone_ids, r["continuous_action"])
    pa = decode_plant_actions(meta, r["continuous_action"], r["discrete_action"])
    V = enc.V
    k = enc.zone_ahu
    zone_cols = {
        "t": np.repeat(np.arange(T), V).astype(np.int32),
        "zone": np.tile(np.arange(V), T).astype(np.int16),
        "ahu": np.tile(k, T).astype(np.int16),
    }
    for j, name in enumerate(ZONE_FEATURES):
        zone_cols[f"f_{name}"] = Z[:T, :, j].ravel()
    # Commanded actions at t, each on a fixed physical scale (see ACTION_COLUMNS).
    f32 = lambda x: np.asarray(x).ravel().astype(np.float32)  # noqa: E731
    rep = lambda x: np.repeat(np.asarray(x), V)  # noqa: E731  # per-step value onto every zone row
    zone_cols["a_damper"] = f32(np.clip((za["damper_cmd"] - 0.1) / 0.9, 0, 1))
    zone_cols["a_reheat"] = f32(np.clip(za["reheat_cmd"], 0, 1))
    zone_cols["a_ahu_run"] = pa["ahu_run_cmd"][:, k].ravel().astype(np.int8)
    zone_cols["a_ahu_sat"] = f32((pa["sat_cmd_k"][:, k] - 293.15) / 5.0)
    zone_cols["a_ahu_pressure"] = f32(pa["static_pressure_cmd_pa"][:, k] / 2000.0)
    zone_cols["a_ahu_recirc"] = f32(np.clip(pa["recirc_cmd"][:, k], 0, 0.7) / 0.7)
    zone_cols["a_water_temp"] = f32(rep((pa["water_temp_cmd_k"] - 330.0) / 20.0))
    zone_cols["a_water_dp"] = f32(rep(pa["water_dp_cmd"] / 20.0))
    zone_cols["a_pump_run"] = rep(pa["pump_run_cmd"]).astype(np.int8)
    zone_cols["temp_k"] = TK[:T].ravel()
    zone_cols["y_dtemp"] = (TK[1:] - TK[:T]).ravel()         # K over the next step
    zone_cols["y_dtemp_1h"] = np.concatenate(                 # K over the next hour
        [TK[min(t + 12, T)] - TK[t] for t in range(T)]).astype(np.float32)
    zone_cols["y_shortfall"] = SH[1:].ravel()
    zdf = pd.DataFrame(zone_cols)
    for c, v in ids.items():
        zdf[c] = v
    (out_root / "zone").mkdir(parents=True, exist_ok=True)
    zdf.to_parquet(out_root / "zone" / f"{stem}.parquet", compression="zstd", index=False)

    # ---- step table ------------------------------------------------------
    step = {"t": np.arange(T, dtype=np.int32), "timestamp": r["timestamp"]}
    for j, name in enumerate(GLOBAL_FEATURES):
        step[f"g_{name}"] = G[:T, j]
    step["outside_air_k"] = r["outside_air_k"]
    step["mean_temp_k"] = TK[:T].mean(1)
    step["mean_shortfall_next"] = SH[1:].mean(1)
    step["worst_shortfall_next"] = SH[1:].max(1)
    step["n_zones_out_of_band_next"] = (SH[1:] > 0).sum(1).astype(np.int16)
    step["ahu_on_frac"] = pa["ahu_run_cmd"].mean(1).astype(np.float32)
    step["pump_run"] = pa["pump_run_cmd"].astype(np.int8)
    step["water_temp_cmd_k"] = pa["water_temp_cmd_k"].astype(np.float32)
    step["mean_sat_cmd_k"] = pa["sat_cmd_k"].mean(1).astype(np.float32)
    step["mean_static_pressure_cmd_pa"] = pa["static_pressure_cmd_pa"].mean(1).astype(np.float32)
    step["mean_recirc_cmd"] = pa["recirc_cmd"].mean(1).astype(np.float32)
    step["mean_damper_cmd"] = za["damper_cmd"].mean(1).astype(np.float32)
    step["mean_reheat_cmd"] = za["reheat_cmd"].mean(1).astype(np.float32)
    step["reward"] = r["reward"]
    step["comfort_term"] = 0.6 * r["rc_normalized_productivity_regret"]
    step["cost_term"] = -0.2 * r["rc_normalized_energy_cost"]
    step["carbon_term"] = -0.2 * r["rc_normalized_carbon_emission"]
    for key in ("rc_electricity_energy_cost", "rc_natural_gas_energy_cost",
                "rc_carbon_emitted", "rc_total_occupancy"):
        step[key[3:]] = r[key]
    sdf = pd.DataFrame(step)
    for c, v in ids.items():
        sdf[c] = v
    sdf["n_zones"], sdf["n_ahus"] = V, enc.K
    sdf["mean_area_m2"] = float(np.exp(geo["log_area"]).mean() * 50)
    sdf["mean_exterior_fraction"] = float(geo["exterior_fraction"].mean())
    (out_root / "step").mkdir(parents=True, exist_ok=True)
    sdf.to_parquet(out_root / "step" / f"{stem}.parquet", compression="zstd", index=False)

    # Per-building static zone table, written once per building.
    static = out_root / "zones_static" / f"{s['building']}.parquet"
    if not static.exists():
        static.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"building": s["building"], "zone": np.arange(V), "zone_id": zone_ids,
                      "ahu": k, "area_m2": np.exp(geo["log_area"]) * 50,
                      "exterior_fraction": geo["exterior_fraction"]}).to_parquet(static, index=False)
    return {"stem": stem, "rows_zone": len(zdf), "rows_step": len(sdf)}


def build(tag: str, workers: int = 4, force: bool = False, version: int = FEATURE_VERSION,
          out_tag: str | None = None) -> list[dict]:
    root = paths.ROLLOUTS_DIR / tag
    out_root = FEATURES_DIR / (out_tag or tag)
    jobs = []
    for npz in sorted(root.rglob("*.npz")):
        if not npz.with_suffix(".json").exists():
            continue  # still being written (the .json lands last)
        stem = f"{npz.parent.parent.name}__{npz.parent.name}__{npz.stem}"
        if force or not (out_root / "step" / f"{stem}.parquet").exists():
            jobs.append(str(npz))
    print(f"{len(jobs)} rollout(s) to encode")
    done = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(encode_rollout, j, str(out_root), version): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            done.append(res)
            print(f"  [{i}/{len(jobs)}] {res['stem']}: {res['rows_zone']:,} zone rows")
    F = _encoder_module(version)
    spec = {"version": version, "zone_features": F.ZONE_FEATURES, "global_features": F.GLOBAL_FEATURES}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "feature_spec.json").write_text(json.dumps(spec, indent=2))
    return done
