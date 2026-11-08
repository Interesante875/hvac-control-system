"""Run controllers on scenarios and record full transitions to disk.

One rollout = one (scenario, policy, noise) triple, saved as
``data/rollouts/<tag>/<building>/<weather>/<policy>.npz`` plus a sibling
``.json`` with the building metadata and summary. Arrays in the npz:

    obs                 (T+1, D) float32  observation vector, obs[t] -> action[t]
    continuous_action   (T, C)   float32  what was *commanded* (after noise)
    discrete_action     (T, K+1) int8
    reward              (T,)     float32  r_t, never positive
    rc_<name>           (T,)     float32  reward components (comfort, cost, carbon, ...)
    zone_temp_k         (T, V)   float32  raw zone air temperature after the step
    outside_air_k       (T,)     float32
    timestamp           (T,)     int64    unix seconds (the sim's local wall clock)

The observation is whatever the environment hands the policy (normalised), so
a model trained on ``obs`` sees exactly what it will see at submission time.
"""

from __future__ import annotations

import importlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from hvacdata import paths

REWARD_COMPONENTS = (
    "productivity_reward",
    "electricity_energy_cost",
    "natural_gas_energy_cost",
    "carbon_emitted",
    "normalized_productivity_regret",
    "normalized_energy_cost",
    "normalized_carbon_emission",
    "total_occupancy",
)


class NoisyPolicy:
    """Wraps a controller with temporally-correlated exploration.

    A fresh perturbation is drawn every ``hold`` steps and held, because a
    5-minute step is far shorter than the building's thermal time constant and
    white noise per step mostly averages out. Continuous actions get Gaussian
    noise (then clipped to [-1, 1]); each discrete run command is flipped with
    probability ``p_flip`` per hold period. With sigma=0 and p_flip=0 this is the
    base controller exactly.
    """

    def __init__(self, base: Any, sigma: float, p_flip: float, hold: int, seed: int):
        self.base, self.sigma, self.p_flip, self.hold = base, sigma, p_flip, max(1, hold)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self._dc = self._flip = None

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        a = self.base.get_action(obs)
        c = np.asarray(a["continuous_action"], dtype=np.float32)
        d = np.asarray(a["discrete_action"], dtype=np.int32)
        if self.sigma <= 0 and self.p_flip <= 0:
            return {"continuous_action": c, "discrete_action": d}
        if self.t % self.hold == 0:
            self._dc = self.rng.normal(0.0, self.sigma, c.shape).astype(np.float32)
            self._flip = self.rng.random(d.shape) < self.p_flip
        self.t += 1
        c = np.clip(c + self._dc, -1.0, 1.0)
        d = np.where(self._flip, 1 - d, d).astype(np.int32)
        return {"continuous_action": c, "discrete_action": d}


class BehaviourPolicy:
    """A base controller plus structured exploration, for collecting training data.

    Spec keys (all optional, all off by default):
      sigma        Gaussian noise std on every continuous action, held ``hold`` steps
      zone_sigma   extra noise on zone dampers/reheat only
      p_flip_ahu   per-hold probability of flipping each AHU's run command
      plant_random per-hold probability of replacing the plant setpoints (supply
                   air temperature, fan pressure, recirculation, water temperature,
                   pump pressure) with values drawn uniformly from wide but sane
                   ranges. Zone loops stay closed, so the building stays
                   controlled while the plant visits settings the baselines never
                   use (pid_water pins recirculation at 0.7, for example).
    The hot-water pump is never flipped: switching it off in the cold is what
    tripled the loss of plain action noise in Fairbanks.
    """

    # Native ranges the random plant draws from, mapped to [-1, 1] below.
    PLANT_RANGES = {
        "temp": (285.0, 300.0, 285.0, 305.0),       # draw lo, hi, action lo, hi (K)
        "pressure": (200.0, 1600.0, 0.0, 2000.0),   # Pa
        "recirc": (0.0, 0.7, 0.0, 1.0),
        "water_temp": (315.0, 350.0, 310.0, 350.0),  # K
        "water_dp": (4.0, 20.0, 0.0, 20.0),
    }

    def __init__(self, base: Any, meta: dict, spec: dict, hold: int, seed: int):
        self.base, self.spec, self.hold = base, spec, max(1, hold)
        self.rng = np.random.default_rng(seed)
        self.K = len(meta["ahu_zone_assignments"])
        self.i_temp = np.array(meta["temp_indices"])
        self.i_press = np.array(meta["pressure_indices"][: self.K])
        self.i_recirc = np.array(meta["recirculation_indices"])
        self.i_wt = meta["water_indices"][0]
        self.i_wdp = meta["pressure_indices"][self.K]
        self.i_zone = np.array([i for z in meta["zone_action_indices"].values() for i in z.values()])
        self.t = 0
        self._dc = self._dz = self._flip = self._plant = None

    def _norm(self, x, a_lo, a_hi):
        return 2 * (x - a_lo) / (a_hi - a_lo) - 1

    def _draw(self, n_cont: int) -> None:
        s = self.spec
        self._dc = self.rng.normal(0, s.get("sigma", 0.0), n_cont).astype(np.float32)
        self._dz = self.rng.normal(0, s.get("zone_sigma", 0.0), len(self.i_zone)).astype(np.float32)
        self._flip = self.rng.random(self.K) < s.get("p_flip_ahu", 0.0)
        self._plant = None
        if self.rng.random() < s.get("plant_random", 0.0):
            p = {}
            for k, (lo, hi, alo, ahi) in self.PLANT_RANGES.items():
                n = 1 if k.startswith("water") else self.K
                p[k] = self._norm(self.rng.uniform(lo, hi, n), alo, ahi).astype(np.float32)
            self._plant = p

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        a = self.base.get_action(obs)
        c = np.array(a["continuous_action"], dtype=np.float32)
        d = np.array(a["discrete_action"], dtype=np.int32)
        if self.t % self.hold == 0:
            self._draw(len(c))
        self.t += 1
        c += self._dc
        c[self.i_zone] += self._dz
        if self._plant is not None:
            c[self.i_temp] = self._plant["temp"]
            c[self.i_press] = self._plant["pressure"]
            c[self.i_recirc] = self._plant["recirc"]
            c[self.i_wt] = self._plant["water_temp"][0]
            c[self.i_wdp] = self._plant["water_dp"][0]
        d[: self.K] = np.where(self._flip, 1 - d[: self.K], d[: self.K])
        return {"continuous_action": np.clip(c, -1, 1), "discrete_action": d}


# Named behaviours used by the collection plan (configs/collection.yaml).
BEHAVIOURS: dict[str, tuple[str, dict]] = {
    "pid_water": ("pid_water", {}),
    "pid": ("pid", {}),
    "fixed_setpoint": ("fixed_setpoint", {}),
    "bang_bang": ("bang_bang", {}),
    "pid_water_explore": ("pid_water", {"sigma": 0.08, "zone_sigma": 0.25, "p_flip_ahu": 0.03}),
    "pid_water_plant_random": ("pid_water", {"plant_random": 0.5, "zone_sigma": 0.1}),
}


def resolve_policy(key: str):
    """A kit POLICY_MAP key (``pid_water``) or an import path (``pkg.mod:Class``)."""
    from smart_buildings.evaluate import POLICY_MAP
    if ":" in key:
        mod, cls = key.split(":", 1)
        return getattr(importlib.import_module(mod), cls)
    if key not in POLICY_MAP:
        raise KeyError(f"Unknown policy {key!r}; kit keys: {sorted(POLICY_MAP)} "
                       "or use module:Class")
    return POLICY_MAP[key]


def load_env(scenario_dir: Path):
    """Load the env exactly as the kit's evaluator does. Call inside in_kit()."""
    from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment
    env = load_hybrid_environment(scenario_dir / "config.gin")
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0  # as in evaluate.py
    meta = create_building_meta_data(env, scenario_dir)
    return env, meta


def _jsonable(meta: dict) -> dict:
    out = {}
    for k, v in meta.items():
        if k in ("floor_plan", "zone_map"):
            continue  # large arrays; they live in the scenario dir as .npy
        if k == "holidays":
            v = sorted(str(x) for x in v) if v is not None else None
        out[k] = v
    return json.loads(json.dumps(out, default=str))


def run_one(scenario_dir: str, policy_key: str, out_path: str, steps: int,
            sigma: float, p_flip: float, hold: int, seed: int,
            behaviour: str | None = None) -> dict:
    """One rollout. With ``behaviour`` (a BEHAVIOURS key), its base controller
    and exploration spec replace ``policy_key``/``sigma``/``p_flip``."""
    scen = paths.REPO_ROOT / scenario_dir
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with paths.in_kit():
        from smart_buildings.evaluate import initialize_policy, validate_action
        env, meta = load_env(scen)
        spec = env.action_spec()
        if behaviour is not None:
            base_key, bspec = BEHAVIOURS[behaviour]
            base = initialize_policy(resolve_policy(base_key), meta, spec)
            policy = BehaviourPolicy(base, meta, bspec, hold, seed)
        else:
            base = initialize_policy(resolve_policy(policy_key), meta, spec)
            policy = NoisyPolicy(base, sigma, p_flip, hold, seed)

        vavs = env.building.simulator._hvac._vavs
        ahus = env.building.simulator._hvac.air_handler._ahus
        zone_ids = list(vavs.keys())

        obs, _ = env.reset()
        D = len(obs)
        C = int(spec["continuous_action"].shape[0])
        K1 = int(spec["discrete_action"].shape[0])
        O = np.zeros((steps + 1, D), np.float32)
        A_c = np.zeros((steps, C), np.float32)
        A_d = np.zeros((steps, K1), np.int8)
        R = np.zeros(steps, np.float32)
        RC = {k: np.zeros(steps, np.float32) for k in REWARD_COMPONENTS}
        ZT = np.zeros((steps, len(zone_ids)), np.float32)
        OA = np.zeros(steps, np.float32)
        TS = np.zeros(steps, np.int64)
        O[0] = obs

        t0 = time.time()
        n = 0
        for t in range(steps):
            action = validate_action(policy.get_action(obs), spec)
            obs, reward, terminated, truncated, _ = env.step(action)
            _, rr = env.get_reward_info_and_response()
            O[t + 1] = obs
            A_c[t] = action["continuous_action"]
            A_d[t] = action["discrete_action"]
            R[t] = reward
            for k in REWARD_COMPONENTS:
                RC[k][t] = getattr(rr, k)
            ZT[t] = [vavs[z].zone_air_temperature for z in zone_ids]
            OA[t] = ahus[0].outside_air_temperature_sensor
            TS[t] = int(env.building.current_timestamp.timestamp())
            n = t + 1
            if terminated or truncated:
                break
        elapsed = time.time() - t0

    trunc = slice(0, n)
    np.savez_compressed(
        out, obs=O[: n + 1], continuous_action=A_c[trunc], discrete_action=A_d[trunc],
        reward=R[trunc], zone_temp_k=ZT[trunc], outside_air_k=OA[trunc], timestamp=TS[trunc],
        **{f"rc_{k}": v[trunc] for k, v in RC.items()},
    )
    summary = {
        "scenario_dir": scenario_dir, "building": scen.parent.name, "weather": scen.name,
        "policy": behaviour or policy_key, "behaviour": behaviour or policy_key,
        "base_policy": BEHAVIOURS[behaviour][0] if behaviour else policy_key,
        "sigma": sigma, "p_flip": p_flip, "hold": hold, "seed": seed,
        "steps": n, "full_episode": n == paths.STEPS_PER_EPISODE,
        "total_reward": float(R[trunc].sum()),
        # Per-scenario leaderboard value: mean per-step loss (lower is better).
        "score": float(-R[trunc].mean()) if n else None,
        "comfort_term": float(0.6 * RC["normalized_productivity_regret"][trunc].sum()),
        "cost_term": float(-0.2 * RC["normalized_energy_cost"][trunc].sum()),
        "carbon_term": float(-0.2 * RC["normalized_carbon_emission"][trunc].sum()),
        "n_zones": len(zone_ids), "n_ahus": len(ahus), "obs_dim": D,
        "seconds": round(elapsed, 1),
    }
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({"summary": summary, "zone_ids": zone_ids, "meta": _jsonable(meta)}, f)
    return summary


def collect(entries: list[dict], policies: list[str], tag: str, steps: int, workers: int,
            sigma: float = 0.0, p_flip: float = 0.0, hold: int = 12, seed: int = 0,
            force: bool = False) -> list[dict]:
    jobs = []
    for e in entries:
        for p in policies:
            pname = p.replace(":", ".")
            noise = f"_s{sigma:g}_f{p_flip:g}" if (sigma or p_flip) else ""
            out = paths.ROLLOUTS_DIR / tag / e["building"] / e["weather"] / f"{pname}{noise}.npz"
            if out.exists() and not force:
                continue
            # Distinct but reproducible noise per job (hash() is salted per run).
            job_seed = _stable_seed(e["building"], e["weather"], p, seed)
            jobs.append((e["scenario_dir"], p, str(out), steps, sigma, p_flip, hold, job_seed))
    print(f"{len(jobs)} rollout(s) to run with {workers} worker(s), {steps} steps each")
    results = []
    # One task per child keeps gin's global config and any simulator caches
    # from leaking between scenarios.
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        futs = {pool.submit(run_one, *j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            j = futs[fut]
            try:
                s = fut.result()
            except Exception as exc:  # keep going; report at the end
                print(f"  [{i}/{len(jobs)}] FAILED {j[0]} {j[1]}: {exc!r}")
                continue
            results.append(s)
            print(f"  [{i}/{len(jobs)}] {s['building']}/{s['weather']} {s['policy']}: "
                  f"score={s['score']:.4f} ({s['steps']} steps, {s['seconds']}s)")
    return results


def _stable_seed(*parts: Any) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4], "big")


def build_index(tag: str | None = None):
    """Collect every rollout summary into a DataFrame (and index.csv)."""
    import pandas as pd
    root = paths.ROLLOUTS_DIR / tag if tag else paths.ROLLOUTS_DIR
    rows = []
    for j in sorted(root.rglob("*.json")):
        with open(j) as f:
            s = json.load(f)["summary"]
        s["tag"] = j.relative_to(paths.ROLLOUTS_DIR).parts[0]
        s["path"] = str(j.with_suffix(".npz").relative_to(paths.REPO_ROOT))
        rows.append(s)
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(root / "index.csv", index=False)
    return df
