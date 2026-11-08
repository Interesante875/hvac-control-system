"""Building-size-independent feature encoding of the raw observation.

Why re-encode
-------------
The environment's observation vector is not consistently scaled. Its
normaliser matches field names *exactly* against a 13-entry table and passes
every other field through untouched, so one vector mixes:

* z-scored values: zone temperature ((K - 295) / 5), damper and reheat
  positions, hot-water dp setpoint;
* raw kelvin: zone heating/cooling setpoints, outside air, supply air and
  supply water temperatures;
* raw pascals (0-2000), raw m3/s airflow, raw occupant counts (which grow with
  building size), request counts equal to the number of zones served.

It is also highly redundant: 12 sin/cos pairs of hour-of-day all at the same
24 h frequency (24 columns, rank 2), the same for day-of-week, AHU sensors equal
to their setpoints, three identical fan run flags per AHU, the outside air
temperature repeated once per AHU, and a per-zone airflow setpoint that never
changes. Its length (and the meaning of each position) changes per building.

What this produces
------------------
:class:`BuildingEncoder` is built once per building from the metadata the
policy receives at runtime, and turns each observation into

* ``zone``  (V, F_zone)  one row per zone, physical units, comparable across
  buildings, including the zone's AHU and hot-water context,
* ``glob``  (F_glob,)    building-wide context (weather, time, schedule).

Everything is computable from ``obs`` + metadata + the encoder's own history,
so the same class is used to build training data offline and inside a
submitted policy. Scaling uses fixed physical constants (not statistics fitted
to the training set), so an unseen building or climate cannot shift them.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any

import numpy as np

# Normalisation the environment applies to the few fields it does normalise.
# (StandardScoreObservationNormalizer, exact-name match; see module docstring.)
ZONE_T_MEAN, ZONE_T_STD = 295.0, 5.0
FRAC_MEAN, FRAC_STD = 0.5, 0.5          # damper / reheat / run flags
HWS_DP_MEAN, HWS_DP_STD = 10.0, 6.0     # hws differential_pressure_setpoint

# Reward constants (identical in every shipped and generated config).
COMFORT_KNEE_K = 5.0                    # productivity_deviation_ref
MAX_RECIRC = 0.70
MIN_SAT_K = 285.0
STEPS_PER_HOUR = 12
CONTROL_VOLUME_CM = 20                  # simulator grid cell edge, every config

# Fixed scales for the encoded features.
T_REF_K = 293.15                        # 20 C
DT_SCALE = COMFORT_KNEE_K               # temperature differences in "knees"
OA_SCALE = 15.0
PRESSURE_MAX = 2000.0
WATER_T_REF, WATER_T_SCALE = 330.0, 20.0

_VAV = re.compile(r"^VAV_(\d+)_(.+)$")
_AHU = re.compile(r"^ahs_(ahu_\d+)_(.+)$")

ZONE_FEATURES = [
    # thermal state relative to the comfort band (the reward's own geometry)
    "temp_minus_band_mid",      # (T - (h+c)/2) / 5K
    "below_heat_sp",            # max(0, h - T) / 5K
    "above_cool_sp",            # max(0, T - c) / 5K
    "comfort_shortfall",        # reward-shaped: d/w if d<=w else 1+((d-w)/w)^2
    "band_half_width",          # (c - h) / 2 / 5K   (comfort vs eco setback)
    "temp_abs",                 # (T - 20C) / 5K
    # recent dynamics (from encoder history)
    "dtemp_5min",               # K per step, x12 -> K/h
    "dtemp_1h",                 # T - T(1h ago), K
    # neighbourhood
    "temp_minus_ahu_mean",      # relative to the zones sharing its AHU, /5K
    "temp_minus_ahu_worst",     # distance to that AHU's worst zone shortfall
    "temp_minus_outside",       # (T - OA) / 15K
    # actuator state (what the box actually took, after floors)
    "damper", "reheat",         # fractions in [0, 1]
    # static geometry (from floor_plan / zone_map)
    "log_area",                 # log(area / 50 m2)
    "exterior_fraction",        # share of the zone's boundary cells on the facade
    # its air handler
    "ahu_run",
    "ahu_supply_temp",          # (SAT - 20C) / 5K
    "ahu_sat_minus_zone",       # (SAT - T) / 5K: sign = heating/cooling potential
    "ahu_static_pressure",      # Pa / 2000
    "ahu_recirculation",        # fraction
    "ahu_flow_per_zone",        # m3/s per zone served
    "ahu_cooling_request_frac", # requests / zones served
    "ahu_n_zones",              # zones served / 10
    # hot-water loop
    "water_supply_temp",        # (K - 330) / 20
    "water_dp",                 # setpoint, native / 20
    "pump_run",
]

GLOBAL_FEATURES = [
    "outside_temp",             # (OA - 10C) / 15K
    "outside_dtemp_1h",         # K/h
    "mixed_air_min",            # coldest achievable mixed air vs min SAT, /5K
    "sat_inert",                # 1 if AHU cooling setpoint cannot bite (OA < -6.8C)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "comfort_now", "comfort_soon",
    "hours_to_comfort",         # until next comfort-mode start, /24 (0 if in comfort)
    "occupants_per_zone",       # num_occupants / V
    "occupied",                 # num_occupants > 0  (comfort only scored then)
    "n_zones_log",              # log(V / 50)
    "n_ahus",                   # K / 10
]


def comfort_shortfall(dev_k: np.ndarray, knee: float = COMFORT_KNEE_K) -> np.ndarray:
    """Per-zone productivity shortfall exactly as the reward computes it."""
    d = np.maximum(dev_k, 0.0) / knee
    return np.where(d <= 1.0, d, 1.0 + (d - 1.0) ** 2)


def zone_geometry(meta: dict[str, Any], zone_ids: list[str]) -> dict[str, np.ndarray]:
    """Area and facade exposure per zone from the floor plan and zone map.

    Uses the simulator's own room labelling (building_utils), which is plain
    numpy/OpenCV, so this also works inside a submitted policy. Falls back to
    neutral values if the arrays are absent.
    """
    v = len(zone_ids)
    out = {"log_area": np.zeros(v, np.float32), "exterior_fraction": np.zeros(v, np.float32)}
    fp, zm = meta.get("floor_plan"), meta.get("zone_map")
    if fp is None or zm is None:
        return out
    try:
        from smart_buildings.smart_control.simulator import building_utils
    except ImportError:
        return out
    room_dict, exterior_walls, *_ = building_utils.construct_building_data_types(
        floor_plan=np.asarray(fp), zone_map=np.asarray(zm))
    ext = np.asarray(exterior_walls) != 0  # marked with a negative constant (-2)
    cv_area_m2 = (CONTROL_VOLUME_CM / 100.0) ** 2
    h, w = ext.shape
    for i, z in enumerate(zone_ids):
        cells = room_dict.get(z.replace("VAV_", "room_"))
        if not cells:
            continue
        cells = np.asarray(cells)
        out["log_area"][i] = np.log(max(len(cells) * cv_area_m2, 1.0) / 50.0)
        # boundary cells of the zone, and how many of them touch a facade wall
        occ = np.zeros((h, w), bool)
        occ[cells[:, 0], cells[:, 1]] = True
        n_boundary = n_facade = 0
        for r, c in cells:
            nb = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            nb = [(a, b) for a, b in nb if 0 <= a < h and 0 <= b < w]
            if any(not occ[a, b] for a, b in nb):
                n_boundary += 1
                if any(ext[a, b] for a, b in nb):
                    n_facade += 1
        out["exterior_fraction"][i] = n_facade / max(n_boundary, 1)
    return out


class BuildingEncoder:
    """Encode raw observations of one building into fixed-width features."""

    def __init__(self, meta: dict[str, Any], zone_ids: list[str] | None = None,
                 geometry: dict[str, np.ndarray] | None = None):
        names = meta["observation_field_names"]
        ix = {n: i for i, n in enumerate(names)}
        vav: dict[str, dict[str, int]] = {}
        ahu: dict[str, dict[str, int]] = {}
        for n, i in ix.items():
            if m := _VAV.match(n):
                vav.setdefault(f"VAV_{m.group(1)}", {})[m.group(2)] = i
            elif m := _AHU.match(n):
                ahu.setdefault(m.group(1), {})[m.group(2)] = i

        self.ahu_names = list(meta["ahu_zone_assignments"])
        z2a = {room.replace("room_", "VAV_"): k
               for k, rooms in meta["ahu_zone_assignments"].items() for room in rooms}
        self.zone_ids = zone_ids or sorted(vav, key=lambda z: int(z.split("_")[1]))
        V, K = len(self.zone_ids), len(self.ahu_names)
        self.V, self.K = V, K

        def col(field: str) -> np.ndarray:
            return np.array([vav[z][field] for z in self.zone_ids])

        def acol(field: str) -> np.ndarray:
            return np.array([ahu[a][field] for a in self.ahu_names])

        self.i_temp = col("zone_air_temperature_sensor")
        self.i_heat = col("zone_air_heating_temperature_setpoint_sensor")
        self.i_cool = col("zone_air_cooling_temperature_setpoint_sensor")
        self.i_damper = col("supply_air_damper_percentage_command")
        self.i_reheat = col("reheat_valve_percentage_command")
        self.i_sat = acol("supply_air_temperature_sensor")
        self.i_sp = acol("supply_air_static_pressure_sensor")
        self.i_recirc = acol("recirculation_percentage_command")
        self.i_flow = acol("supply_air_flowrate_sensor")
        self.i_run = acol("supply_fan_run_command")
        self.i_creq = acol("cooling_request_count")
        self.i_oa = ix[f"ahs_{self.ahu_names[0]}_outside_air_temperature_sensor"]
        self.i_wt = ix["hws_supply_water_temperature_sensor"]
        self.i_wdp = ix["hws_differential_pressure_setpoint"]
        self.i_pump = ix["hws_run_status"]
        self.i_hod = (ix["hod_sin_000"], ix["hod_cos_000"])
        self.i_dow = (ix["dow_sin_000"], ix["dow_cos_000"])
        self.i_cnow, self.i_csoon = ix["comfort_mode_now"], ix["comfort_mode_soon"]
        self.i_occ = ix["num_occupants"]

        self.zone_ahu = np.array([self.ahu_names.index(z2a[z]) for z in self.zone_ids])
        self.ahu_n_zones = np.bincount(self.zone_ahu, minlength=K).astype(np.float32)
        self.morning = int(meta.get("morning_start_hour", 6))
        self.evening = int(meta.get("evening_start_hour", 19))

        geo = geometry if geometry is not None else zone_geometry(meta, self.zone_ids)
        self.log_area = geo["log_area"]
        self.exterior_fraction = geo["exterior_fraction"]
        self.reset()

    def reset(self) -> None:
        self._temp_hist: deque[np.ndarray] = deque(maxlen=STEPS_PER_HOUR + 1)
        self._oa_hist: deque[float] = deque(maxlen=STEPS_PER_HOUR + 1)

    # -- decoding helpers -------------------------------------------------
    def zone_temp_k(self, obs: np.ndarray) -> np.ndarray:
        return obs[self.i_temp] * ZONE_T_STD + ZONE_T_MEAN

    def _hours_to_comfort(self, hour: float, dow: int, comfort_now: float) -> float:
        if comfort_now > 0.5:
            return 0.0
        h = (self.morning - hour) % 24
        day = (dow + (1 if hour >= self.morning else 0)) % 7
        while day >= 5:  # comfort mode runs Monday-Friday
            h += 24
            day = (day + 1) % 7
        return h

    # -- main entry ---------------------------------------------------------
    def encode(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        obs = np.asarray(obs, dtype=np.float64)
        T = self.zone_temp_k(obs)
        h, c = obs[self.i_heat], obs[self.i_cool]
        oa = obs[self.i_oa]
        self._temp_hist.append(T)
        self._oa_hist.append(oa)
        prev = self._temp_hist[-2] if len(self._temp_hist) > 1 else T
        hour_ago = self._temp_hist[0]
        oa_hour_ago = self._oa_hist[0]

        dev = np.maximum(h - T, T - c)
        short = comfort_shortfall(dev)
        a = self.zone_ahu
        ahu_mean = np.bincount(a, weights=T, minlength=self.K) / np.maximum(self.ahu_n_zones, 1)
        ahu_worst = np.zeros(self.K)
        np.maximum.at(ahu_worst, a, short)

        sat = obs[self.i_sat]
        run = obs[self.i_run]
        n_served = self.ahu_n_zones[a]

        zone = np.stack([
            (T - (h + c) / 2) / DT_SCALE,
            np.maximum(0, h - T) / DT_SCALE,
            np.maximum(0, T - c) / DT_SCALE,
            short,
            (c - h) / 2 / DT_SCALE,
            (T - T_REF_K) / DT_SCALE,
            (T - prev) * STEPS_PER_HOUR,
            T - hour_ago,
            (T - ahu_mean[a]) / DT_SCALE,
            ahu_worst[a] - short,
            (T - oa) / OA_SCALE,
            obs[self.i_damper] * FRAC_STD + FRAC_MEAN,
            obs[self.i_reheat] * FRAC_STD + FRAC_MEAN,
            self.log_area,
            self.exterior_fraction,
            run[a],
            (sat[a] - T_REF_K) / DT_SCALE,
            (sat[a] - T) / DT_SCALE,
            obs[self.i_sp][a] / PRESSURE_MAX,
            obs[self.i_recirc][a],
            obs[self.i_flow][a] / n_served,
            obs[self.i_creq][a] / n_served,
            n_served / 10.0,
            np.full(self.V, (obs[self.i_wt] - WATER_T_REF) / WATER_T_SCALE),
            np.full(self.V, (obs[self.i_wdp] * HWS_DP_STD + HWS_DP_MEAN) / 20.0),
            np.full(self.V, obs[self.i_pump] * FRAC_STD + FRAC_MEAN),
        ], axis=1).astype(np.float32)

        hs, hc = obs[self.i_hod[0]], obs[self.i_hod[1]]
        ds, dc = obs[self.i_dow[0]], obs[self.i_dow[1]]
        hour = (np.degrees(np.arctan2(hs, hc)) / 15.0) % 24
        dow = int(round((np.degrees(np.arctan2(ds, dc)) / (360 / 7)))) % 7
        return_air = float(T.mean())
        mixed_min = MAX_RECIRC * return_air + (1 - MAX_RECIRC) * oa
        occ = obs[self.i_occ]
        glob = np.array([
            (oa - 283.15) / OA_SCALE,
            oa - oa_hour_ago,
            (mixed_min - MIN_SAT_K) / DT_SCALE,
            float(mixed_min < MIN_SAT_K),
            hs, hc, ds, dc, float(dow >= 5),
            obs[self.i_cnow], obs[self.i_csoon],
            self._hours_to_comfort(hour, dow, obs[self.i_cnow]) / 24.0,
            occ / self.V,
            float(occ > 0),
            np.log(self.V / 50.0),
            self.K / 10.0,
        ], dtype=np.float32)
        return {"zone": zone, "glob": glob, "temp_k": T.astype(np.float32),
                "shortfall": short.astype(np.float32)}


def decode_zone_actions(meta: dict, zone_ids: list[str], continuous: np.ndarray) -> dict:
    """Commanded zone actions in physical fractions [0, 1] (damper, reheat)."""
    za = meta["zone_action_indices"]
    d = np.array([za[z]["supply_air_damper_percentage_command"] for z in zone_ids])
    r = np.array([za[z]["reheat_valve_percentage_command"] for z in zone_ids])
    return {"damper_cmd": (continuous[..., d] + 1) / 2, "reheat_cmd": (continuous[..., r] + 1) / 2}


def decode_plant_actions(meta: dict, continuous: np.ndarray, discrete: np.ndarray) -> dict:
    """Commanded plant actions in native units."""
    K = len(meta["ahu_zone_assignments"])
    lin = lambda x, lo, hi: lo + (x + 1) / 2 * (hi - lo)  # noqa: E731
    return {
        "sat_cmd_k": lin(continuous[..., meta["temp_indices"]], 285.0, 305.0),
        "static_pressure_cmd_pa": lin(continuous[..., meta["pressure_indices"][:K]], 0.0, 2000.0),
        "recirc_cmd": lin(continuous[..., meta["recirculation_indices"]], 0.0, 1.0),
        "water_temp_cmd_k": lin(continuous[..., meta["water_indices"][0]], 310.0, 350.0),
        "water_dp_cmd": lin(continuous[..., meta["pressure_indices"][K]], 0.0, 20.0),
        "ahu_run_cmd": discrete[..., :K],
        "pump_run_cmd": discrete[..., K],
    }
