"""Building-size-independent feature encoding of the raw observation (version 2).

Why re-encode
-------------
The environment's observation vector is not consistently scaled. Its
normaliser matches field names *exactly* against a 13-entry table and passes
every other field through untouched, so one vector mixes z-scored values
(zone temperature, damper, reheat), raw kelvin (setpoints, outside air, supply
air and water temperature), raw pascals, raw m3/s and head counts. It repeats
one 24 h cycle as 12 sin/cos pairs, duplicates sensors and setpoints, and
changes length and layout with every building.

How features are defined
------------------------
Every feature is declared once in ``ZONE_SPEC`` / ``GLOBAL_SPEC`` with its unit,
transform, clip and the reason for it. The encoder computes them in that
order, ``docs/FEATURES.md`` is generated from the same table, and the report
reads it, so the three cannot drift apart.

Practices applied to every column
---------------------------------
* Physical units first, then a fixed physical scale (5 K = the reward's
  comfort knee, 2000 Pa = the fan's range), never statistics fitted on one
  dataset, so an unseen building or climate cannot move the scale.
  ``FeatureScaler`` adds an optional robust standardisation fitted on the
  train split only, for models that need unit variance.
* Heavy-tailed signed values (5-minute temperature change, whose 0.1-99.9%
  range is +-110 K/h) go through ``asinh``: linear near zero, logarithmic in
  the tails, sign preserved. Zero-inflated, unbounded positives (the
  quadratic comfort shortfall, skew 21) go through ``log1p`` with a separate
  0/1 indicator. Everything is clipped to a stated range.
* A reading that means nothing while its equipment is off (supply-air
  temperature of a stopped AHU, pump pressure of a stopped pump) is gated to 0
  by its run flag, so the run flag carries the state and the reading never
  lies.
* Constant or duplicate columns are dropped (cooling-request fraction is
  always 1; static pressure tracks airflow at r=0.96), and two-valued columns
  become explicit 0/1 flags.
* Physics-shaped interactions that no single column holds: heat the supply
  air can deliver (airflow x damper x (supply - zone temperature)) and heat
  the reheat coil can deliver (pump x valve x (water - zone temperature)).
* History that the model would otherwise need a sequence model for: 5-minute
  and 1-hour changes, a 6-hour moving average of zone temperature, 6 h and
  24 h moving averages of outdoor temperature (the envelope's thermal lag),
  hours out of band, hours since the AHU switched.
* Everything is computable from the observation plus the metadata a policy
  receives, so the same class builds training data and runs in a submission.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

# Normalisation the environment applies to the few fields it does normalise.
ZONE_T_MEAN, ZONE_T_STD = 295.0, 5.0
FRAC_MEAN, FRAC_STD = 0.5, 0.5          # damper / reheat / water run status
HWS_DP_MEAN, HWS_DP_STD = 10.0, 6.0     # hws differential_pressure_setpoint

# Plant and reward constants (identical in every shipped and generated config).
COMFORT_KNEE_K = 5.0                    # productivity_deviation_ref
MAX_RECIRC = 0.70
MIN_SAT_K = 285.0
DAMPER_FLOOR = 0.10
STEPS_PER_HOUR = 12
CONTROL_VOLUME_CM = 20

T_REF_K = 293.15                        # 20 C
OA_REF_K = 283.15                       # 10 C
OA_SCALE = 15.0
WATER_T_REF, WATER_T_SCALE = 330.0, 20.0


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    unit: str
    source: str
    transform: str
    clip: tuple[float, float] | None
    why: str


Z, G = "zone", "global"
ZONE_SPEC: list[Feature] = [
    # -- thermal state relative to the comfort band
    Feature("temp_minus_band_mid", "comfort band", "5 K", "zone temp, heating & cooling setpoints",
            "(T - (h+c)/2) / 5", (-4, 4), "Where the zone sits in its own band; setpoints move between comfort and eco hours, so absolute temperature alone is ambiguous."),
    Feature("below_heat_sp", "comfort band", "5 K", "zone temp, heating setpoint",
            "max(0, h - T) / 5", (0, 4), "Hinge: only the side of the band that is violated counts, as in the reward."),
    Feature("above_cool_sp", "comfort band", "5 K", "zone temp, cooling setpoint",
            "max(0, T - c) / 5", (0, 4), "Hinge, cooling side."),
    Feature("out_of_band", "comfort band", "0/1", "zone temp, setpoints",
            "1 if outside the band", None, "Zero-inflated shortfall gets an explicit indicator, so a model separates 'in band' from 'slightly out'."),
    Feature("log_shortfall", "comfort band", "log", "zone temp, setpoints",
            "log1p(reward shortfall)", (0, 3), "The reward's own per-zone penalty (linear to 5 K, then quadratic). Skew 21 raw; log1p tames the tail."),
    Feature("eco_setback", "comfort band", "0/1", "heating & cooling setpoints",
            "1 if band wider than 5 K", None, "Band width takes only two values (3 K comfort, 9 K eco), so it is a flag."),
    Feature("temp_abs", "comfort band", "5 K", "zone temp",
            "(T - 20 C) / 5", (-4, 4), "Absolute temperature drives losses to outside and neighbours."),
    # -- recent dynamics (encoder history)
    Feature("dtemp_5min", "dynamics", "asinh(K / 0.5)", "zone temp, previous step",
            "asinh((T - T[t-1]) / 0.5 K)", (-4, 4), "Air temperature jumps several K per step when an AHU switches; 0.1-99.9% range is +-110 K/h. asinh keeps sign, is linear near 0 and compresses the tail."),
    Feature("dtemp_1h", "dynamics", "asinh(K)", "zone temp, 1 h ago",
            "asinh(T - T[t-12])", (-4, 4), "The slower trend under the switching spikes."),
    Feature("temp_minus_ema_6h", "dynamics", "5 K", "zone temp history",
            "(T - EMA_6h(T)) / 5", (-4, 4), "Distance from the zone's recent mean: a cheap stand-in for the thermal mass state."),
    Feature("hours_out_of_band", "dynamics", "log h", "zone temp, setpoints history",
            "log1p(hours continuously out of band)", (0, 5), "A zone left out for hours needs more than one that just drifted out; the reward's tail punishes neglect."),
    # -- neighbourhood
    Feature("temp_minus_ahu_mean", "neighbours", "5 K", "zone temps of the same AHU",
            "(T - mean T of AHU's zones) / 5", (-4, 4), "One supply temperature serves all zones on an AHU; a zone's position among them decides whether its damper or the AHU should act."),
    Feature("ahu_worst_log_shortfall", "neighbours", "log", "zone temps and setpoints of the same AHU",
            "log1p(max shortfall on the AHU)", (0, 3), "The AHU-level decision is driven by its worst zone, not its average."),
    Feature("gap_to_ahu_worst", "neighbours", "log", "same",
            "worst - own log-shortfall", (0, 3), "How far this zone is from being the one that matters."),
    Feature("temp_minus_outside", "neighbours", "15 K", "zone temp, outside air",
            "(T - OA) / 15", (-4, 4), "Driving force for envelope heat loss or gain."),
    # -- the zone's own actuators, as the boxes actually took them
    Feature("damper_open", "zone actuators", "fraction above floor", "damper position",
            "(d - 0.1) / 0.9", (0, 1), "The simulator floors dampers at 10%; rescaling makes 0 mean 'at the floor'."),
    Feature("reheat", "zone actuators", "fraction", "reheat valve position",
            "v", (0, 1), "Already a fraction once de-normalised."),
    # -- static geometry
    Feature("log_area", "geometry", "log(m2 / 50)", "floor plan, zone map",
            "log(area / 50 m2)", (-3, 3), "Heat capacity and internal gains scale with area; log because areas span 20-200 m2."),
    Feature("exterior_fraction", "geometry", "fraction", "floor plan, zone map",
            "facade cells / boundary cells", (0, 1), "Exposure to outdoor air."),
    # -- its air handler
    Feature("ahu_run", "air handler", "0/1", "supply fan run", "s", None, "On/off state; gates the readings below."),
    Feature("ahu_supply_temp_on", "air handler", "5 K", "supply air temperature, fan run",
            "run x (SAT - 20 C) / 5", (-4, 4), "A stopped AHU still reports its last command as supply temperature; gating by run stops that from lying."),
    Feature("ahu_sat_minus_zone_on", "air handler", "5 K", "supply air temp, zone temp, run",
            "run x (SAT - T) / 5", (-4, 4), "Sign says whether the supply air heats or cools this zone."),
    Feature("ahu_flow_per_zone", "air handler", "m3/s per zone", "supply airflow sensor, zones served",
            "flow / zones served", (0, 4), "Airflow, normalised by how many zones share it. Static pressure is dropped: r=0.96 with this. Clip raised from 2 after 2% of rows hit it."),
    Feature("ahu_recirculation", "air handler", "fraction of cap", "recirculation command",
            "min(r, 0.7) / 0.7", (0, 1), "The simulator caps recirculation at 70%; commands above it do nothing."),
    Feature("ahu_n_zones", "air handler", "10 zones", "AHU assignment", "zones served / 10", (0, 2), "Static: how widely the AHU's air is shared."),
    Feature("ahu_hours_since_switch", "air handler", "log h", "fan run history",
            "log1p(hours since run changed)", (0, 5), "Zone air reacts within minutes of a switch, then settles; this tells the model which regime it is in."),
    # -- physics interactions
    Feature("air_heat_drive", "physics", "asinh(m3/s K)", "run, airflow, damper, supply & zone temp",
            "asinh(run x flow/zone x damper_open x (SAT - T))", (-5, 5), "Heat the supply air can move into the zone is roughly airflow x damper x temperature difference; no single column holds it."),
    Feature("reheat_drive", "physics", "40 K", "pump, reheat valve, water & zone temp",
            "pump x valve x (Tw - T) / 40", (0, 2), "Heat the reheat coil can add."),
    # -- hot-water loop
    Feature("water_supply_temp_on", "hot water", "20 K", "supply water temp, pump",
            "pump x (Tw - 330 K) / 20", (-2, 2), "Gated by the pump like the AHU readings."),
    Feature("pump_run", "hot water", "0/1", "hot-water run status", "s", None, "Pump state."),
]

GLOBAL_SPEC: list[Feature] = [
    Feature("outside_ema_24h", "weather", "15 K", "outside air history", "(EMA_24h(OA) - 10 C) / 15", (-4, 4), "Climate level: the closest thing to a weather label the policy can have. (The raw observation repeats outdoor temperature per AHU; one value is used.)"),
    Feature("outside_vs_24h", "weather", "5 K", "outside air temp and history", "(OA - EMA_24h(OA)) / 5", (-4, 4), "Current outdoor temperature as a departure from its 24 h mean: the daily swing. Outdoor temperature itself was r>0.95 with the 24 h mean; zones still see it directly in temp_minus_outside."),
    Feature("outside_trend_6h", "weather", "5 K", "outside air history", "(EMA_6h(OA) - EMA_24h(OA)) / 5", (-4, 4), "The envelope lags outdoor temperature by hours. Expressed against the 24 h mean, because the 6 h mean alone was r>0.95 with it."),
    Feature("outside_dtemp_1h", "weather", "asinh(K)", "outside air history", "asinh(OA - OA[t-12])", (-4, 4), "Fronts and the daily swing."),
    Feature("sat_inert", "weather", "0/1", "zone temps, outside air", "1 if 0.7 x mean T + 0.3 x OA < 285 K", None, "The coldest air the AHU can mix is below its lowest setpoint, so the supply-temperature action does nothing (31% of hours in Fairbanks). The margin itself was dropped: r>0.95 with outdoor temperature."),
    Feature("hour_sin", "time", "unit circle", "hod features", "sin(2 pi h / 24)", None, "One pair replaces 12 phase-shifted copies of the same cycle."),
    Feature("hour_cos", "time", "unit circle", "hod features", "cos(2 pi h / 24)", None, "Pair with hour_sin."),
    Feature("dow_sin", "time", "unit circle", "dow features", "sin(2 pi d / 7)", None, "One pair replaces 12 copies."),
    Feature("dow_cos", "time", "unit circle", "dow features", "cos(2 pi d / 7)", None, "Pair with dow_sin."),
    Feature("is_weekend", "time", "0/1", "dow features", "d >= 5", None, "Comfort mode never runs on weekends; a flag is clearer than the cycle."),
    Feature("is_holiday", "time", "0/1", "metadata holidays", "date in holidays", None, "Empty in every building so far, kept so a test building with holidays is handled."),
    Feature("comfort_now", "schedule", "0/1", "comfort_mode_now", "s", None, "Comfort scored now."),
    Feature("comfort_soon", "schedule", "0/1", "comfort_mode_soon", "s", None, "Differs from now only in the hour before 06:00 and 19:00: a one-hour lookahead."),
    Feature("hours_to_comfort", "schedule", "24 h", "time, schedule, weekday", "hours until comfort starts / 24", (0, 3), "How long a pre-heat or pre-cool has, including over weekends."),
    Feature("hours_left_in_comfort", "schedule", "12 h", "time, schedule", "hours until comfort ends / 12", (0, 1.2), "Late in the day, conditioning that won't pay back before 19:00 can stop."),
    Feature("occupants_per_zone", "occupancy", "log", "num_occupants, zones", "log1p(occupants / zones)", (0, 2), "Raw count grows with building size; per zone and log-scaled it compares across buildings."),
    Feature("occupied", "occupancy", "0/1", "num_occupants", "occupants > 0", None, "Comfort is only scored when anyone is in."),
    Feature("occupancy_change_1h", "occupancy", "per zone", "num_occupants history", "(occ - occ[t-12]) / zones", (-2, 2), "Arrivals and departures."),
    Feature("bldg_mean_log_shortfall", "building state", "log", "all zones", "mean log1p(shortfall)", (0, 3), "Building-wide comfort state for plant-level decisions (pump, water temperature)."),
    Feature("bldg_frac_out_of_band", "building state", "fraction", "all zones", "share of zones out of band", (0, 1), "How widespread the problem is."),
    Feature("bldg_worst_log_shortfall", "building state", "log", "all zones", "max log1p(shortfall)", (0, 3), "The single worst zone."),
    Feature("n_zones_log", "building", "log(V / 50)", "observation layout", "log(zones / 50)", (-2, 2), "Building size; static."),
    Feature("n_ahus", "building", "10 AHUs", "AHU assignment", "AHUs / 10", (0, 3), "Plant size; static."),
    Feature("mean_exterior_fraction", "building", "fraction", "floor plan", "mean zone facade share", (0, 1), "How exposed the building is overall; static."),
]

ZONE_FEATURES = [f.name for f in ZONE_SPEC]
GLOBAL_FEATURES = [f.name for f in GLOBAL_SPEC]
FEATURE_VERSION = 2

# Raw observation fields deliberately not used, and why (for the catalog).
DROPPED = {
    "VAV supply_air_flowrate_setpoint": "Constant (2.0) in every building.",
    "AHU supply_air_temperature_setpoint": "Identical to the sensor.",
    "AHU supply_air_static_pressure_setpoint": "Identical to the sensor.",
    "AHU supply_air_static_pressure_sensor": "Tracks airflow (r=0.96); airflow is the physical quantity.",
    "AHU supply_air_flowrate_setpoint": "Identical to the sensor.",
    "AHU outside_air_flowrate_sensor": "Follows airflow x (1 - recirculation).",
    "AHU outside_air_temperature_sensor (all but one)": "The same value repeated per AHU.",
    "AHU supervisor_run_command / exhaust_fan_run_command": "Identical to supply fan run.",
    "AHU supply / discharge fan speed": "Follow static pressure.",
    "AHU cooling_request_count": "Equals zones served whenever the AHU is on (fraction is 1.0 in 99.9% of rows).",
    "water supply_water_temperature_setpoint": "Tracked by the sensor.",
    "water differential_pressure_sensor / setpoint": "Read 0 whenever the pump is off (r>0.95 with pump run); the commanded pump pressure is kept as an action column.",
    "water heating_request_count": "Equals the number of zones.",
    "water supervisor_run_command": "Identical to run status.",
    "hod_* / dow_* copies 001-011": "Phase-shifted copies of the same cycle.",
}


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
        occ = np.zeros((h, w), bool)
        occ[cells[:, 0], cells[:, 1]] = True
        # Boundary cells: a 4-neighbour outside the zone. Facade: that neighbour is exterior wall.
        pad = np.pad(occ, 1)
        epad = np.pad(ext, 1)
        r, c = cells[:, 0] + 1, cells[:, 1] + 1
        nb_out = np.zeros(len(cells), bool)
        nb_ext = np.zeros(len(cells), bool)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb_out |= ~pad[r + dr, c + dc]
            nb_ext |= epad[r + dr, c + dc]
        n_boundary = int(nb_out.sum())
        out["exterior_fraction"][i] = float((nb_out & nb_ext).sum()) / max(n_boundary, 1)
    return out


def _ema_alpha(hours: float) -> float:
    return 1.0 - np.exp(-1.0 / (hours * STEPS_PER_HOUR))


A6, A24 = _ema_alpha(6), _ema_alpha(24)
_VAV = re.compile(r"^VAV_(\d+)_(.+)$")
_AHU = re.compile(r"^ahs_(ahu_\d+)_(.+)$")


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
        self.i_recirc = acol("recirculation_percentage_command")
        self.i_flow = acol("supply_air_flowrate_sensor")
        self.i_run = acol("supply_fan_run_command")
        self.i_oa = ix[f"ahs_{self.ahu_names[0]}_outside_air_temperature_sensor"]
        self.i_wt = ix["hws_supply_water_temperature_sensor"]
        self.i_wdp = ix["hws_differential_pressure_setpoint"]
        self.i_pump = ix["hws_run_status"]
        self.i_hod = (ix["hod_sin_000"], ix["hod_cos_000"])
        self.i_dow = (ix["dow_sin_000"], ix["dow_cos_000"])
        self.i_cnow, self.i_csoon = ix["comfort_mode_now"], ix["comfort_mode_soon"]
        self.i_occ = ix["num_occupants"]

        self.zone_ahu = np.array([self.ahu_names.index(z2a[z]) for z in self.zone_ids])
        self.ahu_n_zones = np.bincount(self.zone_ahu, minlength=K).astype(np.float64)
        self.morning = int(meta.get("morning_start_hour", 6))
        self.evening = int(meta.get("evening_start_hour", 19))
        self.holidays = {str(h)[:10] for h in (meta.get("holidays") or [])}

        geo = geometry if geometry is not None else zone_geometry(meta, self.zone_ids)
        self.log_area = geo["log_area"].astype(np.float64)
        self.exterior_fraction = geo["exterior_fraction"].astype(np.float64)
        self._lo_z = np.array([f.clip[0] if f.clip else -np.inf for f in ZONE_SPEC])
        self._hi_z = np.array([f.clip[1] if f.clip else np.inf for f in ZONE_SPEC])
        self._lo_g = np.array([f.clip[0] if f.clip else -np.inf for f in GLOBAL_SPEC])
        self._hi_g = np.array([f.clip[1] if f.clip else np.inf for f in GLOBAL_SPEC])
        self.reset()

    def reset(self) -> None:
        self._temp_hist: deque[np.ndarray] = deque(maxlen=STEPS_PER_HOUR + 1)
        self._oa_hist: deque[float] = deque(maxlen=STEPS_PER_HOUR + 1)
        self._occ_hist: deque[float] = deque(maxlen=STEPS_PER_HOUR + 1)
        self._t_ema = None
        self._oa_ema6 = self._oa_ema24 = None
        self._out_steps = np.zeros(self.V)
        self._run_prev = None
        self._since_switch = np.zeros(self.K)

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

    def encode(self, obs: np.ndarray, timestamp: float | None = None) -> dict[str, np.ndarray]:
        """Encode one observation. Call once per step, in order (keeps history)."""
        obs = np.asarray(obs, dtype=np.float64)
        T = self.zone_temp_k(obs)
        h, c = obs[self.i_heat], obs[self.i_cool]
        oa = float(obs[self.i_oa])
        occ = float(obs[self.i_occ])

        # history
        self._temp_hist.append(T)
        self._oa_hist.append(oa)
        self._occ_hist.append(occ)
        prev, hour_ago = (self._temp_hist[-2] if len(self._temp_hist) > 1 else T), self._temp_hist[0]
        self._t_ema = T.copy() if self._t_ema is None else self._t_ema + A6 * (T - self._t_ema)
        self._oa_ema6 = oa if self._oa_ema6 is None else self._oa_ema6 + A6 * (oa - self._oa_ema6)
        self._oa_ema24 = oa if self._oa_ema24 is None else self._oa_ema24 + A24 * (oa - self._oa_ema24)

        dev = np.maximum(h - T, T - c)
        short = comfort_shortfall(dev)
        lshort = np.log1p(short)
        out = dev > 0
        self._out_steps = np.where(out, self._out_steps + 1, 0)

        a = self.zone_ahu
        n_served = self.ahu_n_zones[a]
        ahu_mean = np.bincount(a, weights=T, minlength=self.K) / np.maximum(self.ahu_n_zones, 1)
        ahu_worst = np.zeros(self.K)
        np.maximum.at(ahu_worst, a, lshort)

        run = (obs[self.i_run] > 0.5).astype(np.float64)
        if self._run_prev is None:
            self._since_switch[:] = 0
        else:
            self._since_switch = np.where(run != self._run_prev, 0, self._since_switch + 1)
        self._run_prev = run
        sat = obs[self.i_sat]
        flow_pz = obs[self.i_flow][a] / n_served
        damper_open = np.clip((obs[self.i_damper] * FRAC_STD + FRAC_MEAN - DAMPER_FLOOR) / (1 - DAMPER_FLOOR), 0, 1)
        reheat = np.clip(obs[self.i_reheat] * FRAC_STD + FRAC_MEAN, 0, 1)
        pump = float(obs[self.i_pump] * FRAC_STD + FRAC_MEAN > 0.5)
        tw = obs[self.i_wt]
        run_z = run[a]

        zone = np.stack([
            (T - (h + c) / 2) / COMFORT_KNEE_K,
            np.maximum(0, h - T) / COMFORT_KNEE_K,
            np.maximum(0, T - c) / COMFORT_KNEE_K,
            out.astype(np.float64),
            lshort,
            ((c - h) > 5.0).astype(np.float64),
            (T - T_REF_K) / COMFORT_KNEE_K,
            np.arcsinh((T - prev) / 0.5),
            np.arcsinh(T - hour_ago),
            (T - self._t_ema) / COMFORT_KNEE_K,
            np.log1p(self._out_steps / STEPS_PER_HOUR),
            (T - ahu_mean[a]) / COMFORT_KNEE_K,
            ahu_worst[a],
            ahu_worst[a] - lshort,
            (T - oa) / OA_SCALE,
            damper_open,
            reheat,
            self.log_area,
            self.exterior_fraction,
            run_z,
            run_z * (sat[a] - T_REF_K) / COMFORT_KNEE_K,
            run_z * (sat[a] - T) / COMFORT_KNEE_K,
            flow_pz,
            np.minimum(obs[self.i_recirc][a], MAX_RECIRC) / MAX_RECIRC,
            n_served / 10.0,
            np.log1p(self._since_switch[a] / STEPS_PER_HOUR),
            np.arcsinh(run_z * flow_pz * damper_open * (sat[a] - T)),
            pump * reheat * (tw - T) / 40.0,
            np.full(self.V, pump * (tw - WATER_T_REF) / WATER_T_SCALE),
            np.full(self.V, pump),
        ], axis=1)
        zone = np.clip(zone, self._lo_z, self._hi_z).astype(np.float32)

        hs, hc = obs[self.i_hod[0]], obs[self.i_hod[1]]
        ds, dc = obs[self.i_dow[0]], obs[self.i_dow[1]]
        hour = (np.degrees(np.arctan2(hs, hc)) / 15.0) % 24
        dow = int(round(np.degrees(np.arctan2(ds, dc)) / (360 / 7))) % 7
        is_holiday = 0.0
        if timestamp is not None and self.holidays:
            import datetime as _dt
            is_holiday = float(_dt.datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d") in self.holidays)
        cnow = float(obs[self.i_cnow])
        mixed = MAX_RECIRC * float(T.mean()) + (1 - MAX_RECIRC) * oa
        left = max(0.0, self.evening - hour) if cnow > 0.5 else 0.0
        glob = np.array([
            (self._oa_ema24 - OA_REF_K) / OA_SCALE,
            (oa - self._oa_ema24) / COMFORT_KNEE_K,
            (self._oa_ema6 - self._oa_ema24) / COMFORT_KNEE_K,
            np.arcsinh(oa - self._oa_hist[0]),
            float(mixed < MIN_SAT_K),
            hs, hc, ds, dc,
            float(dow >= 5), is_holiday,
            cnow, float(obs[self.i_csoon]),
            self._hours_to_comfort(hour, dow, cnow) / 24.0,
            left / 12.0,
            np.log1p(occ / self.V),
            float(occ > 0),
            (occ - self._occ_hist[0]) / self.V,
            float(lshort.mean()), float(out.mean()), float(lshort.max()),
            np.log(self.V / 50.0), self.K / 10.0, float(self.exterior_fraction.mean()),
        ])
        glob = np.clip(glob, self._lo_g, self._hi_g).astype(np.float32)
        return {"zone": zone, "glob": glob, "temp_k": T.astype(np.float32),
                "shortfall": short.astype(np.float32)}


class FeatureScaler:
    """Optional robust standardisation, fitted on the train split only.

    (x - median) / IQR per column, clipped to +-5, for models that want unit
    scale (neural networks, linear models). 0/1 flags and columns whose IQR
    is 0 (mostly-zero hinges) are scaled by their standard deviation instead,
    or left alone if that is 0 too. Saved as JSON so a submission can ship it.
    """

    def __init__(self, center: dict[str, float] | None = None, scale: dict[str, float] | None = None):
        self.center, self.scale = center or {}, scale or {}

    def fit(self, df, columns: list[str]) -> "FeatureScaler":
        for col in columns:
            v = df[col].to_numpy(np.float64)
            uniq = np.unique(v[: 100_000])
            if len(uniq) <= 2:
                self.center[col], self.scale[col] = 0.0, 1.0
                continue
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            iqr = q3 - q1
            sd = v.std()
            self.center[col] = float(med)
            self.scale[col] = float(iqr / 1.349 if iqr > 1e-6 else (sd if sd > 1e-6 else 1.0))
        return self

    def transform(self, df, columns: list[str]):
        out = df.copy()
        for col in columns:
            out[col] = np.clip((df[col] - self.center[col]) / self.scale[col], -5, 5).astype(np.float32)
        return out

    def to_json(self) -> dict:
        return {"center": self.center, "scale": self.scale}

    @classmethod
    def from_json(cls, d: dict) -> "FeatureScaler":
        return cls(d["center"], d["scale"])


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
