"""Load rollouts and reshape them into building-size-independent tables.

Observation and action vectors change length from building to building, so
a model that has to transfer cannot consume them raw. :func:`zone_table`
turns one rollout into one row per (step, zone): the zone's own sensors and
commands, its air handler's, the hot-water loop's, and the building-wide
signals. A model with shared weights across zones can then train on every
building at once, and run unchanged on a building with any number of zones.

Example:
    from hvacdata.dataset import load_rollout, zone_table
    r = load_rollout("data/rollouts/kit/building_train_1/fairbanks_1/pid_water.npz")
    tab = zone_table(r)              # dict of column -> (T*V,) arrays
    import pandas as pd; df = pd.DataFrame(tab)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from hvacdata import paths

_VAV = re.compile(r"^VAV_(\d+)_(.+)$")
_AHU = re.compile(r"^ahs_(ahu_\d+)_(.+)$")
_HWS = re.compile(r"^hws_(.+)$")


def load_rollout(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = paths.REPO_ROOT / p
    arrays = dict(np.load(p))
    with open(p.with_suffix(".json")) as f:
        info = json.load(f)
    return {**arrays, **info}


def obs_layout(field_names: list[str]) -> dict:
    """Index tables over the flat observation, keyed by device and field."""
    vav: dict[str, dict[str, int]] = {}
    ahu: dict[str, dict[str, int]] = {}
    hws: dict[str, int] = {}
    glob: dict[str, int] = {}
    for i, n in enumerate(field_names):
        if m := _VAV.match(n):
            vav.setdefault(f"VAV_{m.group(1)}", {})[m.group(2)] = i
        elif m := _AHU.match(n):
            ahu.setdefault(m.group(1), {})[m.group(2)] = i
        elif m := _HWS.match(n):
            hws[m.group(1)] = i
        else:
            glob[n] = i
    return {"vav": vav, "ahu": ahu, "hws": hws, "global": glob}


def zone_table(r: dict, include_next: bool = True) -> dict[str, np.ndarray]:
    """One row per (step, zone). Columns are prefixed by where they come from:

    z_*   this zone's observation fields          (obs at t)
    a_*   this zone's AHU observation fields      (obs at t)
    w_*   hot-water loop observation fields       (obs at t)
    g_*   building-wide fields: occupancy, comfort flags, time encodings
    act_* actions at t: this zone's damper/reheat, its AHU's setpoints and run
          command, the water setpoints and pump run command
    next_z_zone_air_temperature_sensor   the zone's normalised temperature at t+1
    zone_temp_k / outside_air_k           raw kelvin after the step
    reward / rc_*                         building-level, repeated per zone
    """
    meta = r["meta"]
    lay = obs_layout(meta["observation_field_names"])
    obs = r["obs"]
    T = len(r["reward"])
    ca, da = r["continuous_action"], r["discrete_action"]

    ahu_names = list(meta["ahu_zone_assignments"])  # positional AHU order
    zone_to_ahu = {room.replace("room_", "VAV_"): k
                   for k, rooms in meta["ahu_zone_assignments"].items() for room in rooms}
    zone_ids = r["zone_ids"]  # column order of zone_temp_k

    cols: dict[str, list[np.ndarray]] = {}

    def add(name: str, v: np.ndarray) -> None:
        cols.setdefault(name, []).append(np.asarray(v))

    t_idx = np.arange(T)
    for zi, z in enumerate(zone_ids):
        ahu = zone_to_ahu[z]
        k = ahu_names.index(ahu)
        add("t", t_idx)
        add("zone", np.full(T, zi))
        add("ahu", np.full(T, k))
        for f, i in lay["vav"][z].items():
            add(f"z_{f}", obs[:T, i])
            if include_next and f == "zone_air_temperature_sensor":
                add(f"next_z_{f}", obs[1:T + 1, i])
        for f, i in lay["ahu"][ahu].items():
            add(f"a_{f}", obs[:T, i])
        for f, i in lay["hws"].items():
            add(f"w_{f}", obs[:T, i])
        for f, i in lay["global"].items():
            add(f"g_{f}", obs[:T, i])
        za = meta["zone_action_indices"][z]
        add("act_damper", ca[:, za["supply_air_damper_percentage_command"]])
        add("act_reheat", ca[:, za["reheat_valve_percentage_command"]])
        add("act_ahu_supply_temp", ca[:, meta["temp_indices"][k]])
        add("act_ahu_static_pressure", ca[:, meta["pressure_indices"][k]])
        add("act_ahu_recirculation", ca[:, meta["recirculation_indices"][k]])
        add("act_ahu_run", da[:, k])
        add("act_water_temp", ca[:, meta["water_indices"][0]])
        # The kit files the pump's differential_pressure_setpoint under
        # pressure_indices (its name contains "pressure"), after the K AHUs.
        add("act_water_dp", ca[:, meta["pressure_indices"][len(ahu_names)]])
        add("act_pump_run", da[:, -1])
        add("zone_temp_k", r["zone_temp_k"][:, zi])
        add("outside_air_k", r["outside_air_k"])
        add("reward", r["reward"])
        for key in r:
            if key.startswith("rc_"):
                add(key, r[key])
    return {k: np.concatenate(v) for k, v in cols.items()}
