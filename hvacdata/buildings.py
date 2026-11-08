"""Procedural building configs, sampled with the kit's own generator."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import yaml

from hvacdata import paths


def _ranges(profile: dict) -> dict:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in profile.items()}


def _pick_profiles(seeds: list[int], mix: dict[str, float], salt: str) -> list[str]:
    """Assign each seed a size profile, deterministically, matching the mix."""
    names = sorted(mix)
    counts = {n: int(round(mix[n] * len(seeds))) for n in names}
    # Fix rounding so the counts add up.
    diff = len(seeds) - sum(counts.values())
    counts[max(names, key=lambda n: mix[n])] += diff
    labels = [n for n in names for _ in range(counts[n])]
    random.Random(f"{salt}:{seeds}").shuffle(labels)
    return labels


def check_floor_plan(building_yaml: Path) -> tuple[int, int]:
    """Build the floor plan exactly as scenario generation will, and validate it.

    Returns (n_zones, n_ahus). Raises ValueError for a plan the kit would refuse
    (zones under 20 m2: the kit rejects these because they diverge in
    simulation). The floor plan depends on the building alone, so any weather
    preset will do here.
    """
    import math

    import numpy as np
    from smart_buildings.smart_control.utils import scenario_generator as sg

    with paths.in_kit():
        cfg = sg.load_scenario_from_parts(str(building_yaml),
                                          str(paths.KIT_WEATHER / "fairbanks_1.yaml"))
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        gen = sg._create_floor_plan_generator(cfg.floor_plan)
        gen.generate()
        sg._check_generated_zone_areas(gen, sg.CONTROL_VOLUME_CM, cfg.seed)
        labels, _ = gen.get_room_labels()
    n_zones = int(len(np.unique(np.asarray(labels))) - 1)  # label 0 is background
    n_ahus = max(int(cfg.ahu.num_ahus), math.ceil(n_zones / cfg.ahu.max_zones_per_ahu))
    return n_zones, n_ahus


def generate_buildings(start_seed: int, count: int, cfg: dict, out_dir: Path,
                       split: str, max_tries: int = 10) -> list[dict]:
    """Write ``count`` valid ``building_<seed>.yaml`` files; return their records.

    Seeds run upward from ``start_seed``; a seed whose floor plan the kit would
    reject is skipped, as the kit's own error message advises.
    """
    paths.ensure_kit_importable()
    from smart_buildings.smart_control.utils.config_generator import generate_config
    from smart_buildings.smart_control.utils.materials import (
        draw_materials, envelope_capacity, facade_u)

    slots = list(range(count))
    profiles = _pick_profiles(slots, cfg["profile_mix"], f"{split}:{start_seed}")
    # Materials are Latin-hypercube stratified across the batch, as the kit does.
    digest = hashlib.sha256(repr((split, start_seed, count)).encode()).digest()
    materials = draw_materials(random.Random(int.from_bytes(digest[:8], "big")), count)

    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seed = start_seed
    for profile, mats in zip(profiles, materials):
        for _ in range(max_tries):
            name = f"building_{seed:05d}"
            config = generate_config(
                name=name, seed=seed, building_shape=cfg.get("shape", "composite"),
                composite_style="architectural", **_ranges(cfg["profiles"][profile]),
            )
            config["materials"] = mats
            path = out_dir / f"{name}.yaml"
            with open(path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            seed += 1
            try:
                n_zones, n_ahus = check_floor_plan(path)
                break
            except ValueError as exc:
                print(f"    skip {name}: {str(exc).split('.')[0]}")
                path.unlink()
        else:
            raise RuntimeError(f"no valid floor plan in {max_tries} seeds near {seed}")
        fp = config["floor_plan"]
        records.append({
            "name": name, "seed": config["seed"], "split": split, "profile": profile,
            "path": str(path), "width": fp["width"], "height": fp["height"],
            "num_floors": fp["num_floors"], "n_zones": n_zones, "n_ahus": n_ahus,
            "facade_u": round(facade_u(mats), 3),
            "envelope_c_mj": round(envelope_capacity(mats) / 1e6, 3),
        })
    return records
