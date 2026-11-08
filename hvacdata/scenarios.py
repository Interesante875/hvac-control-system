"""Splits (which building meets which weather strip) and scenario materialization."""

from __future__ import annotations

import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from hvacdata import paths


def _rel(p: Path | str) -> str:
    """Path relative to the repo root, for portable manifests."""
    return str(Path(p).resolve().relative_to(paths.REPO_ROOT))


def kit_split() -> list[dict]:
    """The kit's 30 shipped training scenarios, unchanged."""
    with open(paths.KIT_TRAINING_SET) as f:
        pairs = yaml.safe_load(f)["scenarios"]
    return [{
        "building": p["building"], "weather": p["weather"],
        "station": p["weather"].rsplit("_", 1)[0],
        "scenario_dir": _rel(paths.KIT_SCENARIOS / p["building"] / p["weather"]),
    } for p in pairs]


def make_splits(buildings: list[dict], strips: list[dict], cfg: dict) -> dict:
    """Pair buildings with strips.

    train: each train building gets ``train_strips_per_building`` strips from
           train stations, cycling through stations in a shuffled order so no
           station repeats for a building until every station has been used.
    val:   each val building gets ``val_strips_per_station`` strips from every
           val station. Val buildings and val stations never appear in train.
    """
    rng = random.Random(cfg.get("seed", 0))
    by_station: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for s in strips:
        by_station[s["split"]][s["station"]].append(s)

    def entry(b: dict, s: dict) -> dict:
        return {"building": b["name"], "weather": s["name"], "station": s["station"],
                "building_yaml": _rel(b["path"]), "weather_yaml": _rel(s["path"]),
                "scenario_dir": _rel(paths.SCENARIOS_DIR / b["name"] / s["name"])}

    out: dict[str, list[dict]] = {"kit": kit_split(), "train": [], "val": []}

    train_stations = sorted(by_station["train"])
    n = cfg["train_strips_per_building"]
    for b in (b for b in buildings if b["split"] == "train"):
        chosen: list[dict] = []
        while len(chosen) < n:
            order = train_stations[:]
            rng.shuffle(order)
            for st in order[: n - len(chosen)]:
                chosen.append(rng.choice(by_station["train"][st]))
        out["train"] += [entry(b, s) for s in chosen]

    for b in (b for b in buildings if b["split"] == "val"):
        for st in sorted(by_station["val"]):
            for s in rng.sample(by_station["val"][st], cfg["val_strips_per_station"]):
                out["val"].append(entry(b, s))
    return out


def load_splits() -> dict:
    with open(paths.SPLITS_FILE) as f:
        return yaml.safe_load(f)


def _materialize(building_yaml: str, weather_yaml: str) -> str:
    with paths.in_kit():
        from smart_buildings.smart_control.utils.scenario_generator import (
            generate_scenario_from_config, load_scenario_from_parts)
        config = load_scenario_from_parts(
            str(paths.REPO_ROOT / building_yaml), str(paths.REPO_ROOT / weather_yaml),
            # Relative to the kit root: the generator bakes this into config.gin,
            # and relative paths keep the tree portable between machines.
            output_base_dir=paths.kit_relative(paths.SCENARIOS_DIR),
        )
        result = generate_scenario_from_config(config)
        return result["output_dir"]


def materialize(entries: list[dict], workers: int = 4, force: bool = False) -> list[str]:
    """Generate config.gin + floor plan + zone map for each split entry."""
    todo = [e for e in entries if "building_yaml" in e and
            (force or not (paths.REPO_ROOT / e["scenario_dir"] / "config.gin").exists())]
    print(f"{len(entries) - len(todo)} scenario(s) already present, {len(todo)} to generate")
    done = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_materialize, e["building_yaml"], e["weather_yaml"]): e for e in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            e = futs[fut]
            fut.result()
            done.append(e["scenario_dir"])
            print(f"  [{i}/{len(todo)}] {e['building']} + {e['weather']}")
    saved = dedupe_floor_plans()
    if saved:
        print(f"Hardlinked identical floor plans / zone maps, saving {saved / 1e9:.2f} GB")
    return done


def dedupe_floor_plans(root: Path = paths.SCENARIOS_DIR) -> int:
    """Hardlink identical .npy files within each building (they depend on the
    building only, so every weather strip of a building carries a copy)."""
    import filecmp
    import os

    saved = 0
    for bdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for fname in ("floor_plan.npy", "zone_map.npy"):
            files = sorted(bdir.glob(f"*/{fname}"))
            if len(files) < 2:
                continue
            ref = files[0]
            for f in files[1:]:
                if os.path.samefile(ref, f) or not filecmp.cmp(ref, f, shallow=False):
                    continue
                tmp = f.with_suffix(".tmp")
                os.link(ref, tmp)
                os.replace(tmp, f)
                saved += ref.stat().st_size
    return saved
