#!/usr/bin/env python3
"""Write data/collection_plan.csv from configs/collection.yaml.

Each train building gets `episodes_per_building` of its weather windows, chosen
to cover different stations, and behaviours are dealt out so every behaviour
sees every climate group and building size. Rows are ordered longest-first
(estimated from zone count) so parallel workers finish together.
"""

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvacdata import paths  # noqa: E402
from hvacdata.rollouts import BEHAVIOURS  # noqa: E402
from hvacdata.scenarios import load_splits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "collection.yaml")
    ap.add_argument("--out", type=Path, default=paths.DATA_ROOT / "collection_plan.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    rng = random.Random(cfg["seed"])
    unknown = set(cfg["behaviours"]) - set(BEHAVIOURS)
    if unknown:
        sys.exit(f"unknown behaviours {unknown}; known: {sorted(BEHAVIOURS)}")

    entries = load_splits()[cfg["split"]]
    zones = {r["name"]: int(r["n_zones"]) for r in csv.DictReader(open(paths.BUILDINGS_DIR / "buildings.csv"))}
    by_b: dict[str, list[dict]] = {}
    for e in entries:
        by_b.setdefault(e["building"], []).append(e)

    chosen = []
    for b, es in sorted(by_b.items()):
        # one window per station first, then repeats, in random order
        rng.shuffle(es)
        seen, first, rest = set(), [], []
        for e in es:
            (rest if e["station"] in seen else first).append(e)
            seen.add(e["station"])
        n = cfg["episodes_per_building"]
        if zones.get(b, 0) > cfg.get("large_building_zones", 10**9):
            n = cfg.get("episodes_per_large_building", n)
        chosen += (first + rest)[:n]

    # Deal behaviours in proportion to their weights, shuffled.
    w = cfg["behaviours"]
    pool = [b for b, k in w.items() for _ in range(k)]
    deck = []
    while len(deck) < len(chosen):
        rng.shuffle(pool)
        deck += pool
    deck = deck[: len(chosen)]
    # Shuffle the episode order before dealing so behaviour isn't tied to building order.
    rng.shuffle(chosen)

    rows = []
    for i, (e, b) in enumerate(zip(chosen, deck)):
        rows.append({"building": e["building"], "weather": e["weather"], "station": e["station"],
                     "scenario_dir": e["scenario_dir"], "behaviour": b, "steps": cfg["steps"],
                     "hold": cfg["hold"], "seed": cfg["seed"] * 100_000 + i,
                     "n_zones": zones.get(e["building"], 0), "tag": cfg["tag"]})
    rows.sort(key=lambda r: -r["n_zones"])
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    print(f"Wrote {len(rows)} episodes to {args.out}")
    print("by behaviour:", dict(Counter(r["behaviour"] for r in rows)))
    print("by station:  ", dict(Counter(r["station"] for r in rows)))
    zone_steps = sum(r["n_zones"] * r["steps"] for r in rows)
    print(f"zone-steps: {zone_steps / 1e6:.1f} M; est. {sum(r['steps'] * (0.03 + 0.0014 * r['n_zones']) for r in rows) / 3600:.0f} core-hours")


if __name__ == "__main__":
    main()
