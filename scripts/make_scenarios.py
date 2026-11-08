#!/usr/bin/env python3
"""Pair buildings with weather strips into splits, then generate the scenarios.

    python scripts/make_scenarios.py                 # write splits.yaml, generate all
    python scripts/make_scenarios.py --split val     # generate only the val split
    python scripts/make_scenarios.py --splits-only   # just (re)write data/splits.yaml

Splits:
  kit    the starter kit's 30 shipped scenarios (already generated in the kit)
  train  generated train buildings x strips from train stations
  val    generated val buildings x strips from val stations (both unseen in
         train), shaped like the hidden test: 6 buildings x 5 stations
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvacdata import paths  # noqa: E402
from hvacdata.scenarios import load_splits, make_splits, materialize  # noqa: E402


def read_csv(p: Path) -> list[dict]:
    with open(p) as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "data.yaml")
    ap.add_argument("--split", choices=["train", "val"], nargs="*", default=["train", "val"])
    ap.add_argument("--splits-only", action="store_true")
    ap.add_argument("--keep-splits", action="store_true",
                    help="reuse the existing data/splits.yaml instead of rewriting it")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.keep_splits and paths.SPLITS_FILE.exists():
        splits = load_splits()
    else:
        cfg = yaml.safe_load(open(args.config))["splits"]
        buildings = read_csv(paths.BUILDINGS_DIR / "buildings.csv")
        strips = read_csv(paths.DATA_ROOT / "weather" / "strips.csv")
        splits = make_splits(buildings, strips, cfg)
        with open(paths.SPLITS_FILE, "w") as f:
            yaml.safe_dump(splits, f, sort_keys=False)
        print(f"Wrote {paths.SPLITS_FILE}")
    for name, entries in splits.items():
        st = Counter(e["station"] for e in entries)
        print(f"  {name:5s}: {len(entries):4d} scenarios, {len({e['building'] for e in entries})} buildings, "
              f"{len(st)} stations")

    if args.splits_only:
        return
    for name in args.split:
        print(f"\nGenerating {name} scenarios")
        materialize(splits[name], workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
