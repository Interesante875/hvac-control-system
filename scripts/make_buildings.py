#!/usr/bin/env python3
"""Sample train and val buildings with the kit's generator.

Writes data/buildings/building_<seed>.yaml and data/buildings/buildings.csv.
"""

import argparse
import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvacdata import paths  # noqa: E402
from hvacdata.buildings import generate_buildings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "data.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))["buildings"]

    rows = []
    for split in ("train", "val"):
        s = cfg[split]
        recs = generate_buildings(s["start_seed"], s["count"], cfg, paths.BUILDINGS_DIR, split)
        for r in recs:
            r["path"] = str(Path(r["path"]).relative_to(paths.REPO_ROOT))
        rows += recs
        print(f"  {split}: {len(recs)} buildings (seeds {recs[0]['seed']}..{recs[-1]['seed']}), "
              f"{sum(r['n_zones'] for r in recs)} zones total")

    out = paths.BUILDINGS_DIR / "buildings.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} building configs -> {out}")


if __name__ == "__main__":
    main()
