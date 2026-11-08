#!/usr/bin/env python3
"""Encode recorded rollouts into Parquet feature tables (see hvacdata/build.py).

    python scripts/build_features.py --tag val
    python scripts/build_features.py --tag analysis --force

Output: data/features/<tag>/{zone,step,zones_static}/*.parquet and
feature_spec.json. Read with pandas.read_parquet("data/features/<tag>/zone").
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvacdata.build import build  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="subdirectory of data/rollouts")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--version", type=int, default=2, help="feature encoder version (1 kept for comparison)")
    ap.add_argument("--out-tag", help="write under data/features/<out-tag> instead of <tag>")
    args = ap.parse_args()
    build(args.tag, workers=args.workers, force=args.force, version=args.version, out_tag=args.out_tag)


if __name__ == "__main__":
    main()
