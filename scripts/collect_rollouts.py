#!/usr/bin/env python3
"""Run controllers on a split and record full transitions (see hvacdata/rollouts.py).

    # Baseline reference scores on the val split (mirrors the hidden test)
    python scripts/collect_rollouts.py --split val --policies pid_water pid fixed_setpoint bang_bang

    # Training data: the best baseline, plus exploratory variants of it
    python scripts/collect_rollouts.py --split train --policies pid_water
    python scripts/collect_rollouts.py --split train --policies pid_water --sigma 0.3 --p-flip 0.05

    # Quick smoke test (NOT scoreable: only full 8640-step episodes compare)
    python scripts/collect_rollouts.py --split kit --limit 2 --steps 288 --tag smoke

    # Your own controller, by import path
    python scripts/collect_rollouts.py --split val --policies mypkg.policy:MyPolicy
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvacdata import paths  # noqa: E402
from hvacdata.rollouts import build_index, collect  # noqa: E402
from hvacdata.scenarios import load_splits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["kit", "train", "val"])
    ap.add_argument("--policies", nargs="+", default=["pid_water"])
    ap.add_argument("--steps", type=int, default=paths.STEPS_PER_EPISODE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N scenarios of the split")
    ap.add_argument("--buildings", nargs="*", help="restrict to these buildings")
    ap.add_argument("--stations", nargs="*", help="restrict to these weather stations")
    ap.add_argument("--sigma", type=float, default=0.0, help="continuous exploration noise std")
    ap.add_argument("--p-flip", type=float, default=0.0, help="run-command flip prob per hold")
    ap.add_argument("--hold", type=int, default=12, help="steps each perturbation is held (12 = 1h)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="output subdir under data/rollouts (default: split)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    entries = load_splits()[args.split]
    if args.buildings:
        entries = [e for e in entries if e["building"] in args.buildings]
    if args.stations:
        entries = [e for e in entries if e["station"] in args.stations]
    if args.limit:
        entries = entries[: args.limit]
    missing = [e for e in entries if not (paths.REPO_ROOT / e["scenario_dir"] / "config.gin").exists()]
    if missing:
        sys.exit(f"{len(missing)} scenario(s) not generated yet, e.g. {missing[0]['scenario_dir']}. "
                 f"Run scripts/make_scenarios.py --split {args.split}")

    tag = args.tag or args.split
    collect(entries, args.policies, tag, args.steps, args.workers,
            sigma=args.sigma, p_flip=args.p_flip, hold=args.hold, seed=args.seed, force=args.force)

    df = build_index(tag)
    if len(df):
        full = df[df["full_episode"]]
        print(f"\nIndex: {paths.ROLLOUTS_DIR / tag / 'index.csv'} ({len(df)} rollouts)")
        if len(full):
            print("Mean per-step loss over full episodes (leaderboard metric; lower is better):")
            print(full.groupby(["policy", "sigma", "p_flip"])["score"].agg(["mean", "max", "count"])
                  .round(4).to_string())
        else:
            print("No full-length episodes, so no comparable scores.")


if __name__ == "__main__":
    main()
