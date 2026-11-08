#!/usr/bin/env python3
"""Run the episodes in data/collection_plan.csv. Resumable: finished ones are skipped.

    python scripts/collect_plan.py                  # everything, all cores
    python scripts/collect_plan.py --limit 20       # first 20 (longest) episodes
    python scripts/collect_plan.py --behaviours pid_water_plant_random

Output: data/rollouts/<tag>/<building>/<weather>/<behaviour>.npz (+ .json).
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvacdata import paths  # noqa: E402
from hvacdata.rollouts import build_index, run_one  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, default=paths.DATA_ROOT / "collection_plan.csv")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int)
    ap.add_argument("--behaviours", nargs="*")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.plan)))
    if args.behaviours:
        rows = [r for r in rows if r["behaviour"] in args.behaviours]
    if args.limit:
        rows = rows[: args.limit]
    todo = []
    for r in rows:
        out = paths.ROLLOUTS_DIR / r["tag"] / r["building"] / r["weather"] / f"{r['behaviour']}.npz"
        if out.with_suffix(".json").exists():
            continue
        out.unlink(missing_ok=True)  # a half-written file from an interrupted run
        todo.append((r, out))
    print(f"{len(rows) - len(todo)} done, {len(todo)} to run on {args.workers} workers", flush=True)

    t0 = time.time()
    tag = rows[0]["tag"] if rows else "diverse"
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
        futs = {pool.submit(run_one, r["scenario_dir"], r["behaviour"], str(out), int(r["steps"]),
                            0.0, 0.0, int(r["hold"]), int(r["seed"]), r["behaviour"]): r
                for r, out in todo}
        for i, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            try:
                s = f.result()
                msg = f"score={s['score']:.4f} {s['seconds']:.0f}s"
            except Exception as exc:
                msg = f"FAILED {exc!r}"
            el = (time.time() - t0) / 60
            print(f"[{i}/{len(todo)} {el:.0f} min] {r['building']}/{r['weather']} {r['behaviour']}: {msg}",
                  flush=True)
    if todo:
        build_index(tag)


if __name__ == "__main__":
    main()
