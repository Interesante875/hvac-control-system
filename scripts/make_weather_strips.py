#!/usr/bin/env python3
"""Cut 30-day weather strips from every station, through the whole year.

Writes data/weather/strips/<station>_dMMDD.yaml and a manifest
data/weather/strips.csv with each strip's split and outdoor-temperature stats.
"""

import argparse
import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvacdata import paths, weather  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "data.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))["weather"]
    st_cfg = weather.load_station_config()

    rows = []
    for st in weather.list_stations(st_cfg):
        for start in weather.strip_start_dates(st_cfg["year"], cfg["every_days"], cfg.get("offset_days", 0)):
            rec = weather.write_strip(st, start, paths.WEATHER_STRIP_DIR)
            rec.update(weather.strip_stats(st.csv_path, rec["start"]))
            rec["convection_coefficient"] = st.convection_coefficient
            rec["path"] = str(Path(rec["path"]).relative_to(paths.REPO_ROOT))
            rows.append(rec)
        print(f"  {st.name:14s} {st.split:5s} conv={st.convection_coefficient:5.1f}")

    out = paths.DATA_ROOT / "weather" / "strips.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_val = sum(r["split"] == "val" for r in rows)
    print(f"Wrote {len(rows)} strips ({len(rows) - n_val} train, {n_val} val) -> {out}")


if __name__ == "__main__":
    main()
