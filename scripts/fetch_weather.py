#!/usr/bin/env python3
"""Download extra weather stations (configs/stations.yaml `extra:`) from Open-Meteo.

    python scripts/fetch_weather.py              # fetch missing stations
    python scripts/fetch_weather.py --force      # re-download all
    python scripts/fetch_weather.py --only seattle miami
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvacdata import paths, weather  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    cfg = weather.load_station_config()
    year = cfg["year"]
    for name, spec in (cfg.get("extra") or {}).items():
        if args.only and name not in args.only:
            continue
        out = paths.WEATHER_CSV_DIR / f"{name}.csv"
        if out.exists() and not args.force:
            print(f"  {name}: present")
            continue
        try:
            weather.fetch_open_meteo(name, spec["lat"], spec["lon"], year, paths.WEATHER_CSV_DIR)
        except Exception as exc:
            print(f"  {name}: FAILED ({exc!r}); its strips will be skipped")
            continue
        print(f"  {name}: wrote {out} (convection {weather.convection_from_csv(out)} W/m2K)")
        time.sleep(1.0)  # be polite to the free API


if __name__ == "__main__":
    main()
