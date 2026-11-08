"""Weather stations and 30-day weather strips.

The kit's ReplayWeatherController reads only ``Time`` (``%Y%m%d-%H%M``),
``TempF`` and ``Humidity`` from a CSV, and treats ``Time`` as if it were UTC.
The shipped CSVs actually hold *local* time (Phoenix peaks at 13-17h), so the
building's occupancy schedule lines up with local daylight. Fetched stations
are therefore written in local time too, to keep that alignment.

A strip is a small YAML naming a CSV and an episode start. Strips are the unit
the splits are built from.
"""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from hvacdata import paths

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Convection coefficient (W/m2/K) for the building's outer surface. The kit
# ships one per station without saying how it was derived; across the six
# shipped stations it tracks mean wind speed at roughly 5-7 x (m/s), so new
# stations use 6 x mean wind, clipped to the shipped range with some margin.
CONVECTION_PER_MS = 6.0
CONVECTION_BOUNDS = (8.0, 30.0)

EPISODE_DAYS = 30
STRIP_HOUR = "06:00"  # every shipped strip starts at 06:00


@dataclass(frozen=True)
class Station:
    name: str
    split: str
    csv_path: Path
    convection_coefficient: float
    shipped: bool


def load_station_config(path: Path = paths.STATIONS_FILE) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_open_meteo(name: str, lat: float, lon: float, year: int, out_dir: Path,
                     retries: int = 4) -> Path:
    """Download one year of hourly weather and write it in the kit's CSV format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.csv"
    query = urllib.parse.urlencode({
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,wind_speed_10m",
        "timezone": "auto",  # local time, matching the shipped CSVs
        "wind_speed_unit": "kmh",
    })
    url = f"{OPEN_METEO_URL}?{query}"
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                payload = json.load(resp)
            break
        except Exception:  # network hiccups or rate limiting
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2

    h = payload["hourly"]
    rows = []
    for t, temp_c, rh, dew, wind in zip(
        h["time"], h["temperature_2m"], h["relative_humidity_2m"],
        h["dew_point_2m"], h["wind_speed_10m"],
    ):
        if temp_c is None:
            continue
        stamp = datetime.fromisoformat(t).strftime("%Y%m%d-%H%M")
        rows.append({
            "Time": stamp,
            "StationName": name,
            "Location": f"{lat},{lon}",
            "TempC": round(temp_c, 2),
            "DewPointC": "" if dew is None else round(dew, 2),
            "WindspeedKmph": "" if wind is None else round(wind, 2),
            "Humidity": "" if rh is None else round(rh, 1),
            "TempF": round(temp_c * 9 / 5 + 32, 2),
        })
    # Humidity gaps would become NaN inside the simulator; fill forward.
    last_rh = 50.0
    for r in rows:
        if r["Humidity"] == "":
            r["Humidity"] = last_rh
        last_rh = r["Humidity"]

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["", *rows[0].keys()])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"": i, **r})
    return out


def convection_from_csv(csv_path: Path) -> float:
    with open(csv_path) as f:
        winds = [float(r["WindspeedKmph"]) for r in csv.DictReader(f)
                 if r.get("WindspeedKmph") not in (None, "")]
    if not winds:
        return 17.0
    mean_ms = sum(winds) / len(winds) / 3.6
    lo, hi = CONVECTION_BOUNDS
    return round(min(hi, max(lo, CONVECTION_PER_MS * mean_ms)), 1)


def shipped_convection(station: str) -> float:
    with open(paths.KIT_WEATHER / f"{station}_1.yaml") as f:
        return float(yaml.safe_load(f)["convection_coefficient"])


def list_stations(cfg: dict | None = None, require_csv: bool = True) -> list[Station]:
    cfg = cfg or load_station_config()
    out: list[Station] = []
    for name, spec in (cfg.get("shipped") or {}).items():
        out.append(Station(name, spec["split"], paths.KIT_WEATHER / "csv" / f"{name}.csv",
                           shipped_convection(name), True))
    for name, spec in (cfg.get("extra") or {}).items():
        csv_path = paths.WEATHER_CSV_DIR / f"{name}.csv"
        if not csv_path.exists():
            if require_csv:
                print(f"  skip {name}: {csv_path} missing (run scripts/fetch_weather.py)")
            continue
        out.append(Station(name, spec["split"], csv_path, convection_from_csv(csv_path), False))
    return out


def shipped_strip_names() -> set[str]:
    return {p.stem for p in paths.KIT_WEATHER.glob("*.yaml")}


def kit_training_strips() -> set[str]:
    with open(paths.KIT_TRAINING_SET) as f:
        return {s["weather"] for s in yaml.safe_load(f)["scenarios"]}


def strip_start_dates(year: int, every_days: int, offset_days: int = 0) -> list[datetime]:
    """Episode starts through the year, leaving room for a full 30-day episode."""
    first = datetime(year, 1, 1) + timedelta(days=offset_days)
    last = datetime(year, 12, 31) - timedelta(days=EPISODE_DAYS + 1)
    out = []
    d = first
    while d <= last:
        out.append(d)
        d += timedelta(days=every_days)
    return out


def write_strip(station: Station, start: datetime, out_dir: Path) -> dict:
    """Write a strip YAML for the kit's scenario generator; return its record."""
    name = f"{station.name}_d{start.strftime('%m%d')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "name": name,
        # The kit resolves a relative weather_csv against its own weather dir
        # (and then rewrites it relative to the kit root in config.gin), so
        # store it relative to that: portable across machines and checkouts.
        "weather_csv": os.path.relpath(station.csv_path.resolve(), paths.KIT_WEATHER),
        "convection_coefficient": station.convection_coefficient,
        "start_timestamp": f"{start.strftime('%Y-%m-%d')} {STRIP_HOUR}",
    }
    path = out_dir / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    return {"name": name, "station": station.name, "split": station.split,
            "start": spec["start_timestamp"], "path": str(path)}


def strip_stats(csv_path: Path, start: str) -> dict:
    """Outdoor-temperature summary over a strip's 30 days (for sanity checks)."""
    t0 = datetime.strptime(start, "%Y-%m-%d %H:%M")
    t1 = t0 + timedelta(days=EPISODE_DAYS)
    temps = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            t = datetime.strptime(r["Time"], "%Y%m%d-%H%M")
            if t0 <= t < t1:
                temps.append((float(r["TempF"]) - 32) * 5 / 9)
    if not temps:
        return {"n_hours": 0}
    temps.sort()
    n = len(temps)
    return {"n_hours": n, "min_c": round(temps[0], 1), "mean_c": round(sum(temps) / n, 1),
            "max_c": round(temps[-1], 1), "p10_c": round(temps[n // 10], 1),
            "p90_c": round(temps[(9 * n) // 10], 1)}
