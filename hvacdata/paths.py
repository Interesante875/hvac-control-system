"""Repository layout and the working-directory rule for the starter kit."""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = Path(os.environ.get("SB_KIT_ROOT", REPO_ROOT / "external" / "starter-kit")).resolve()
DATA_ROOT = Path(os.environ.get("SB_DATA_ROOT", REPO_ROOT / "data")).resolve()

# Kit-owned inputs.
KIT_BUILDINGS = KIT_ROOT / "smart_buildings" / "buildings"
KIT_WEATHER = KIT_ROOT / "smart_buildings" / "weather"
KIT_SCENARIOS = KIT_ROOT / "smart_buildings" / "training_scenarios"
KIT_TRAINING_SET = KIT_ROOT / "smart_buildings" / "training_set.yaml"

# Our generated data. All of it is reproducible from the scripts, so it is
# gitignored except for the small manifests listed in .gitignore.
WEATHER_CSV_DIR = DATA_ROOT / "weather" / "csv"
WEATHER_STRIP_DIR = DATA_ROOT / "weather" / "strips"
BUILDINGS_DIR = DATA_ROOT / "buildings"
SCENARIOS_DIR = DATA_ROOT / "scenarios"
ROLLOUTS_DIR = DATA_ROOT / "rollouts"
SPLITS_FILE = DATA_ROOT / "splits.yaml"
STATIONS_FILE = REPO_ROOT / "configs" / "stations.yaml"

STEPS_PER_EPISODE = 8640  # 30 days at 5-minute steps; the only scoreable length.


def ensure_kit_importable() -> None:
    """Put the kit on sys.path and fail early with a useful message if absent."""
    if not (KIT_ROOT / "smart_buildings").is_dir():
        raise FileNotFoundError(
            f"Starter kit not found at {KIT_ROOT}. Run ./setup.sh "
            "(or `git submodule update --init`)."
        )
    os.environ.setdefault("WRAPT_DISABLE_EXTENSIONS", "true")
    if str(KIT_ROOT) not in sys.path:
        sys.path.insert(0, str(KIT_ROOT))


@contextlib.contextmanager
def in_kit() -> Iterator[None]:
    """Run a block with cwd at the kit root.

    Generated config.gin files store weather CSV, floor-plan and zone-map paths
    relative to the kit root, so loading or generating a scenario anywhere else
    fails with a missing-file error.
    """
    ensure_kit_importable()
    prev = Path.cwd()
    os.chdir(KIT_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


def kit_relative(path: Path) -> str:
    """Path as the kit's generator should see it: relative to the kit root."""
    return os.path.relpath(Path(path).resolve(), KIT_ROOT)
