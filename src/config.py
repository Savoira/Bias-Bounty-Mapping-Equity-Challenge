"""Shared paths and constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("BB_DATA_DIR", ROOT / "data"))
FEAT_DIR = Path(os.environ.get("BB_FEAT_DIR", ROOT / "features"))
OUT_DIR = Path(os.environ.get("BB_OUT_DIR", ROOT / "submissions"))

REGIONS = ["maricopa-az", "northern-ca", "eastern-ok", "south-central-tx", "eastern-wa"]

# Organiser definitions (README of the data package).
NAMED_CLASSES = ("motorway", "trunk", "primary", "secondary")
FIRE_CATS = ("fire_department",)
EMS_CATS = ("ambulance_and_ems_services",)
SCHOOL_CATS = ("elementary_school", "middle_school", "high_school", "school",
               "private_school", "public_school")

# Buffer widths (metres, EPSG:5070) used to find highways that run along a tract
# boundary. TIGER tract boundaries are drawn on TIGER road edges, so a TIGER highway
# on a boundary intersects BOTH neighbouring tracts, while the (slightly offset)
# Overture copy of that highway lands in only one of them.
RING_WIDTHS = (15, 40)


def ref_path(region: str, layer: str) -> Path:
    return DATA_DIR / "reference" / region / f"{region}-{layer}.parquet"


def strata_path(region: str, table: str) -> Path:
    return DATA_DIR / "strata" / region / f"{region}-{table}.parquet"


def sample_path(region: str) -> Path:
    return DATA_DIR / "reference" / region / f"{region}-sample-submission.csv"
