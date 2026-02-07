"""Dataset download and processing utilities for CiPA Blinova 2018."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests

DATASET_ID = "cipa_blinova_2018"
URL = "https://cipaproject.org/wp-content/uploads/2018/09/Blinova_etal_2018_data.xlsx"
FILENAME = "Blinova_etal_2018_data.xlsx"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / DATASET_ID
CACHE_DIR = ROOT / "data" / "cache"
TARGET = RAW_DIR / FILENAME
CACHE_META = CACHE_DIR / f"{DATASET_ID}.json"

PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / f"{DATASET_ID}.csv"
REPORTS_DIR = ROOT / "reports"
DATA_DICT_PATH = REPORTS_DIR / f"data_dictionary_{DATASET_ID}.md"

COLUMN_MAP = {
    "Drug_Name": "drug_name",
    "Cell_type": "cell_type",
    "risk": "risk_class",
    "Platform": "platform",
    "Type_of_EADs": "ead_type",
    "conc": "concentration_level",
    "EAD": "ead",
    "ddFPDc": "dd_fpdc",
    "site": "site",
}


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_cache_metadata(size_bytes: int, sha256: str, headers: dict[str, str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": DATASET_ID,
        "url": URL,
        "filename": FILENAME,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "size_bytes": size_bytes,
        "sha256": sha256,
        "etag": headers.get("ETag"),
        "last_modified": headers.get("Last-Modified"),
    }
    CACHE_META.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def download_cipa_blinova(force: bool = False) -> Tuple[Path, bool]:
    """Download the CiPA Blinova 2018 Excel file.

    Returns (path, skipped).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET.exists() and TARGET.stat().st_size > 0 and not force:
        if not CACHE_META.exists():
            sha256 = _sha256_file(TARGET)
            _write_cache_metadata(TARGET.stat().st_size, sha256, headers={})
        return TARGET, True

    response = requests.get(URL, stream=True, timeout=60)
    response.raise_for_status()

    tmp_path = TARGET.with_suffix(TARGET.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    tmp_path.replace(TARGET)

    size_bytes = TARGET.stat().st_size
    sha256 = _sha256_file(TARGET)
    _write_cache_metadata(size_bytes, sha256, headers=response.headers)

    return TARGET, False


def _standardize_strings(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def process_cipa_blinova(force: bool = False) -> Tuple[Path, bool]:
    """Process the CiPA Blinova 2018 dataset into a clean CSV.

    Returns (path, skipped).
    """
    if not TARGET.exists():
        raise FileNotFoundError(f"Raw file not found: {TARGET}")

    if PROCESSED_PATH.exists() and not force:
        return PROCESSED_PATH, True

    df = pd.read_excel(TARGET)
    df = df.rename(columns=COLUMN_MAP)

    ordered_cols = [
        "drug_name",
        "cell_type",
        "risk_class",
        "platform",
        "ead_type",
        "concentration_level",
        "ead",
        "dd_fpdc",
        "site",
    ]
    df = df[ordered_cols]

    df["drug_name"] = _standardize_strings(df["drug_name"])
    df["cell_type"] = _standardize_strings(df["cell_type"])
    df["risk_class"] = _standardize_strings(df["risk_class"]).str.upper()
    df["platform"] = _standardize_strings(df["platform"])
    df["ead_type"] = _standardize_strings(df["ead_type"]).str.upper()

    df["concentration_level"] = pd.to_numeric(df["concentration_level"], errors="coerce").astype("Int64")
    df["ead"] = pd.to_numeric(df["ead"], errors="coerce").astype("Int64")
    df["dd_fpdc"] = pd.to_numeric(df["dd_fpdc"], errors="coerce")
    df["site"] = pd.to_numeric(df["site"], errors="coerce").astype("Int64")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DICT_PATH.write_text(
        """
# Data Dictionary: CiPA Blinova 2018 (processed)

Unit of observation: one measurement row per drug × concentration level × platform × cell type × site (as recorded in the source file).

Columns:
- drug_name: Compound name (string).
- cell_type: Cell type label from source (e.g., CDI, AXG).
- risk_class: Risk class label from source (L/M/H). Missing values kept as-is.
- platform: Platform label from source (e.g., ACA, AXN, MCS, AMD, ECR, CLY).
- ead_type: EAD type code from source (e.g., A, B, C, D, Q). Missing values kept as-is.
- concentration_level: Concentration level as provided in the source file (integer 1–4). Units not specified in the file.
- ead: Early afterdepolarization flag from source (0/1).
- dd_fpdc: Numeric metric from source column ddFPDc. Units not specified in the file.
- site: Site identifier from source (integer).
""".lstrip(),
        encoding="utf-8",
    )

    return PROCESSED_PATH, False
