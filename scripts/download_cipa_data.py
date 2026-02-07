#!/usr/bin/env python3
"""Download and cache the CiPA Myocyte Validation Study dataset (Blinova et al., 2018)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

DATASET_NAME = "cipa_blinova_2018"
URL = "https://cipaproject.org/wp-content/uploads/2018/09/Blinova_etal_2018_data.xlsx"
FILENAME = "Blinova_etal_2018_data.xlsx"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / DATASET_NAME
CACHE_DIR = ROOT / "data" / "cache"
TARGET = RAW_DIR / FILENAME
CACHE_META = CACHE_DIR / f"{DATASET_NAME}.json"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_cache_metadata(size_bytes: int, sha256: str, headers: dict[str, str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": DATASET_NAME,
        "url": URL,
        "filename": FILENAME,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "size_bytes": size_bytes,
        "sha256": sha256,
        "etag": headers.get("ETag"),
        "last_modified": headers.get("Last-Modified"),
    }
    CACHE_META.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size > 0:
        print(f"Found existing file at {TARGET}. Skipping download.")
        if not CACHE_META.exists():
            sha256 = sha256_file(TARGET)
            write_cache_metadata(TARGET.stat().st_size, sha256, headers={})
            print(f"Wrote cache metadata to {CACHE_META}.")
        return

    print(f"Downloading dataset to {TARGET} ...")
    response = requests.get(URL, stream=True, timeout=60)
    response.raise_for_status()

    tmp_path = TARGET.with_suffix(TARGET.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    tmp_path.replace(TARGET)

    size_bytes = TARGET.stat().st_size
    sha256 = sha256_file(TARGET)
    write_cache_metadata(size_bytes, sha256, headers=response.headers)
    print(f"Saved {size_bytes} bytes.")
    print(f"Wrote cache metadata to {CACHE_META}.")


if __name__ == "__main__":
    download()
