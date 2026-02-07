"""Mechanistic (CiPA-aligned) data utilities for hERG multi-lab OSF dataset."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests

from arie.data import load_processed_dataset
from arie.names import normalize_compound

MECHANISTIC_ID = "herg_multilab_2025"
OSF_NODE = "a6k5t"
OSF_API_BASE = "https://api.osf.io/v2"
OSF_STORAGE_ROOT = f"{OSF_API_BASE}/nodes/{OSF_NODE}/files/osfstorage/"
OSF_REQUEST_DELAY = 0.2

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / MECHANISTIC_ID
CACHE_DIR = ROOT / "data" / "cache"
CACHE_META = CACHE_DIR / f"{MECHANISTIC_ID}.json"

PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / f"mechanistic_{MECHANISTIC_ID}.csv"
REPORTS_DIR = ROOT / "reports"
DATA_DICT_PATH = REPORTS_DIR / f"data_dictionary_mechanistic_{MECHANISTIC_ID}.md"

JOIN_SUMMARY_PATH = ROOT / "results" / "mechanistic_join_summary.json"
CHEMBL_GAPFILL_PATH = ROOT / "results" / "chembl_gapfill_herg_multilab_2025.csv"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _osf_get_json(url: str, max_retries: int = 6, sleep_seconds: float = 2.0) -> dict:
    for attempt in range(max_retries):
        resp = requests.get(url, timeout=60)
        if resp.status_code == 429:
            time.sleep(sleep_seconds * (attempt + 1))
            continue
        resp.raise_for_status()
        payload = resp.json()
        time.sleep(OSF_REQUEST_DELAY)
        return payload
    raise RuntimeError(f"Rate limited on {url}")


def _find_folder_id(items: Iterable[dict], name: str) -> str | None:
    for item in items:
        attrs = item["attributes"]
        if attrs["kind"] == "folder" and attrs["name"] == name:
            return item["id"]
    return None


def _list_folder(folder_id: str, page_size: int = 100) -> List[dict]:
    url = f"{OSF_STORAGE_ROOT}{folder_id}/?page[size]={page_size}"
    data = _osf_get_json(url)
    items = data.get("data", [])
    next_link = data.get("links", {}).get("next")
    while next_link:
        if "page[size]" not in next_link:
            next_link = next_link + ("&" if "?" in next_link else "?") + f"page[size]={page_size}"
        more = _osf_get_json(next_link)
        items.extend(more.get("data", []))
        next_link = more.get("links", {}).get("next")
    return items


def _discover_osf_concentration_files() -> List[dict]:
    """Discover concentration-inhibition.xlsx files under /data/*/hERG/*/subtracted/tables."""
    root = _osf_get_json(OSF_STORAGE_ROOT)
    data_folder_id = _find_folder_id(root.get("data", []), "data")
    if data_folder_id is None:
        raise RuntimeError("Could not locate data folder in OSF storage.")

    files = []
    labs = _list_folder(data_folder_id)
    for lab_item in labs:
        if lab_item["attributes"]["kind"] != "folder":
            continue
        lab_name = lab_item["attributes"]["name"]
        lab_id = lab_item["id"]

        lab_contents = _list_folder(lab_id)
        for phase_name in ("hERG", "hERG_Phase_II"):
            phase_id = _find_folder_id(lab_contents, phase_name)
            if phase_id is None:
                continue

            drug_folders = _list_folder(phase_id)
            for drug_item in drug_folders:
                if drug_item["attributes"]["kind"] != "folder":
                    continue
                drug_name = drug_item["attributes"]["name"]
                drug_id = drug_item["id"]

                drug_contents = _list_folder(drug_id)
                sub_id = _find_folder_id(drug_contents, "subtracted")
                if sub_id is None:
                    continue

                sub_contents = _list_folder(sub_id)
                tables_id = _find_folder_id(sub_contents, "tables")
                if tables_id is None:
                    continue

                tables_contents = _list_folder(tables_id)
                for file_item in tables_contents:
                    attrs = file_item["attributes"]
                    if attrs["kind"] != "file":
                        continue
                    if attrs["name"] != "concentration-inhibition.xlsx":
                        continue
                    files.append(
                        {
                            "lab": lab_name,
                            "phase": phase_name,
                            "drug": drug_name,
                            "osf_path": attrs.get("materialized_path"),
                            "download_url": file_item["links"].get("download"),
                            "size": attrs.get("size"),
                        }
                    )
    return files


def _write_cache_metadata(files: List[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": MECHANISTIC_ID,
        "osf_node": OSF_NODE,
        "source": "OSF",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    CACHE_META.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def download_mechanistic_data(force: bool = False) -> Tuple[Path, bool]:
    """Download OSF concentration-inhibition tables.

    Returns (raw_dir, skipped).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_META.exists() and not force:
        meta = json.loads(CACHE_META.read_text())
        cached_files = meta.get("files", [])
        if cached_files:
            missing = []
            for entry in cached_files:
                local_path = Path(entry["local_path"])
                if not local_path.exists():
                    missing.append(local_path)
            if not missing:
                return RAW_DIR, True

    # Discover files (from OSF) and download.
    files = _discover_osf_concentration_files()
    downloaded = []

    for entry in files:
        lab = entry["lab"]
        phase = entry.get("phase", "hERG")
        drug = entry["drug"]
        download_url = entry["download_url"]
        if not download_url:
            continue
        local_dir = RAW_DIR / lab / phase / drug
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / "concentration-inhibition.xlsx"

        if local_path.exists() and local_path.stat().st_size > 0 and not force:
            sha256 = _sha256_file(local_path)
            downloaded.append(
                {
                    **entry,
                    "local_path": str(local_path),
                    "sha256": sha256,
                }
            )
            continue

        resp = requests.get(download_url, stream=True, timeout=60)
        resp.raise_for_status()
        tmp_path = local_path.with_suffix(".tmp")
        with tmp_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp_path.replace(local_path)

        sha256 = _sha256_file(local_path)
        downloaded.append(
            {
                **entry,
                "local_path": str(local_path),
                "sha256": sha256,
            }
        )

    _write_cache_metadata(downloaded)
    return RAW_DIR, False


def _convert_to_um(value: float | int | None, unit: str | None) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if unit is None:
        return None
    unit = str(unit).strip().lower()
    if unit in {"nm", "nM".lower()} or "nm" in unit:
        return float(value) / 1000.0
    if unit in {"um", "µm", "uM".lower()} or "um" in unit or "µm" in unit:
        return float(value)
    if unit in {"mm", "mM".lower()} or "mm" in unit:
        return float(value) * 1000.0
    return None


def process_mechanistic_data(
    force: bool = False,
    include_chembl_gapfill: bool = False,
    chembl_gapfill_path: Path | None = None,
    cipa_processed_path: Path | None = None,
) -> Tuple[Path, bool]:
    """Process OSF concentration-inhibition tables into a canonical mechanistic table."""
    if PROCESSED_PATH.exists() and not force:
        return PROCESSED_PATH, True

    files = sorted(RAW_DIR.glob("**/concentration-inhibition.xlsx"))
    if not files:
        raise FileNotFoundError(f"No concentration-inhibition.xlsx files under {RAW_DIR}")

    rows = []
    for path in files:
        rel_parts = path.relative_to(RAW_DIR).parts
        lab = rel_parts[0] if len(rel_parts) > 0 else "unknown"
        phase = rel_parts[1] if len(rel_parts) > 1 else "unknown"
        drug_folder = rel_parts[2] if len(rel_parts) > 2 else "unknown"
        if phase not in {"hERG", "hERG_Phase_II"}:
            continue

        df = pd.read_excel(path, sheet_name="ic50nh")
        if df.empty:
            continue

        # Expect a single row per drug in this sheet.
        for _, row in df.iterrows():
            drug_raw = str(row.get("DRUG", drug_folder)).strip()
            unit = row.get("CONCU")
            ic50 = row.get("IC50")
            nh = row.get("nh")
            ic50_uM = _convert_to_um(ic50, unit)

            rows.append(
                {
                    "lab": lab,
                    "phase": phase,
                    **normalize_compound(drug_raw),
                    "ic50": ic50,
                    "ic50_unit": unit,
                    "ic50_uM": ic50_uM,
                    "ic50_lcl": row.get("IC50_LCL"),
                    "ic50_ucl": row.get("IC50_UCL"),
                    "nh": nh,
                    "nh_lcl": row.get("nh_LCL"),
                    "nh_ucl": row.get("nh_UCL"),
                }
            )

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise RuntimeError("No mechanistic rows were parsed from OSF tables.")

    grouped = long_df.groupby("drug_name_parent", dropna=True)
    processed = grouped.agg(
        drug_name_raw=("drug_name_raw", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
        drug_name_normalized=("drug_name_normalized", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
        herg_ic50_uM_mean=("ic50_uM", "mean"),
        herg_ic50_uM_std=("ic50_uM", "std"),
        herg_ic50_uM_min=("ic50_uM", "min"),
        herg_ic50_uM_max=("ic50_uM", "max"),
        herg_nh_mean=("nh", "mean"),
        herg_nh_std=("nh", "std"),
        labs_n=("lab", "nunique"),
    ).reset_index()
    processed = processed.rename(columns={"drug_name_parent": "drug_name_parent"})
    processed["mechanistic_source"] = "OSF"
    ordered_cols = [
        "drug_name_raw",
        "drug_name_normalized",
        "drug_name_parent",
        "herg_ic50_uM_mean",
        "herg_ic50_uM_std",
        "herg_ic50_uM_min",
        "herg_ic50_uM_max",
        "herg_nh_mean",
        "herg_nh_std",
        "labs_n",
        "mechanistic_source",
    ]
    processed = processed[ordered_cols]

    if include_chembl_gapfill:
        gapfill_path = chembl_gapfill_path or CHEMBL_GAPFILL_PATH
        if gapfill_path.exists():
            processed = _apply_chembl_gapfill(
                processed,
                gapfill_path=gapfill_path,
                cipa_processed_path=cipa_processed_path,
            )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_PATH, index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DICT_PATH.write_text(
        f"""
# Data Dictionary: Mechanistic hERG Multi-lab 2025 (OSF, processed)

Source: OSF project {OSF_NODE} ("Alvarez Baron et al. HESI BAA manual patch clamp hERG data from five laboratories").
We aggregate per-drug IC50 and Hill coefficient values derived from the `ic50nh` sheet
within each `concentration-inhibition.xlsx` file (per lab, per drug).

Unit of observation: one row per drug (aggregated across labs).

Columns:
- drug_name_raw: Drug name as reported in the OSF tables (mode across labs).
- drug_name_normalized: Normalized drug name (casefolded, alphanumeric only) used for joins.
- herg_ic50_uM_mean: Mean IC50 in µM across labs.
- herg_ic50_uM_std: Standard deviation of IC50 in µM across labs.
- herg_ic50_uM_min: Minimum IC50 in µM across labs.
- herg_ic50_uM_max: Maximum IC50 in µM across labs.
- herg_nh_mean: Mean Hill coefficient across labs.
- herg_nh_std: Standard deviation of Hill coefficient across labs.
- labs_n: Number of labs contributing data for the drug.
""".lstrip(),
        encoding="utf-8",
    )

    return PROCESSED_PATH, False


def load_mechanistic_features(path: Path | None = None) -> pd.DataFrame:
    path = path or PROCESSED_PATH
    df = pd.read_csv(path)
    required = {"drug_name_raw", "drug_name_normalized", "drug_name_parent"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Mechanistic features missing columns: {sorted(missing)}")
    return df


def _apply_chembl_gapfill(
    osf_df: pd.DataFrame,
    gapfill_path: Path,
    cipa_processed_path: Path | None = None,
) -> pd.DataFrame:
    gapfill = pd.read_csv(gapfill_path)
    required = {
        "drug_name_parent",
        "drug_name_raw",
        "drug_name_normalized",
        "ic50_uM_mean",
        "ic50_uM_std",
        "ic50_uM_min",
        "ic50_uM_max",
        "n_activities",
    }
    missing = required.difference(gapfill.columns)
    if missing:
        raise ValueError(f"ChEMBL gapfill missing columns: {sorted(missing)}")

    cipa = load_processed_dataset(cipa_processed_path)
    cipa_parents = sorted({normalize_compound(name)["drug_name_parent"] for name in cipa["drug_name"]})
    osf_parents = set(osf_df["drug_name_parent"].tolist())
    missing_parents = sorted(set(cipa_parents).difference(osf_parents))

    gapfill = gapfill[gapfill["drug_name_parent"].isin(missing_parents)].copy()
    gapfill = gapfill[gapfill["n_activities"] > 0].copy()
    if gapfill.empty:
        return osf_df

    gapfill_features = pd.DataFrame(
        {
            "drug_name_raw": gapfill["drug_name_raw"],
            "drug_name_normalized": gapfill["drug_name_normalized"],
            "drug_name_parent": gapfill["drug_name_parent"],
            "herg_ic50_uM_mean": gapfill["ic50_uM_mean"],
            "herg_ic50_uM_std": gapfill["ic50_uM_std"],
            "herg_ic50_uM_min": gapfill["ic50_uM_min"],
            "herg_ic50_uM_max": gapfill["ic50_uM_max"],
            "herg_nh_mean": np.nan,
            "herg_nh_std": np.nan,
            "labs_n": gapfill["n_activities"],
            "mechanistic_source": "ChEMBL",
        }
    )

    combined = pd.concat([osf_df, gapfill_features], ignore_index=True)
    return combined


def summarize_mechanistic_join(
    processed_path: Path | None = None,
    mechanistic_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    """Join CiPA processed data with mechanistic features and report match rate."""
    output_path = output_path or JOIN_SUMMARY_PATH

    cipa = load_processed_dataset(processed_path)
    mech = load_mechanistic_features(mechanistic_path)

    cipa_unique = sorted(cipa["drug_name"].dropna().astype("string").unique())
    mech_unique = sorted(mech["drug_name_raw"].dropna().astype("string").unique())

    cipa_norm = sorted({normalize_compound(name)["drug_name_normalized"] for name in cipa_unique})
    mech_norm = sorted({normalize_compound(name)["drug_name_normalized"] for name in mech_unique})

    cipa_parent = sorted({normalize_compound(name)["drug_name_parent"] for name in cipa_unique})
    mech_parent = sorted({normalize_compound(name)["drug_name_parent"] for name in mech_unique})

    matched_parent = sorted(set(cipa_parent).intersection(set(mech_parent)))
    unmatched_cipa_parent = sorted(set(cipa_parent).difference(set(mech_parent)))
    unmatched_mech_parent = sorted(set(mech_parent).difference(set(cipa_parent)))

    summary = {
        "cipa_unique_drugs": int(len(cipa_unique)),
        "mechanistic_unique_drugs": int(len(mech_unique)),
        "cipa_unique_drugs_normalized": int(len(cipa_norm)),
        "mechanistic_unique_drugs_normalized": int(len(mech_norm)),
        "matched_drugs_normalized": int(len(set(cipa_norm).intersection(set(mech_norm)))),
        "match_rate_normalized": float(len(set(cipa_norm).intersection(set(mech_norm))) / len(cipa_norm))
        if cipa_norm
        else 0.0,
        "cipa_unique_drugs_parent": int(len(cipa_parent)),
        "mechanistic_unique_drugs_parent": int(len(mech_parent)),
        "matched_drugs_parent": int(len(matched_parent)),
        "match_rate_parent": float(len(matched_parent) / len(cipa_parent)) if cipa_parent else 0.0,
        "cipa_drug_list_raw": cipa_unique,
        "mechanistic_drug_list_raw": mech_unique,
        "cipa_drug_list_normalized": cipa_norm,
        "mechanistic_drug_list_normalized": mech_norm,
        "cipa_drug_list_parent": cipa_parent,
        "mechanistic_drug_list_parent": mech_parent,
        "unmatched_cipa_drugs_parent": unmatched_cipa_parent,
        "unmatched_mechanistic_drugs_parent": unmatched_mech_parent,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
