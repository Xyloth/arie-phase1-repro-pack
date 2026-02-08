"""Multi-channel mechanistic data ingestion for CiPA ion-channel panel."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests

from arie.data import load_processed_dataset
from arie.names import normalize_compound

DATASET_ID = "mechanistic_multichannel_cipa"
REPO_OWNER = "FDA"
REPO_NAME = "CiPA"
REPO_API_BASE = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
REPO_HTML = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"
DATA_PATH = "Hill_fitting/data/mergedpatchclampdata-20160514.csv"
README_PATH = "Hill_fitting/data/README.md"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / DATASET_ID
CACHE_DIR = ROOT / "data" / "cache"
INDEX_PATH = CACHE_DIR / f"{DATASET_ID}_index.json"
PROCESSED_PATH = ROOT / "data" / "processed" / f"{DATASET_ID}.csv"
JOIN_SUMMARY_PATH = ROOT / "results" / "mechanistic_multichannel_join_summary.json"
JOIN_DEBUG_PATH = ROOT / "results" / "mechanistic_multichannel_join_debug.json"
DATA_DICT_PATH = ROOT / "reports" / "data_dictionary_mechanistic_multichannel_cipa.md"
COVERAGE_REPORT_PATH = ROOT / "reports" / "mechanistic_coverage_multichannel.md"
PROCESSED_META_PATH = CACHE_DIR / f"{DATASET_ID}_processed_meta.json"

# Bump when processing logic changes in a way that affects outputs.
PIPELINE_VERSION = "2026-02-08a"

CHANNEL_MAP = {
    "Calcium": "Cav1.2",
    "Peak sodium": "Nav1.5_peak",
    "Late sodium": "Nav1.5_late",
    "hERG": "hERG",
    "IKs": "IKs",
    "IK1": "IK1",
    "Kv4.3": "Kv4.3",
}


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_processed_meta() -> dict | None:
    if PROCESSED_META_PATH.exists():
        return json.loads(PROCESSED_META_PATH.read_text())
    return None


def _write_processed_meta(index: dict, processed_path: Path) -> None:
    meta = {
        "dataset_id": DATASET_ID,
        "pipeline_version": PIPELINE_VERSION,
        "source_commit": index.get("commit_sha"),
        "source_commit_date": index.get("commit_date"),
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "processed_path": str(processed_path),
        "processed_sha256": _sha256_file(processed_path),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _needs_reprocess(index: dict, meta: dict | None) -> bool:
    if meta is None:
        return True
    if meta.get("pipeline_version") != PIPELINE_VERSION:
        return True
    if meta.get("source_commit") != index.get("commit_sha"):
        return True
    return False


def _github_get_json(url: str) -> dict | list:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _fetch_repo_metadata() -> Dict[str, str]:
    repo = _github_get_json(REPO_API_BASE)
    default_branch = repo.get("default_branch", "master")
    return {
        "repo_html": repo.get("html_url", REPO_HTML),
        "default_branch": default_branch,
    }


def _fetch_latest_commit(path: str, ref: str) -> Dict[str, str]:
    url = f"{REPO_API_BASE}/commits?path={path}&per_page=1&sha={ref}"
    data = _github_get_json(url)
    if isinstance(data, list) and data:
        commit = data[0]
        return {
            "commit_sha": commit.get("sha"),
            "commit_date": commit.get("commit", {})
            .get("committer", {})
            .get("date"),
        }
    return {"commit_sha": None, "commit_date": None}


def _fetch_file_index(ref: str) -> List[dict]:
    url = f"{REPO_API_BASE}/contents/Hill_fitting/data?ref={ref}"
    items = _github_get_json(url)
    if not isinstance(items, list):
        raise RuntimeError("Unexpected GitHub API response when listing files.")
    return items


def discover_multichannel_index(force: bool = False) -> dict:
    if INDEX_PATH.exists() and not force:
        return json.loads(INDEX_PATH.read_text())

    metadata = _fetch_repo_metadata()
    ref = metadata["default_branch"]
    file_index = _fetch_file_index(ref)
    commit_info = _fetch_latest_commit(DATA_PATH, ref)

    index = {
        "dataset_id": DATASET_ID,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_html": metadata["repo_html"],
        "default_branch": ref,
        "commit_sha": commit_info.get("commit_sha"),
        "commit_date": commit_info.get("commit_date"),
        "files": [
            {
                "name": item.get("name"),
                "path": item.get("path"),
                "download_url": item.get("download_url"),
                "size": item.get("size"),
                "sha": item.get("sha"),
                "type": item.get("type"),
            }
            for item in file_index
            if item.get("type") == "file"
        ],
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def _download_file(url: str, dest: Path, force: bool = False) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return {"local_path": str(dest), "sha256": _sha256_file(dest), "skipped": True}

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with tmp.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    tmp.replace(dest)
    return {"local_path": str(dest), "sha256": _sha256_file(dest), "skipped": False}


def download_mechanistic_multichannel_data(force: bool = False) -> Tuple[Path, dict, bool]:
    """Download multi-channel CiPA data from FDA GitHub repo.

    Returns (raw_dir, index, skipped).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    index = discover_multichannel_index(force=force)

    files = {item["path"]: item for item in index.get("files", [])}
    required_paths = [DATA_PATH, README_PATH]
    missing = [p for p in required_paths if p not in files]
    if missing:
        raise RuntimeError(f"Required file(s) missing from index: {missing}")

    skipped_all = True
    for path in required_paths:
        file_info = files[path]
        url = file_info.get("download_url")
        if not url:
            raise RuntimeError(f"No download URL for {path}")
        dest = RAW_DIR / Path(path).name
        download_info = _download_file(url, dest, force=force)
        file_info["local_path"] = download_info["local_path"]
        file_info["sha256"] = download_info["sha256"]
        if not download_info["skipped"]:
            skipped_all = False

    INDEX_PATH.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return RAW_DIR, index, skipped_all


def _convert_concentration_to_um(value: float, unit: str | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    if unit is None or pd.isna(unit):
        return None
    unit = str(unit).strip().lower()
    if unit == "nm":
        return float(value) / 1000.0
    if unit in {"um", "µm"}:
        return float(value)
    if unit == "mm":
        return float(value) * 1000.0
    return None


def process_mechanistic_multichannel_data(
    force: bool = False,
    enable_identity_alias: bool = False,
) -> Tuple[Path, dict, dict, bool]:
    """Process multi-channel data into standardized long format."""
    index = discover_multichannel_index(force=False)
    if PROCESSED_PATH.exists() and not force:
        meta = _load_processed_meta()
        if not _needs_reprocess(index, meta):
            summary = _write_join_summary(index, enable_identity_alias=enable_identity_alias)
            debug = _write_join_debug(index)
            _write_data_dictionary()
            _write_coverage_report(summary)
            return PROCESSED_PATH, summary, debug, True

    raw_path = RAW_DIR / Path(DATA_PATH).name
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    df = df.rename(columns=str.strip)

    df["drug_name_raw"] = df["Drug"].astype("string").str.strip()
    norm = df["drug_name_raw"].apply(
        lambda x: normalize_compound(x, enable_identity_alias=enable_identity_alias)
    )
    df["drug_name_normalized"] = norm.apply(lambda x: x["drug_name_normalized"])
    df["drug_name_parent"] = norm.apply(lambda x: x["drug_name_parent"])

    df["channel_raw"] = df["channel"].astype("string").str.strip()
    df["channel_key"] = df["channel_raw"].map(CHANNEL_MAP).fillna(df["channel_raw"])

    df["concentration_raw"] = pd.to_numeric(df["Conc"], errors="coerce")
    df["concentration_unit"] = df["Units"].astype("string").str.strip()
    df["concentration_uM"] = df.apply(
        lambda row: _convert_concentration_to_um(row["concentration_raw"], row["concentration_unit"]),
        axis=1,
    )

    df["metric"] = "pct_inhibition"
    df["value"] = pd.to_numeric(df["block"], errors="coerce")
    df["units"] = "percent"
    df["mechanistic_source"] = "FDA_CiPA_GitHub"
    df["lab_or_provider"] = "Crumb2016"
    df["n_measurements"] = 1
    df["provenance_note"] = (
        "Hill_fitting/data/mergedpatchclampdata-20160514.csv "
        "(Crumb et al. 2016); columns Drug/Conc/Units/channel/block"
    )

    ordered_cols = [
        "drug_name_raw",
        "drug_name_parent",
        "drug_name_normalized",
        "channel_key",
        "metric",
        "value",
        "units",
        "concentration_raw",
        "concentration_unit",
        "concentration_uM",
        "mechanistic_source",
        "lab_or_provider",
        "n_measurements",
        "provenance_note",
    ]
    processed = df[ordered_cols].copy()

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_PATH, index=False)
    _write_processed_meta(index, PROCESSED_PATH)

    summary = _write_join_summary(index, enable_identity_alias=enable_identity_alias)
    debug = _write_join_debug(index)
    _write_data_dictionary()
    _write_coverage_report(summary)
    return PROCESSED_PATH, summary, debug, False


def _write_join_summary(index: dict, enable_identity_alias: bool) -> dict:
    cipa = load_processed_dataset()
    cipa_names = cipa["drug_name"].dropna().astype("string").unique().tolist()
    cipa_parent = sorted(
        {normalize_compound(name, enable_identity_alias=False)["drug_name_parent"] for name in cipa_names}
    )
    cipa_parent_identity = sorted(
        {normalize_compound(name, enable_identity_alias=True)["drug_name_parent"] for name in cipa_names}
    )

    processed = pd.read_csv(PROCESSED_PATH)
    channels = sorted(processed["channel_key"].dropna().unique().tolist())

    def _coverage(parent_set: List[str], alias_enabled: bool) -> List[dict]:
        coverage = []
        for channel in channels:
            subset = processed[processed["channel_key"] == channel]
            names = subset["drug_name_raw"].dropna().astype("string").unique().tolist()
            parents = sorted(
                {
                    normalize_compound(name, enable_identity_alias=alias_enabled)["drug_name_parent"]
                    for name in names
                }
            )
            covered = sorted(set(parent_set).intersection(set(parents)))
            missing = sorted(set(parent_set).difference(set(parents)))
            coverage.append(
                {
                    "channel_key": channel,
                    "covered_parents": len(covered),
                    "missing_parents": missing,
                    "coverage_rate": float(len(covered) / len(parent_set)) if parent_set else 0.0,
                }
            )
        return coverage

    coverage_default = _coverage(cipa_parent, alias_enabled=False)
    coverage_identity = _coverage(cipa_parent_identity, alias_enabled=True)

    summary = {
        "dataset_id": DATASET_ID,
        "pipeline_version": PIPELINE_VERSION,
        "identity_alias_enabled": bool(enable_identity_alias),
        "source_repo": index.get("repo_html"),
        "source_commit": index.get("commit_sha"),
        "source_commit_date": index.get("commit_date"),
        "rows": int(len(processed)),
        "unique_drugs": int(processed["drug_name_parent"].nunique()),
        "unique_channels": int(len(channels)),
        "channel_list": channels,
        "cipa_parent_count": int(len(cipa_parent)),
        "coverage_by_channel": coverage_default,
        "coverage_by_channel_identity": coverage_identity,
    }

    JOIN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOIN_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _normalization_transform_stats(names: List[str]) -> dict:
    total = len(names)
    casefold_changed = 0
    punctuation_removed = 0
    salt_stripped = 0

    for raw in names:
        raw_str = str(raw)
        if raw_str != raw_str.casefold():
            casefold_changed += 1
        if re.search(r"[^a-z0-9]", raw_str.casefold()):
            punctuation_removed += 1
        norm = normalize_compound(raw_str, enable_identity_alias=False)
        if norm["drug_name_parent"] != norm["drug_name_normalized"]:
            salt_stripped += 1

    transforms = [
        {"transform": "casefold", "count": casefold_changed},
        {"transform": "punctuation_or_space_removed", "count": punctuation_removed},
        {"transform": "salt_or_drop_stripped", "count": salt_stripped},
    ]
    for entry in transforms:
        entry["rate"] = float(entry["count"] / total) if total else 0.0

    return {
        "total": total,
        "transforms": sorted(transforms, key=lambda x: x["count"], reverse=True),
    }


def _closest_matches(target: str, candidates: List[str], top_k: int = 5) -> List[dict]:
    scored = []
    for cand in candidates:
        score = difflib.SequenceMatcher(None, target, cand).ratio()
        scored.append({"candidate": cand, "score": round(float(score), 4)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _write_join_debug(index: dict) -> dict:
    cipa = load_processed_dataset()
    cipa_names = sorted(cipa["drug_name"].dropna().astype("string").unique().tolist())

    processed = pd.read_csv(PROCESSED_PATH)
    source_names = sorted(processed["drug_name_raw"].dropna().astype("string").unique().tolist())

    cipa_parent_off = sorted(
        {normalize_compound(name, enable_identity_alias=False)["drug_name_parent"] for name in cipa_names}
    )
    cipa_parent_on = sorted(
        {normalize_compound(name, enable_identity_alias=True)["drug_name_parent"] for name in cipa_names}
    )
    source_parent_off = sorted(
        {normalize_compound(name, enable_identity_alias=False)["drug_name_parent"] for name in source_names}
    )
    source_parent_on = sorted(
        {normalize_compound(name, enable_identity_alias=True)["drug_name_parent"] for name in source_names}
    )

    matched_off = sorted(set(cipa_parent_off).intersection(set(source_parent_off)))
    unmatched_cipa_off = sorted(set(cipa_parent_off).difference(set(source_parent_off)))
    unmatched_source_off = sorted(set(source_parent_off).difference(set(cipa_parent_off)))

    matched_on = sorted(set(cipa_parent_on).intersection(set(source_parent_on)))
    unmatched_cipa_on = sorted(set(cipa_parent_on).difference(set(source_parent_on)))
    unmatched_source_on = sorted(set(source_parent_on).difference(set(cipa_parent_on)))

    suggestions = []
    for name in unmatched_cipa_off:
        suggestions.append(
            {
                "cipa_parent": name,
                "closest_source_parents": _closest_matches(name, source_parent_off),
            }
        )

    debug = {
        "dataset_id": DATASET_ID,
        "source_repo": index.get("repo_html"),
        "source_commit": index.get("commit_sha"),
        "source_commit_date": index.get("commit_date"),
        "alias_off": {
            "cipa_parent_set": cipa_parent_off,
            "source_parent_set": source_parent_off,
            "matched_parents": matched_off,
            "unmatched_cipa_parents": unmatched_cipa_off,
            "unmatched_source_parents": unmatched_source_off,
        },
        "alias_on": {
            "cipa_parent_set": cipa_parent_on,
            "source_parent_set": source_parent_on,
            "matched_parents": matched_on,
            "unmatched_cipa_parents": unmatched_cipa_on,
            "unmatched_source_parents": unmatched_source_on,
        },
        "normalization_transforms": {
            "cipa": _normalization_transform_stats(cipa_names),
            "source": _normalization_transform_stats(source_names),
        },
        "unmatched_cipa_suggestions": suggestions,
    }

    JOIN_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOIN_DEBUG_PATH.write_text(json.dumps(debug, indent=2) + "\n", encoding="utf-8")
    return debug


def _write_data_dictionary() -> None:
    DATA_DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_DICT_PATH.write_text(
        """
# Data Dictionary: Mechanistic Multi-channel CiPA (processed)

Source: FDA CiPA GitHub repository (Hill_fitting/data/mergedpatchclampdata-20160514.csv).
Primary reference: Crumb et al. 2016 (CiPA ion channel panel).

Normalization:
- Compound names are normalized with shared rules (casefold + punctuation removal + salt stripping).
- Identity-changing aliases are disabled by default (see `normalize_compound(..., enable_identity_alias=False)`).

Channel mapping (`channel_key`):
- Calcium -> Cav1.2
- Peak sodium -> Nav1.5_peak
- Late sodium -> Nav1.5_late
- hERG -> hERG
- IKs -> IKs
- IK1 -> IK1
- Kv4.3 -> Kv4.3

Measurements:
- Each row represents a single patch-clamp measurement.
- `metric` is `pct_inhibition` with `value` from the `block` column.
- `units` is recorded as `percent` (the source does not explicitly state units for `block`).

Concentration:
- `concentration_raw` and `concentration_unit` are taken from `Conc` and `Units`.
- `concentration_uM` is derived from `concentration_raw` using `Units` (nM -> µM, µM -> µM, mM -> µM).

Columns:
- drug_name_raw, drug_name_normalized, drug_name_parent
- channel_key
- metric, value, units
- concentration_raw, concentration_unit, concentration_uM
- mechanistic_source, lab_or_provider, n_measurements, provenance_note
""".lstrip(),
        encoding="utf-8",
    )


def _write_coverage_report(summary: dict) -> None:
    COVERAGE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Multi-channel Mechanistic Coverage (CiPA)",
        "",
        f"Source: {summary.get('source_repo')}",
        f"Commit: {summary.get('source_commit')}",
        "",
        f"Identity alias enabled: {summary.get('identity_alias_enabled')}",
        f"CiPA parent count: {summary.get('cipa_parent_count')}",
        "",
        "## Coverage by channel (identity alias OFF)",
        "",
    ]
    for entry in summary.get("coverage_by_channel", []):
        lines.append(
            f"- {entry['channel_key']}: {entry['covered_parents']} / {summary.get('cipa_parent_count')} "
            f"(missing: {', '.join(entry['missing_parents']) if entry['missing_parents'] else 'none'})"
        )
    lines.extend(["", "## Coverage by channel (identity alias ON)", ""])
    for entry in summary.get("coverage_by_channel_identity", []):
        lines.append(
            f"- {entry['channel_key']}: {entry['covered_parents']} / {summary.get('cipa_parent_count')} "
            f"(missing: {', '.join(entry['missing_parents']) if entry['missing_parents'] else 'none'})"
        )
    COVERAGE_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
