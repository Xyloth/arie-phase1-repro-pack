"""Mechanistic feature table and join utilities for hERG multi-lab data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple

import pandas as pd

from arie.data import load_processed_dataset
from arie.mechanistic import PROCESSED_PATH as MECH_PROCESSED_PATH
from arie.names import normalize_compound

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "mechanistic_features_herg_multilab_2025.csv"
JOIN_SUMMARY_PATH = ROOT / "results" / "mechanistic_feature_join_summary.json"

NUMERIC_FEATURE_COLS = [
    "herg_ic50_uM_mean",
    "herg_ic50_uM_median",
    "herg_ic50_uM_std",
    "herg_ic50_uM_min",
    "herg_ic50_uM_max",
    "herg_ic50_uM_count",
    "herg_nh_mean",
    "herg_nh_median",
    "herg_nh_std",
    "herg_nh_min",
    "herg_nh_max",
    "herg_nh_count",
]

PROVENANCE_COUNT_COLS = [
    "ic50_count_osf",
    "ic50_count_chembl",
    "nh_count_osf",
    "nh_count_chembl",
]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_mechanistic_feature_table(
    processed_path: Path | None = None,
    output_path: Path | None = None,
    force: bool = False,
) -> Tuple[Path, bool]:
    """Create a deterministic per-parent feature table from mechanistic data."""
    processed_path = processed_path or MECH_PROCESSED_PATH
    output_path = output_path or FEATURES_PATH

    if output_path.exists() and not force:
        return output_path, True

    mech = pd.read_csv(processed_path)
    required = {"drug_name_parent", "mechanistic_source"}
    missing = required.difference(mech.columns)
    if missing:
        raise ValueError(f"Mechanistic table missing columns: {sorted(missing)}")

    missing_numeric = [col for col in NUMERIC_FEATURE_COLS if col not in mech.columns]
    if missing_numeric:
        raise ValueError(f"Mechanistic table missing numeric columns: {missing_numeric}")

    source_rank = {"OSF": 0, "ChEMBL": 1}
    mech = mech.copy()
    mech["_source_rank"] = mech["mechanistic_source"].map(source_rank).fillna(99)
    mech = mech.sort_values(["drug_name_parent", "_source_rank"])

    def _join_sources(series: pd.Series) -> str:
        unique = set(series)
        ordered = sorted(unique, key=lambda x: source_rank.get(x, 99))
        return "+".join(ordered)

    sources = (
        mech.groupby("drug_name_parent")["mechanistic_source"]
        .apply(_join_sources)
        .rename("mechanistic_sources_present")
    )

    preferred = mech.sort_values(["drug_name_parent", "_source_rank"]).drop_duplicates(
        "drug_name_parent", keep="first"
    )
    agg_numeric = preferred.set_index("drug_name_parent")[NUMERIC_FEATURE_COLS]

    osf_counts = (
        mech[mech["mechanistic_source"] == "OSF"]
        .set_index("drug_name_parent")[["herg_ic50_uM_count", "herg_nh_count"]]
        .rename(
            columns={
                "herg_ic50_uM_count": "ic50_count_osf",
                "herg_nh_count": "nh_count_osf",
            }
        )
    )
    chembl_counts = (
        mech[mech["mechanistic_source"] == "ChEMBL"]
        .set_index("drug_name_parent")[["herg_ic50_uM_count", "herg_nh_count"]]
        .rename(
            columns={
                "herg_ic50_uM_count": "ic50_count_chembl",
                "herg_nh_count": "nh_count_chembl",
            }
        )
    )
    provenance_counts = (
        pd.concat([osf_counts, chembl_counts], axis=1)
        .fillna(0)
        .astype(int)
    )

    features = pd.concat([sources, agg_numeric, provenance_counts], axis=1).reset_index()
    ordered_cols = [
        "drug_name_parent",
        "mechanistic_sources_present",
        *NUMERIC_FEATURE_COLS,
        *PROVENANCE_COUNT_COLS,
    ]
    features = features[ordered_cols].sort_values("drug_name_parent").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return output_path, False


def load_mechanistic_feature_table(path: Path | None = None) -> pd.DataFrame:
    path = path or FEATURES_PATH
    return pd.read_csv(path)


def summarize_mechanistic_feature_join(
    cipa_processed_path: Path | None = None,
    features_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    """Join CiPA processed data with mechanistic features and report match rates."""
    output_path = output_path or JOIN_SUMMARY_PATH

    cipa = load_processed_dataset(cipa_processed_path)
    cipa = cipa.copy()
    cipa["drug_name_parent"] = cipa["drug_name"].apply(
        lambda x: normalize_compound(x)["drug_name_parent"]
    )

    features = load_mechanistic_feature_table(features_path)

    merged = cipa.merge(features, on="drug_name_parent", how="left")

    cipa_parent = sorted(cipa["drug_name_parent"].unique())
    mech_parent = sorted(features["drug_name_parent"].unique())
    matched_parent = sorted(set(cipa_parent).intersection(set(mech_parent)))

    def _missing_rate(column: str) -> float:
        return float(merged[column].isna().mean()) if column in merged.columns else float("nan")

    summary = {
        "cipa_rows": int(len(cipa)),
        "cipa_unique_drugs_parent": int(len(cipa_parent)),
        "mechanistic_unique_drugs_parent": int(len(mech_parent)),
        "matched_drugs_parent": int(len(matched_parent)),
        "missing_rate_ic50_mean": _missing_rate("herg_ic50_uM_mean"),
        "missing_rate_nh_mean": _missing_rate("herg_nh_mean"),
        "features_path": str(features_path or FEATURES_PATH),
        "mechanistic_processed_path": str(MECH_PROCESSED_PATH),
        "mechanistic_features_sha256": _sha256_file(features_path or FEATURES_PATH)
        if (features_path or FEATURES_PATH).exists()
        else None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
