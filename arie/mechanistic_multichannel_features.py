"""Canonical multi-channel mechanistic feature table for CiPA-28."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from arie.data import load_processed_dataset
from arie.names import normalize_compound

ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = ROOT / "data" / "processed" / "mechanistic_multichannel_features_cipa28.csv"
JOIN_SUMMARY_PATH = ROOT / "results" / "mechanistic_multichannel_feature_join_summary.json"
CONCORDANCE_JSON_PATH = ROOT / "results" / "mechanistic_multichannel_concordance.json"
CONCORDANCE_REPORT_PATH = ROOT / "reports" / "mechanistic_multichannel_concordance.md"
DATA_DICT_PATH = ROOT / "reports" / "data_dictionary_mechanistic_multichannel_features_cipa28.md"

CHEMBL_STRICT_PATH = ROOT / "data" / "processed" / "mechanistic_multichannel_chembl.csv"
CHEMBL_RELAXED_PATH = ROOT / "data" / "processed" / "mechanistic_multichannel_chembl_relaxed.csv"
CIPA_REPO_PATH = ROOT / "data" / "processed" / "mechanistic_multichannel_cipa.csv"
CRUMB_PATH = ROOT / "data" / "processed" / "mechanistic_multichannel_crumb2016.csv"

CHANNELS = [
    "hERG",
    "Nav1.5_peak",
    "Nav1.5_late",
    "Cav1.2",
    "IKs",
    "IK1",
    "Kv4.3",
]

SOURCE_PRIORITY = [
    "chembl_strict_ic50_eq",
    "chembl_relaxed_ic50_non_eq",
    "cipa_repo_ic50",
    "crumb2016_ic50_eq",
    "missing",
]


def _load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_cipa_parent_set(enable_identity_alias: bool = False) -> List[str]:
    cipa = load_processed_dataset()
    parents = sorted(
        {
            normalize_compound(name, enable_identity_alias=enable_identity_alias)["drug_name_parent"]
            for name in cipa["drug_name"].dropna()
        }
    )
    return parents


def _aggregate_chembl_strict(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    strict = df.copy()
    strict = strict[
        (strict["standard_type"].str.upper() == "IC50")
        & (strict["standard_relation"] == "=")
        & (~strict["value_uM"].isna())
    ]
    if strict.empty:
        return strict
    agg = (
        strict.groupby(["drug_name_parent", "target_channel"], as_index=False)["value_uM"]
        .agg(["median", "count"])
        .reset_index()
        .rename(columns={"median": "ic50_uM", "count": "n_records"})
    )
    return agg


def _aggregate_chembl_relaxed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    relaxed = df.copy()
    if "is_strict_row" in relaxed.columns:
        relaxed = relaxed[~relaxed["is_strict_row"].fillna(False)]
    relaxed = relaxed[
        (relaxed["standard_type"].str.upper() == "IC50")
        & (~relaxed["value_uM"].isna())
    ]
    if relaxed.empty:
        return relaxed
    agg = (
        relaxed.groupby(["drug_name_parent", "target_channel"], as_index=False)["value_uM"]
        .agg(["median", "count"])
        .reset_index()
        .rename(columns={"median": "ic50_uM", "count": "n_records"})
    )
    return agg


def _aggregate_crumb(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    crumb = df.copy()
    crumb = crumb[
        (crumb["ic50_relation"] == "=")
        & (~crumb["ic50_uM"].isna())
    ]
    if crumb.empty:
        return crumb
    agg = (
        crumb.groupby(["drug_name_parent", "channel"], as_index=False)["ic50_uM"]
        .agg(["median", "count"])
        .reset_index()
        .rename(columns={"median": "ic50_uM", "count": "n_records"})
        .rename(columns={"channel": "target_channel"})
    )
    return agg


def _aggregate_cipa_repo(df: pd.DataFrame) -> pd.DataFrame:
    """Return IC50 aggregates if present in CiPA repo data; empty if not."""
    if df.empty:
        return df
    if "metric" not in df.columns:
        return pd.DataFrame()
    ic50 = df[df["metric"].str.contains("ic50", case=False, na=False)].copy()
    if ic50.empty:
        return pd.DataFrame()
    ic50 = ic50[~ic50["value"].isna()]
    agg = (
        ic50.groupby(["drug_name_parent", "channel_key"], as_index=False)["value"]
        .agg(["median", "count"])
        .reset_index()
        .rename(columns={"median": "ic50_uM", "count": "n_records"})
        .rename(columns={"channel_key": "target_channel"})
    )
    return agg


def _build_source_maps() -> Tuple[dict, dict, dict, dict]:
    chembl_strict = _load_optional_csv(CHEMBL_STRICT_PATH)
    chembl_relaxed = _load_optional_csv(CHEMBL_RELAXED_PATH)
    cipa_repo = _load_optional_csv(CIPA_REPO_PATH)
    crumb = _load_optional_csv(CRUMB_PATH)

    strict_agg = _aggregate_chembl_strict(chembl_strict)
    relaxed_agg = _aggregate_chembl_relaxed(chembl_relaxed)
    cipa_agg = _aggregate_cipa_repo(cipa_repo)
    crumb_agg = _aggregate_crumb(crumb)

    return (
        _dict_from_agg(strict_agg),
        _dict_from_agg(relaxed_agg),
        _dict_from_agg(cipa_agg),
        _dict_from_agg(crumb_agg),
    )


def _dict_from_agg(df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, float]]:
    mapping: Dict[Tuple[str, str], Dict[str, float]] = {}
    if df.empty:
        return mapping
    for _, row in df.iterrows():
        key = (row["drug_name_parent"], row["target_channel"])
        mapping[key] = {"ic50_uM": float(row["ic50_uM"]), "n_records": int(row["n_records"])}
    return mapping


def _select_value(
    parent: str,
    channel: str,
    strict_map: dict,
    relaxed_map: dict,
    crumb_map: dict,
    cipa_map: dict,
) -> Tuple[float | None, str, int, bool]:
    key = (parent, channel)
    if key in strict_map:
        entry = strict_map[key]
        return entry["ic50_uM"], "chembl_strict", entry["n_records"], True
    if key in relaxed_map:
        entry = relaxed_map[key]
        return entry["ic50_uM"], "chembl_relaxed", entry["n_records"], False
    if key in cipa_map:
        entry = cipa_map[key]
        return entry["ic50_uM"], "cipa_repo", entry["n_records"], False
    if key in crumb_map:
        entry = crumb_map[key]
        return entry["ic50_uM"], "crumb2016", entry["n_records"], False
    return None, "missing", 0, False


def build_feature_table() -> pd.DataFrame:
    if not CHEMBL_STRICT_PATH.exists():
        raise FileNotFoundError(f"Missing strict ChEMBL file: {CHEMBL_STRICT_PATH}")
    if not CIPA_REPO_PATH.exists():
        raise FileNotFoundError(f"Missing CiPA repo file: {CIPA_REPO_PATH}")
    if not CRUMB_PATH.exists():
        raise FileNotFoundError(f"Missing Crumb2016 file: {CRUMB_PATH}")

    strict_map, relaxed_map, cipa_map, crumb_map = _build_source_maps()

    parents = load_cipa_parent_set(enable_identity_alias=False)
    rows: List[dict] = []
    for parent in parents:
        record: Dict[str, object] = {"drug_name_parent": parent}
        missing_channels: List[str] = []
        present_count = 0
        for channel in CHANNELS:
            value, source, n_records, is_strict = _select_value(
                parent, channel, strict_map, relaxed_map, crumb_map, cipa_map
            )
            record[f"ic50_uM_{channel}"] = value
            record[f"source_{channel}"] = source
            record[f"n_records_{channel}"] = n_records
            record[f"is_strict_{channel}"] = bool(is_strict)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                missing_channels.append(channel)
            else:
                present_count += 1
        record["n_channels_present"] = present_count
        record["n_channels_missing"] = len(CHANNELS) - present_count
        record["missing_channels_list"] = "|".join(missing_channels)
        rows.append(record)

    columns = ["drug_name_parent"]
    for channel in CHANNELS:
        columns.extend(
            [
                f"ic50_uM_{channel}",
                f"source_{channel}",
                f"n_records_{channel}",
                f"is_strict_{channel}",
            ]
        )
    columns.extend(["n_channels_present", "n_channels_missing", "missing_channels_list"])
    return pd.DataFrame(rows, columns=columns)


def _coverage_summary(df: pd.DataFrame, parent_col: str, channels: List[str]) -> dict:
    summary = {}
    for channel in channels:
        value_col = f"ic50_uM_{channel}"
        source_col = f"source_{channel}"
        present_mask = df[value_col].notna()
        present = int(present_mask.sum())
        missing = int((~present_mask).sum())
        missing_parents = sorted(df.loc[~present_mask, parent_col].tolist())
        source_counts = df.loc[present_mask, source_col].value_counts().to_dict()
        summary[channel] = {
            "present": present,
            "missing": missing,
            "missing_parents": missing_parents,
            "source_counts": source_counts,
        }
    return summary


def _alias_parent(name: str) -> str:
    return normalize_compound(name, enable_identity_alias=True)["drug_name_parent"]


def _coverage_alias_on(df: pd.DataFrame, channels: List[str]) -> dict:
    df_alias = df.copy()
    df_alias["drug_name_parent_alias"] = df_alias["drug_name_parent"].apply(_alias_parent)
    parents_alias = load_cipa_parent_set(enable_identity_alias=True)

    summary = {}
    for channel in channels:
        value_col = f"ic50_uM_{channel}"
        grouped = df_alias.groupby("drug_name_parent_alias")[value_col].apply(
            lambda s: s.notna().any()
        )
        present_alias = sorted(grouped[grouped].index.tolist())
        missing_alias = sorted(set(parents_alias) - set(present_alias))
        summary[channel] = {
            "present": len(present_alias),
            "missing": len(missing_alias),
            "missing_parents": missing_alias,
        }
    return {
        "identity_alias_enabled": True,
        "cipa_parent_count": len(parents_alias),
        "coverage": summary,
    }


def _expected_source(
    parent: str,
    channel: str,
    strict_map: dict,
    relaxed_map: dict,
    cipa_map: dict,
    crumb_map: dict,
) -> str:
    key = (parent, channel)
    if key in strict_map:
        return "chembl_strict"
    if key in relaxed_map:
        return "chembl_relaxed"
    if key in cipa_map:
        return "cipa_repo"
    if key in crumb_map:
        return "crumb2016"
    return "missing"


def _source_selection_counts(df: pd.DataFrame, channels: List[str]) -> dict:
    counts = {}
    for channel in channels:
        col = f"source_{channel}"
        counts[channel] = df[col].value_counts().to_dict()
    return counts


def write_join_summary(df: pd.DataFrame, concordance: dict | None = None) -> dict:
    parents = load_cipa_parent_set(enable_identity_alias=False)
    coverage = _coverage_summary(df, "drug_name_parent", CHANNELS)
    strict_map, relaxed_map, cipa_map, crumb_map = _build_source_maps()

    violations = []
    for _, row in df.iterrows():
        parent = row["drug_name_parent"]
        for channel in CHANNELS:
            selected = row[f"source_{channel}"]
            expected = _expected_source(parent, channel, strict_map, relaxed_map, cipa_map, crumb_map)
            if selected != expected:
                violations.append(
                    {
                        "drug_name_parent": parent,
                        "channel": channel,
                        "selected": selected,
                        "expected": expected,
                    }
                )

    summary = {
        "dataset_id": "mechanistic_multichannel_features_cipa28",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity_alias_enabled": False,
        "source_priority": SOURCE_PRIORITY,
        "channels": CHANNELS,
        "cipa_parent_count": len(parents),
        "coverage": coverage,
        "source_selection_counts": _source_selection_counts(df, CHANNELS),
        "priority_rule_assertions": {
            "violations_total": len(violations),
            "violations": violations,
        },
        "column_naming_policy": "channel names are used verbatim in column suffixes (e.g., ic50_uM_Nav1.5_peak).",
        "concordance_overlap_counts": {
            key: comp.get("n_pairs")
            for key, comp in (concordance or {}).get("comparisons", {}).items()
        },
        "alias_on": _coverage_alias_on(df, CHANNELS),
    }
    JOIN_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _concordance_pairs(
    label: str,
    left: Dict[Tuple[str, str], float],
    right: Dict[Tuple[str, str], float],
    left_label: str,
    right_label: str,
) -> dict:
    overlaps = []
    for key, left_val in left.items():
        if key not in right:
            continue
        right_val = right[key]
        if left_val is None or right_val is None:
            continue
        if left_val <= 0 or right_val <= 0:
            continue
        fold = max(left_val, right_val) / min(left_val, right_val)
        overlaps.append(
            {
                "drug_name_parent": key[0],
                "channel": key[1],
                left_label: left_val,
                right_label: right_val,
                "fold_diff": fold,
            }
        )

    if not overlaps:
        return {
            "label": label,
            "n_pairs": 0,
            "median_fold_diff": None,
            "mean_fold_diff": None,
            "max_fold_diff": None,
            "top_mismatches": [],
        }

    folds = [row["fold_diff"] for row in overlaps]
    overlaps_sorted = sorted(overlaps, key=lambda x: x["fold_diff"], reverse=True)
    return {
        "label": label,
        "n_pairs": len(overlaps),
        "median_fold_diff": float(np.median(folds)),
        "mean_fold_diff": float(np.mean(folds)),
        "max_fold_diff": float(np.max(folds)),
        "top_mismatches": overlaps_sorted[:10],
    }


def write_concordance(df_features: pd.DataFrame) -> dict:
    chembl_selected = {}
    for _, row in df_features.iterrows():
        parent = row["drug_name_parent"]
        for channel in CHANNELS:
            value = row[f"ic50_uM_{channel}"]
            source = row[f"source_{channel}"]
            if pd.isna(value):
                continue
            if source in {"chembl_strict", "chembl_relaxed"}:
                chembl_selected[(parent, channel)] = float(value)

    cipa_repo = _aggregate_cipa_repo(_load_optional_csv(CIPA_REPO_PATH))
    crumb = _aggregate_crumb(_load_optional_csv(CRUMB_PATH))

    cipa_map = _dict_from_agg(cipa_repo)
    crumb_map = _dict_from_agg(crumb)

    concordance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparisons": {
            "chembl_vs_cipa_repo": _concordance_pairs(
                "chembl_vs_cipa_repo",
                chembl_selected,
                {k: v["ic50_uM"] for k, v in cipa_map.items()},
                "chembl_ic50_uM",
                "cipa_ic50_uM",
            ),
            "chembl_vs_crumb2016": _concordance_pairs(
                "chembl_vs_crumb2016",
                chembl_selected,
                {k: v["ic50_uM"] for k, v in crumb_map.items()},
                "chembl_ic50_uM",
                "crumb_ic50_uM",
            ),
            "cipa_repo_vs_crumb2016": _concordance_pairs(
                "cipa_repo_vs_crumb2016",
                {k: v["ic50_uM"] for k, v in cipa_map.items()},
                {k: v["ic50_uM"] for k, v in crumb_map.items()},
                "cipa_ic50_uM",
                "crumb_ic50_uM",
            ),
        },
        "notes": [
            "CiPA repo dataset is percent inhibition only unless IC50 rows are present; "
            "if n_pairs=0, there were no IC50 rows to compare.",
            "Crumb2016 IC50s are derived from fitted block curves; only relation '=' rows used.",
            "ChEMBL values use IC50 median in µM; strict preferred else relaxed.",
        ],
    }
    CONCORDANCE_JSON_PATH.write_text(json.dumps(concordance, indent=2) + "\n", encoding="utf-8")
    _write_concordance_report(concordance)
    return concordance


def _write_concordance_report(concordance: dict) -> None:
    lines = []
    lines.append("# Multichannel IC50 Concordance")
    lines.append("")
    lines.append("This report compares IC50_uM values across sources where both provide data.")
    lines.append("")
    for key, comp in concordance.get("comparisons", {}).items():
        lines.append(f"## {key}")
        lines.append("")
        lines.append(f"Pairs compared: {comp.get('n_pairs')}")
        lines.append(f"Median fold-diff: {comp.get('median_fold_diff')}")
        lines.append(f"Mean fold-diff: {comp.get('mean_fold_diff')}")
        lines.append(f"Max fold-diff: {comp.get('max_fold_diff')}")
        lines.append("")
        mismatches = comp.get("top_mismatches", [])
        if mismatches:
            lines.append("Top mismatches (fold-diff):")
            lines.append("")
            lines.append("drug_name_parent | channel | fold_diff")
            lines.append("---|---|---")
            for row in mismatches[:10]:
                lines.append(
                    f"{row['drug_name_parent']} | {row['channel']} | {row['fold_diff']:.3f}"
                )
        else:
            lines.append("No overlapping IC50 pairs for this comparison.")
        lines.append("")
    CONCORDANCE_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_data_dictionary() -> None:
    lines = []
    lines.append("# Data Dictionary: mechanistic_multichannel_features_cipa28")
    lines.append("")
    lines.append("One row per CiPA parent drug; channels are stored as IC50 in µM.")
    lines.append("")
    lines.append("Source priority:")
    lines.append("1. ChEMBL strict IC50 '=' (µM-convertible).")
    lines.append("2. ChEMBL relaxed IC50 (non '=' relations) if strict absent.")
    lines.append("3. CiPA GitHub IC50 (only if any IC50 rows are present).")
    lines.append("4. Crumb2016 IC50 with relation '=' (fitted from block curves).")
    lines.append("5. Missing if no comparable IC50.")
    lines.append("")
    lines.append("CiPA GitHub panel provides percent inhibition only in the current file;")
    lines.append("no IC50 rows were observed, so it is not used for IC50 selection.")
    lines.append("")
    lines.append("Columns:")
    lines.append("- drug_name_parent: normalized parent compound name.")
    lines.append("- Column naming policy: channel names are used verbatim in column suffixes (e.g., ic50_uM_Nav1.5_peak).")
    for channel in CHANNELS:
        lines.append(f"- ic50_uM_{channel}: selected IC50 in µM for {channel}.")
        lines.append(f"- source_{channel}: data source used (chembl_strict, chembl_relaxed, crumb2016, missing).")
        lines.append(f"- n_records_{channel}: number of activity rows used in the aggregate.")
        lines.append(f"- is_strict_{channel}: True only if strict ChEMBL row used.")
    lines.append("- n_channels_present: number of channels with IC50 values present.")
    lines.append("- n_channels_missing: number of channels missing IC50 values.")
    lines.append("- missing_channels_list: pipe-delimited list of missing channels.")
    lines.append("")
    lines.append("Identity aliasing is disabled for feature construction. Coverage with aliasing")
    lines.append("enabled is reported in the join summary as a secondary diagnostic.")
    DATA_DICT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_and_write() -> pd.DataFrame:
    df = build_feature_table()
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)
    concordance = write_concordance(df)
    write_join_summary(df, concordance=concordance)
    write_data_dictionary()
    return df


__all__ = [
    "build_and_write",
    "build_feature_table",
    "write_join_summary",
    "write_concordance",
    "write_data_dictionary",
]
