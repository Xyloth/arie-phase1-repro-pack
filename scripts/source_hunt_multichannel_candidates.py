#!/usr/bin/env python3
"""Compute coverage vs CiPA-28 for multi-channel candidate datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from arie.data import load_processed_dataset
from arie.names import normalize_compound

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "source_hunt_multichannel_candidates.json"
REPORT_PATH = ROOT / "reports" / "source_hunt_multichannel_candidates.md"

REQUIRED_CHANNELS = [
    "hERG",
    "Cav1.2",
    "Nav1.5_peak",
    "Nav1.5_late",
    "IKs",
    "IK1",
    "Kv4.3",
]

CHANNEL_MAP = {
    "ICaL": "Cav1.2",
    "INa": "Nav1.5_peak",
    "INaL": "Nav1.5_late",
    "Ito": "Kv4.3",
    "IKs": "IKs",
    "IK1": "IK1",
    "hERG": "hERG",
}

WIDE_CHANNEL_COLS = {
    "ICaL_IC50": "Cav1.2",
    "INa_IC50": "Nav1.5_peak",
    "INaL_IC50": "Nav1.5_late",
    "Ito_IC50": "Kv4.3",
    "IKs_IC50": "IKs",
    "IK1_IC50": "IK1",
    "hERG_IC50": "hERG",
}


def _cipa_parents(alias: bool = False) -> list[str]:
    cipa = load_processed_dataset()
    return sorted(
        {
            normalize_compound(name, enable_identity_alias=alias)["drug_name_parent"]
            for name in cipa["drug_name"].dropna()
        }
    )


def _coverage_long(df: pd.DataFrame, drug_col: str, channel_col: str, alias: bool) -> dict:
    df = df.copy()
    df["drug_parent"] = df[drug_col].apply(
        lambda x: normalize_compound(x, enable_identity_alias=alias)["drug_name_parent"]
    )
    df["channel_key"] = df[channel_col].map(CHANNEL_MAP).fillna(df[channel_col])

    cipa_parents = _cipa_parents(alias=alias)
    coverage = []
    for channel in REQUIRED_CHANNELS:
        subset = df[df["channel_key"] == channel]
        parents = sorted(subset["drug_parent"].dropna().unique().tolist())
        covered = sorted(set(cipa_parents).intersection(set(parents)))
        missing = sorted(set(cipa_parents).difference(set(parents)))
        coverage.append(
            {
                "target_channel": channel,
                "covered_parents": len(covered),
                "missing_parents": missing,
            }
        )
    return {"coverage_by_channel": coverage}


def _coverage_wide(df: pd.DataFrame, drug_col: str, alias: bool) -> dict:
    cipa_parents = _cipa_parents(alias=alias)
    coverage = []
    for col, channel in WIDE_CHANNEL_COLS.items():
        if col not in df.columns:
            continue
        subset = df[df[col].notna()]
        parents = sorted(
            {
                normalize_compound(x, enable_identity_alias=alias)["drug_name_parent"]
                for x in subset[drug_col]
            }
        )
        covered = sorted(set(cipa_parents).intersection(set(parents)))
        missing = sorted(set(cipa_parents).difference(set(parents)))
        coverage.append(
            {
                "target_channel": channel,
                "covered_parents": len(covered),
                "missing_parents": missing,
            }
        )

    coverage_map = {c["target_channel"]: c for c in coverage}
    ordered = []
    for channel in REQUIRED_CHANNELS:
        ordered.append(coverage_map.get(channel, {"target_channel": channel, "covered_parents": 0, "missing_parents": cipa_parents}))
    return {"coverage_by_channel": ordered}


def _compute_candidate(candidate: dict) -> dict:
    path = Path(candidate["path"])
    df = pd.read_csv(path)
    if candidate["format"] == "long":
        off = _coverage_long(df, candidate["drug_col"], candidate["channel_col"], alias=False)
        on = _coverage_long(df, candidate["drug_col"], candidate["channel_col"], alias=True)
    elif candidate["format"] == "wide":
        off = _coverage_wide(df, candidate["drug_col"], alias=False)
        on = _coverage_wide(df, candidate["drug_col"], alias=True)
    else:
        raise ValueError(f"Unknown format: {candidate['format']}")

    result = {
        "candidate_id": candidate["candidate_id"],
        "title": candidate.get("title"),
        "citation": candidate.get("citation"),
        "license": candidate.get("license"),
        "urls": candidate.get("urls"),
        "notes": candidate.get("notes"),
        "recommended_role": candidate.get("recommended_role"),
        "data_acquisition": candidate.get("data_acquisition"),
        "verdict": candidate.get("verdict"),
        "channels": REQUIRED_CHANNELS,
        "coverage_alias_off": off["coverage_by_channel"],
        "coverage_alias_on": on["coverage_by_channel"],
    }
    return result


def _write_report(candidates: list[dict]) -> None:
    lines = [
        "# Source Hunt: Multichannel Candidates (CiPA-28)",
        "",
        "Coverage counts are computed against the CiPA-28 parent set (alias OFF by default).",
        "No public dataset found that provides IKs/IK1/Kv4.3 potency for all 28 drugs; candidates below are the best available.",
        "",
    ]
    for cand in candidates:
        lines.extend(
            [
                f"## {cand['candidate_id']}",
                "",
                f"Title: {cand.get('title')}",
                f"Citation: {cand.get('citation')}",
                f"License/terms: {cand.get('license')}",
                f"URLs: {', '.join(cand.get('urls', []))}",
                f"Data acquisition: {cand.get('data_acquisition')}",
                f"Verdict: {cand.get('verdict')}",
                f"Recommended role: {cand.get('recommended_role')}",
                f"Notes: {cand.get('notes')}",
                "",
                "Coverage by channel (alias OFF):",
            ]
        )
        for entry in cand["coverage_alias_off"]:
            missing = entry["missing_parents"]
            lines.append(
                f"- {entry['target_channel']}: {entry['covered_parents']} / 28 "
                f"(missing: {', '.join(missing) if missing else 'none'})"
            )
        lines.append("")
        lines.append("Coverage by channel (alias ON):")
        for entry in cand["coverage_alias_on"]:
            missing = entry["missing_parents"]
            lines.append(
                f"- {entry['target_channel']}: {entry['covered_parents']} / 28 "
                f"(missing: {', '.join(missing) if missing else 'none'})"
            )
        lines.append("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute coverage for multichannel candidates.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(ROOT / "data" / "raw" / "source_hunt"),
        help="Directory containing candidate files.",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    candidates = [
        {
            "candidate_id": "fda_cipa_manual_training_2018",
            "title": "Manual patch-clamp training panel (CiPA Model-Validation-2018)",
            "citation": (
                "Crumb WJ Jr, Vicente J, Johannesen L, Strauss DG (2016) "
                "J Pharmacol Toxicol Methods 81:251-262. DOI:10.1016/j.vascn.2016.03.009."
            ),
            "license": "FDA/CiPA GitHub repository GPL-3.0 (LICENSE in repo)",
            "urls": [
                "https://github.com/FDA/CiPA/tree/Model-Validation-2018/Hill_Fitting/data",
                "https://raw.githubusercontent.com/FDA/CiPA/Model-Validation-2018/Hill_Fitting/data/manual_trainingdrug_block.csv",
            ],
            "path": str(data_dir / "manual_trainingdrug_block.csv"),
            "format": "long",
            "drug_col": "drug",
            "channel_col": "channel",
            "notes": "Percent block vs concentration across 7 channels; 31 drugs.",
            "data_acquisition": "Direct raw CSV download from GitHub (reproducible).",
            "verdict": "Good secondary gap-fill (limited overlap).",
            "recommended_role": "Secondary gap-fill for IKs/IK1/Kv4.3 (limited drug overlap).",
        },
        {
            "candidate_id": "fda_cipa_manual_validation_2018",
            "title": "Manual patch-clamp validation panel (CiPA Model-Validation-2018)",
            "citation": (
                "FDA/CiPA Model-Validation-2018 data (see repo). "
                "Underlying patch-clamp context cites Crumb et al. 2016."
            ),
            "license": "FDA/CiPA GitHub repository GPL-3.0 (LICENSE in repo)",
            "urls": [
                "https://github.com/FDA/CiPA/tree/Model-Validation-2018/Hill_Fitting/data",
                "https://raw.githubusercontent.com/FDA/CiPA/Model-Validation-2018/Hill_Fitting/data/manual_validationdrug_block.csv",
            ],
            "path": str(data_dir / "manual_validationdrug_block.csv"),
            "format": "long",
            "drug_col": "drug",
            "channel_col": "channel",
            "notes": "Percent block vs concentration for 4 channels only (no IKs/IK1/Kv4.3).",
            "data_acquisition": "Direct raw CSV download from GitHub (reproducible).",
            "verdict": "Reject for IKs/IK1/Kv4.3 coverage.",
            "recommended_role": "Not suitable as primary; only hERG/Na/Ca.",
        },
        {
            "candidate_id": "fda_cipa_hts_training_2018",
            "title": "HTS patch-clamp training panel (CiPA Model-Validation-2018)",
            "citation": (
                "FDA/CiPA Model-Validation-2018 data (see repo). "
                "Underlying context cites Crumb et al. 2016."
            ),
            "license": "FDA/CiPA GitHub repository GPL-3.0 (LICENSE in repo)",
            "urls": [
                "https://github.com/FDA/CiPA/tree/Model-Validation-2018/Hill_Fitting/data",
                "https://raw.githubusercontent.com/FDA/CiPA/Model-Validation-2018/Hill_Fitting/data/HTS_trainingdrug_block.csv",
            ],
            "path": str(data_dir / "HTS_trainingdrug_block.csv"),
            "format": "long",
            "drug_col": "drug",
            "channel_col": "channel",
            "notes": "High-throughput screening version of training panel; 7 channels, 12 drugs.",
            "data_acquisition": "Direct raw CSV download from GitHub (reproducible).",
            "verdict": "Reject as primary; possible secondary with HTS caveats.",
            "recommended_role": "Secondary gap-fill; limited overlap and HTS assay heterogeneity.",
        },
        {
            "candidate_id": "fda_cipa_li2017_ic50",
            "title": "Li2017 IC50 summary (CiPA Model-Validation-2018)",
            "citation": (
                "FDA/CiPA Model-Validation-2018 data (Li2017_IC50.csv). "
                "Paper citation not explicit in repo; use repo link as stable archive."
            ),
            "license": "FDA/CiPA GitHub repository GPL-3.0 (LICENSE in repo)",
            "urls": [
                "https://github.com/FDA/CiPA/tree/Model-Validation-2018/Hill_Fitting/data",
                "https://raw.githubusercontent.com/FDA/CiPA/Model-Validation-2018/Hill_Fitting/data/Li2017_IC50.csv",
            ],
            "path": str(data_dir / "Li2017_IC50.csv"),
            "format": "wide",
            "drug_col": "drug",
            "notes": "Aggregated IC50/Hill values for 7 channels; units not specified in file.",
            "data_acquisition": "Direct raw CSV download from GitHub (reproducible).",
            "verdict": "Reject as primary (12 drugs only).",
            "recommended_role": "Possible reference for 7-channel IC50s.",
        },
    ]

    results = [_compute_candidate(c) for c in candidates]
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _write_report(results)

    print(f"Wrote {RESULTS_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
