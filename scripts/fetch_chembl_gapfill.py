#!/usr/bin/env python3
"""Fetch ChEMBL hERG (KCNH2) activity data for missing CiPA compounds."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from arie.chembl import fetch_kcnh2_activities, get_release_info, resolve_compound, summarize_values
from arie.names import normalize_compound

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

OUTPUT_CSV = RESULTS_DIR / "chembl_gapfill_herg_multilab_2025.csv"
OUTPUT_JSON = RESULTS_DIR / "chembl_gapfill_herg_multilab_2025.json"

TARGET_PARENTS = [
    "dlsotalol",
    "loratadine",
    "nifedipine",
    "nitrendipine",
    "quinidine",
]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    release = get_release_info()

    rows = []
    resolutions = []
    for name in TARGET_PARENTS:
        resolved = resolve_compound(name)
        if resolved is None:
            resolutions.append(
                {
                    "query_name": name,
                    "resolved": False,
                    "reason": "No pref_name/synonym match in ChEMBL search results.",
                }
            )
            rows.append(
                {
                    "drug_name_raw": name,
                    "drug_name_normalized": normalize_compound(name)["drug_name_normalized"],
                    "drug_name_parent": normalize_compound(name)["drug_name_parent"],
                    "chembl_molecule_id": None,
                    "resolved_name": None,
                    "resolution_method": None,
                    "n_activities": 0,
                    "ic50_uM_mean": None,
                    "ic50_uM_std": None,
                    "ic50_uM_median": None,
                    "ic50_uM_iqr": None,
                    "ic50_uM_min": None,
                    "ic50_uM_max": None,
                    "chembl_release": release.get("chembl_db_version"),
                }
            )
            continue

        chembl_id = resolved["molecule_chembl_id"]
        values_um = fetch_kcnh2_activities(chembl_id)
        stats = summarize_values(values_um)

        norm = normalize_compound(resolved["resolved_name"] or name)
        rows.append(
            {
                "drug_name_raw": resolved["resolved_name"] or name,
                "drug_name_normalized": norm["drug_name_normalized"],
                "drug_name_parent": norm["drug_name_parent"],
                "chembl_molecule_id": chembl_id,
                "resolved_name": resolved["resolved_name"],
                "resolution_method": resolved["resolution_method"],
                "n_activities": stats["n"],
                "ic50_uM_mean": stats["mean"],
                "ic50_uM_std": stats["std"],
                "ic50_uM_median": stats["median"],
                "ic50_uM_iqr": stats["iqr"],
                "ic50_uM_min": stats["min"],
                "ic50_uM_max": stats["max"],
                "chembl_release": release.get("chembl_db_version"),
            }
        )
        resolutions.append(
            {
                "query_name": name,
                "resolved": True,
                "chembl_id": chembl_id,
                "resolved_name": resolved["resolved_name"],
                "resolution_method": resolved["resolution_method"],
                "activity_count": stats["n"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "chembl_release": release,
                "target_chembl_id": "CHEMBL240",
                "queries": TARGET_PARENTS,
                "resolutions": resolutions,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"ChEMBL release: {release.get('chembl_db_version')} ({release.get('chembl_release_date')})")


if __name__ == "__main__":
    main()
