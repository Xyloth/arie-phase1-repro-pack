#!/usr/bin/env python3
"""Process Crumb 2016 CiPA supplement into structured long-format data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from arie.mechanistic_multichannel_crumb2016 import (
    CACHE_META,
    PROCESSED_PATH,
    RAW_DIR,
    SUPPLEMENT_URL,
    DEBUG_PATH,
    parse_supplement,
    write_reports,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process Crumb 2016 CiPA supplement.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if processed file exists.",
    )
    return parser.parse_args()


def _load_metadata() -> dict:
    if CACHE_META.exists():
        return json.loads(CACHE_META.read_text())
    return {}


def main() -> None:
    args = _parse_args()
    if PROCESSED_PATH.exists() and not args.force:
        print(f"Found existing processed file at {PROCESSED_PATH}. Use --force to reprocess.")
        return

    pdf_path = RAW_DIR / "Crumb2016_supplement.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing supplement PDF: {pdf_path}")

    meta = _load_metadata()
    downloaded = {d.get("path"): d for d in meta.get("downloads", [])}
    entry = downloaded.get(str(pdf_path))
    retrieved_at = meta.get("retrieved_at_utc")
    source_sha256 = entry.get("sha256") if entry else None

    df, debug = parse_supplement(pdf_path, enable_identity_alias=False)
    df["retrieved_at_utc"] = retrieved_at
    df["source_sha256"] = source_sha256
    df["source_url"] = SUPPLEMENT_URL

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    summary = write_reports(
        df,
        metadata={"sha256": source_sha256, "retrieved_at_utc": retrieved_at},
        debug=debug,
    )

    print(f"Wrote {PROCESSED_PATH}")
    print(f"Wrote {DEBUG_PATH}")
    print(f"Channels: {summary['channels']}")

    print("\nCoverage by channel (alias OFF)")
    for entry in summary["coverage_alias_off"]:
        print(
            f"- {entry['channel']}: {entry['covered_parents']} / {summary['cipa_parent_count']}"
        )

    print("\nCoverage by channel (alias ON)")
    for entry in summary["coverage_alias_on"]:
        print(
            f"- {entry['channel']}: {entry['covered_parents']} / {summary['cipa_parent_count']}"
        )

    # Sanity examples
    quinidine_rows = df[(df["drug_name_parent"] == "quinidine") & (df["channel"].isin(["IKs", "Kv4.3"]))]
    if not quinidine_rows.empty:
        print("\nExample: quinidine (IKs, Kv4.3)")
        print(quinidine_rows[["drug_name_raw", "channel", "ic50_uM", "ic50_relation", "hill_n"]].to_string(index=False))

    # IK1 example from debug data
    ik1_debug = next((d for d in debug.get("entries", []) if d.get("channel") == "IK1"), None)
    if ik1_debug:
        print("\nExample: IK1 raw block means (from debug)")
        print(
            f"{ik1_debug['drug_name_raw']} IK1 means @ concs {ik1_debug['concentrations_uM']}: "
            f"{ik1_debug['mean_block_pct']}"
        )


if __name__ == "__main__":
    main()
