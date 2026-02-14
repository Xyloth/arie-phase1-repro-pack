#!/usr/bin/env python3
"""Process manual Crumb 2016 text extraction into long-format data."""

from __future__ import annotations

import argparse
from pathlib import Path

from arie.mechanistic_multichannel_crumb_text import (
    DEFAULT_INPUT,
    DEBUG_PATH,
    JOIN_SUMMARY_PATH,
    PROCESSED_PATH,
    REPORT_PATH,
    parse_text,
    write_reports,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process Crumb 2016 text extraction.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to crumb_extraction.txt (case-insensitive).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if processed file exists.",
    )
    parser.add_argument(
        "--enable-identity-alias",
        action="store_true",
        help="Enable identity alias mapping (default OFF).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if PROCESSED_PATH.exists() and not args.force:
        print(f"Found existing processed file at {PROCESSED_PATH}. Use --force to reprocess.")
        return

    input_path = Path(args.input)
    df, debug = parse_text(input_path, enable_identity_alias=args.enable_identity_alias)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    summary = write_reports(df, debug=debug, enable_identity_alias=args.enable_identity_alias)

    print(f"Wrote {PROCESSED_PATH}")
    print(f"Wrote {DEBUG_PATH}")
    print(f"Wrote {JOIN_SUMMARY_PATH}")
    print(f"Wrote {REPORT_PATH}")

    print("\nCounts:")
    print(f"- drugs parsed (raw): {len(debug.get('drugs_parsed', []))}")
    print(f"- unique parsed parents: {summary['alias_off']['parsed_parent_count']}")
    print(f"- CiPA parents: {summary['alias_off']['cipa_parent_count']}")
    print(f"- intersection parents: {summary['alias_off']['intersection_count']}")
    print(f"- missing CiPA parents: {summary['alias_off']['missing_cipa_count']}")
    print(f"- extra parsed parents: {summary['alias_off']['extra_parsed_count']}")
    print(f"- intersection_parents: {summary['intersection_parents']}")
    print(f"- top 10 extra_parents: {summary['extra_parents'][:10]}")
    print("- missing CiPA closest matches:")
    for missing in summary["missing_cipa_parents"]:
        suggestions = summary.get("missing_cipa_parent_suggestions", {}).get(missing, [])
        if suggestions:
            best = suggestions[0]
            print(
                f"  - {missing} -> {best['candidate']} (similarity={best['similarity']})"
            )
        else:
            print(f"  - {missing} -> no close match")

    print("\nRows by channel:")
    for channel in summary["channels"]:
        print(f"- {channel}: {summary['rows_by_channel'].get(channel, 0)}")

    sample_drug = "amiodarone"
    sample_df = df[df["drug_name_raw"].str.casefold() == sample_drug]
    if not sample_df.empty:
        print(f"\nSpot check for {sample_drug}:")
        for channel in summary["channels"]:
            subset = sample_df[sample_df["channel"] == channel]
            if subset.empty:
                continue
            concs = subset["concentration_raw"].tolist()
            reps = subset["block_pct_n"].tolist()
            print(f"- {channel}: concentrations={concs} reps_per_conc={reps}")

    print("\nCoverage by channel (alias OFF)")
    for entry in summary["coverage_alias_off"]:
        print(
            f"- {entry['channel']}: {entry['channel_coverage_count']} / {summary['alias_off']['cipa_parent_count']}"
        )

    print("\nCoverage by channel (alias ON)")
    for entry in summary["coverage_alias_on"]:
        print(
            f"- {entry['channel']}: {entry['channel_coverage_count']} / {summary['alias_on']['cipa_parent_count']}"
        )


if __name__ == "__main__":
    main()
