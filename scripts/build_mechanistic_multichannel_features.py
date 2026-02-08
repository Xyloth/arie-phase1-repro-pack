#!/usr/bin/env python
"""Build canonical multi-channel mechanistic feature table for CiPA-28."""

from __future__ import annotations

import argparse
import json

from arie.mechanistic_multichannel_features import JOIN_SUMMARY_PATH, build_and_write


def _print_summary() -> None:
    if not JOIN_SUMMARY_PATH.exists():
        print("Join summary not found; run build first.")
        return
    summary = json.loads(JOIN_SUMMARY_PATH.read_text())
    coverage = summary.get("coverage", {})
    source_counts = summary.get("source_selection_counts", {})
    concordance = summary.get("concordance_overlap_counts", {})

    print("Coverage per channel:")
    for channel, info in coverage.items():
        print(f"- {channel}: {info.get('present')}/28")

    print("\nChosen-source distribution per channel:")
    for channel, counts in source_counts.items():
        counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"- {channel}: {counts_str}")

    print("\nConcordance overlap counts:")
    for key, val in concordance.items():
        print(f"- {key}: {val}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print coverage, source selection counts, and concordance overlap counts.",
    )
    args = parser.parse_args()

    df = build_and_write()
    print(f"Wrote features: {df.shape[0]} rows, {df.shape[1]} columns")
    if args.print_summary:
        _print_summary()


if __name__ == "__main__":
    main()
