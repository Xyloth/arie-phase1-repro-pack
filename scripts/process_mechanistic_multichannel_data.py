#!/usr/bin/env python3
"""Process CiPA multi-channel mechanistic data into standardized long format."""

from __future__ import annotations

import argparse

from arie.mechanistic_multichannel import process_mechanistic_multichannel_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process CiPA multi-channel mechanistic data.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if processed file exists.",
    )
    parser.add_argument(
        "--enable-identity-alias",
        action="store_true",
        help="Enable identity-changing aliases (default: off).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path, summary, debug, skipped = process_mechanistic_multichannel_data(
        force=args.force,
        enable_identity_alias=args.enable_identity_alias,
    )
    if skipped:
        print(f"Found existing processed file at {path}. Skipping processing.")
    else:
        print(f"Wrote processed file to {path}.")

    print("Source metadata")
    print(f"- repo: {summary.get('source_repo')}")
    print(f"- commit: {summary.get('source_commit')} ({summary.get('source_commit_date')})")
    print(f"- identity_alias_enabled: {summary.get('identity_alias_enabled')}")
    print(f"- rows: {summary.get('rows')}")
    print(f"- unique_drugs: {summary.get('unique_drugs')}")
    print(f"- unique_channels: {summary.get('unique_channels')}")
    print(f"- channel_list: {summary.get('channel_list')}")

    print("\nCoverage by channel (alias OFF)")
    for entry in summary.get("coverage_by_channel", []):
        print(
            f"- {entry['channel_key']}: {entry['covered_parents']} / {summary.get('cipa_parent_count')} "
            f"(missing {len(entry['missing_parents'])})"
        )

    print("\nCoverage by channel (alias ON)")
    for entry in summary.get("coverage_by_channel_identity", []):
        print(
            f"- {entry['channel_key']}: {entry['covered_parents']} / {summary.get('cipa_parent_count')} "
            f"(missing {len(entry['missing_parents'])})"
        )

    print("\nAlias OFF sets")
    print(f"- cipa_parent_set: {debug['alias_off']['cipa_parent_set']}")
    print(f"- source_parent_set: {debug['alias_off']['source_parent_set']}")
    print(f"- matched_parents: {debug['alias_off']['matched_parents']}")
    print(f"- unmatched_cipa_parents: {debug['alias_off']['unmatched_cipa_parents']}")
    print(f"- unmatched_source_parents: {debug['alias_off']['unmatched_source_parents']}")

    print("\nAlias ON sets")
    print(f"- cipa_parent_set: {debug['alias_on']['cipa_parent_set']}")
    print(f"- source_parent_set: {debug['alias_on']['source_parent_set']}")
    print(f"- matched_parents: {debug['alias_on']['matched_parents']}")
    print(f"- unmatched_cipa_parents: {debug['alias_on']['unmatched_cipa_parents']}")
    print(f"- unmatched_source_parents: {debug['alias_on']['unmatched_source_parents']}")

    print("\nNormalization transform stats")
    print(f"- cipa: {debug['normalization_transforms']['cipa']}")
    print(f"- source: {debug['normalization_transforms']['source']}")

    if debug.get("unmatched_cipa_suggestions"):
        print("\nUnmatched CiPA suggestions (alias OFF)")
        for entry in debug["unmatched_cipa_suggestions"]:
            print(f"- {entry['cipa_parent']}: {entry['closest_source_parents']}")


if __name__ == "__main__":
    main()
