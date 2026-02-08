#!/usr/bin/env python3
"""Fetch ChEMBL multichannel activities for CiPA-28 parent drugs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arie.chembl_multichannel import (
    fetch_multichannel_chembl,
    summarize_coverage,
    build_target_map,
    prefilter_diagnostics,
    build_relaxed_dataset,
    CHANNEL_ORDER,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "processed" / "mechanistic_multichannel_chembl.csv"
RELAXED_PATH = ROOT / "data" / "processed" / "mechanistic_multichannel_chembl_relaxed.csv"
SUMMARY_PATH = ROOT / "results" / "mechanistic_multichannel_chembl_join_summary.json"
REPORT_PATH = ROOT / "reports" / "mechanistic_coverage_multichannel_chembl.md"
TARGET_MAP_PATH = ROOT / "results" / "mechanistic_multichannel_chembl_target_map.json"
PREFILTER_PATH = ROOT / "results" / "mechanistic_multichannel_chembl_prefilter_diagnostics.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch ChEMBL multichannel mechanistic data.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fetch even if cache exists.",
    )
    parser.add_argument(
        "--enable-identity-alias",
        action="store_true",
        help="Enable identity-changing aliases (default: off).",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run target map + prefilter diagnostics and optional relaxed dataset.",
    )
    return parser.parse_args()


def _write_report(summary: dict, relaxed_summary: dict | None = None) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ChEMBL Multichannel Coverage (CiPA-28)",
        "",
        f"Identity alias enabled: {summary.get('identity_alias_enabled')}",
        f"CiPA parent count: {summary.get('cipa_parent_count')}",
        "",
        "## Coverage by channel (alias OFF)",
        "",
    ]
    for entry in summary.get("coverage_by_channel", []):
        missing = entry.get("missing_parents") or []
        lines.append(
            f"- {entry['target_channel']}: {entry['covered_parents']} / {summary.get('cipa_parent_count')} "
            f"(activities: {entry['activity_count']}; missing: {', '.join(missing) if missing else 'none'})"
        )
    lines.extend(["", "## Coverage by channel (alias ON)", ""])
    for entry in summary.get("coverage_by_channel_identity", []):
        missing = entry.get("missing_parents") or []
        lines.append(
            f"- {entry['target_channel']}: {entry['covered_parents']} / {summary.get('cipa_parent_count')} "
            f"(activities: {entry['activity_count']}; missing: {', '.join(missing) if missing else 'none'})"
        )
    lines.extend(
        [
            "",
            f"Coverage with ≥1 channel (alias OFF): {summary.get('coverage_any_channel')}",
            f"Coverage with all channels (alias OFF): {summary.get('coverage_all_channels')}",
            f"Coverage with ≥1 channel (alias ON): {summary.get('coverage_any_channel_identity')}",
            f"Coverage with all channels (alias ON): {summary.get('coverage_all_channels_identity')}",
            "",
        ]
    )

    if relaxed_summary is not None:
        lines.extend(
            [
                "## Relaxed coverage (warning: relaxed rows are bounded evidence)",
                "",
                f"Coverage with ≥1 channel (relaxed, alias OFF): {relaxed_summary.get('coverage_any_channel')}",
                f"Coverage with all channels (relaxed, alias OFF): {relaxed_summary.get('coverage_all_channels')}",
                f"Coverage with ≥1 channel (relaxed, alias ON): {relaxed_summary.get('coverage_any_channel_identity')}",
                f"Coverage with all channels (relaxed, alias ON): {relaxed_summary.get('coverage_all_channels_identity')}",
                "",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    df, meta = fetch_multichannel_chembl(
        enable_identity_alias=args.enable_identity_alias,
        force=args.force,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    summary = summarize_coverage(df, enable_identity_alias=args.enable_identity_alias)
    summary["output_path"] = str(OUTPUT_PATH)
    summary["channels"] = CHANNEL_ORDER
    summary["identity_alias_enabled"] = args.enable_identity_alias
    summary_payload = {
        "strict": summary,
    }

    relaxed_summary = None
    if args.diagnose:
        target_map = build_target_map()
        TARGET_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        TARGET_MAP_PATH.write_text(json.dumps(target_map, indent=2) + "\n", encoding="utf-8")

        diagnostics = prefilter_diagnostics(
            enable_identity_alias=args.enable_identity_alias,
            force=args.force,
        )
        PREFILTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREFILTER_PATH.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")

        relaxed_df = build_relaxed_dataset(
            enable_identity_alias=args.enable_identity_alias,
            force=args.force,
        )
        if not relaxed_df.empty and len(relaxed_df) > len(df):
            RELAXED_PATH.parent.mkdir(parents=True, exist_ok=True)
            relaxed_df.to_csv(RELAXED_PATH, index=False)
            relaxed_summary = summarize_coverage(relaxed_df, enable_identity_alias=args.enable_identity_alias)
            relaxed_summary["output_path"] = str(RELAXED_PATH)
            relaxed_summary["channels"] = CHANNEL_ORDER
            relaxed_summary["identity_alias_enabled"] = args.enable_identity_alias
            relaxed_summary["warning"] = (
                "Relaxed dataset includes non-equality relations and/or Ki values; "
                "treat as bounded evidence."
            )
            summary_payload["relaxed"] = relaxed_summary

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    _write_report(summary, relaxed_summary=relaxed_summary)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Identity alias enabled: {args.enable_identity_alias}")
    if args.diagnose:
        print(f"Wrote {TARGET_MAP_PATH}")
        print(f"Wrote {PREFILTER_PATH}")
        if relaxed_summary is not None:
            print(f"Wrote {RELAXED_PATH}")

    print("\nCoverage by channel (alias OFF)")
    for entry in summary.get("coverage_by_channel", []):
        print(
            f"- {entry['target_channel']}: {entry['covered_parents']} / {summary.get('cipa_parent_count')} "
            f"(activities {entry['activity_count']})"
        )

    print("\nCoverage by channel (alias ON)")
    for entry in summary.get("coverage_by_channel_identity", []):
        print(
            f"- {entry['target_channel']}: {entry['covered_parents']} / {summary.get('cipa_parent_count')} "
            f"(activities {entry['activity_count']})"
        )


if __name__ == "__main__":
    main()
