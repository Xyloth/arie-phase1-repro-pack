#!/usr/bin/env python3
"""Process mechanistic hERG dataset into a canonical feature table."""

from __future__ import annotations

import argparse

from arie.mechanistic import process_mechanistic_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process mechanistic hERG dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if the processed file is present.",
    )
    parser.add_argument(
        "--include-chembl",
        action="store_true",
        help="Include ChEMBL gap-fill rows if available.",
    )
    parser.add_argument(
        "--chembl-gapfill-path",
        type=str,
        default=None,
        help="Path to ChEMBL gap-fill CSV (default: results/chembl_gapfill_herg_multilab_2025.csv).",
    )
    parser.add_argument(
        "--enable-identity-alias",
        action="store_true",
        help="Enable identity-changing aliases (default: off).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path, skipped = process_mechanistic_data(
        force=args.force,
        include_chembl_gapfill=args.include_chembl,
        chembl_gapfill_path=args.chembl_gapfill_path,
        enable_identity_alias=args.enable_identity_alias,
    )
    print(f"Identity alias enabled: {args.enable_identity_alias}")
    if skipped:
        print(f"Found existing processed file at {path}. Skipping processing.")
    else:
        print(f"Wrote processed mechanistic data to {path}.")


if __name__ == "__main__":
    main()
