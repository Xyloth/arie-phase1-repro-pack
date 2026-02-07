#!/usr/bin/env python3
"""Process the CiPA Blinova 2018 dataset into a clean, analysis-ready table."""

from __future__ import annotations

import argparse

from arie.datasets import process_cipa_blinova


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process the CiPA Blinova 2018 dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if the processed file exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path, skipped = process_cipa_blinova(force=args.force)
    if skipped:
        print(f"Found existing processed file at {path}. Skipping processing.")
    else:
        print(f"Wrote processed data to {path}.")


if __name__ == "__main__":
    main()
