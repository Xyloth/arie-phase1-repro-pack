#!/usr/bin/env python3
"""Download and cache the CiPA Myocyte Validation Study dataset (Blinova et al., 2018)."""

from __future__ import annotations

import argparse

from arie.datasets import download_cipa_blinova


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the CiPA Blinova 2018 dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the file is present.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path, skipped = download_cipa_blinova(force=args.force)
    if skipped:
        print(f"Found existing file at {path}. Skipping download.")
    else:
        print(f"Downloaded dataset to {path}.")


if __name__ == "__main__":
    main()
