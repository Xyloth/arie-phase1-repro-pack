#!/usr/bin/env python3
"""Download and cache the mechanistic hERG multi-lab dataset."""

from __future__ import annotations

import argparse

from arie.mechanistic import download_mechanistic_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download mechanistic hERG dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the file is present.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path, skipped = download_mechanistic_data(force=args.force)
    if skipped:
        print(f"Found existing file at {path}. Skipping download.")
    else:
        print(f"Downloaded dataset to {path}.")


if __name__ == "__main__":
    main()
