#!/usr/bin/env python3
"""Download Crumb 2016 CiPA supplement PDF."""

from __future__ import annotations

import argparse

from arie.mechanistic_multichannel_crumb2016 import download_crumb2016


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Crumb 2016 CiPA supplement PDF.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached.",
    )
    parser.add_argument(
        "--include-proof",
        action="store_true",
        help="Also download the proof PDF.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    meta = download_crumb2016(force=args.force, include_proof=args.include_proof)
    print(f"Downloaded {len(meta['downloads'])} file(s).")
    for entry in meta["downloads"]:
        print(f"- {entry['path']} (sha256={entry['sha256']})")
    print(f"Retrieved at: {meta['retrieved_at_utc']}")


if __name__ == "__main__":
    main()
