#!/usr/bin/env python3
"""Download multi-channel CiPA mechanistic data (FDA GitHub)."""

from __future__ import annotations

import argparse

from arie.mechanistic_multichannel import download_mechanistic_multichannel_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CiPA multi-channel mechanistic data.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_dir, index, skipped = download_mechanistic_multichannel_data(force=args.force)
    print(f"Raw directory: {raw_dir}")
    print(f"Repo: {index.get('repo_html')}")
    print(f"Default branch: {index.get('default_branch')}")
    print(f"Commit: {index.get('commit_sha')} ({index.get('commit_date')})")
    print(f"Retrieved at: {index.get('retrieved_at_utc')}")
    print(f"Files indexed: {len(index.get('files', []))}")
    print("Status: " + ("skipped (cached)" if skipped else "downloaded"))


if __name__ == "__main__":
    main()
