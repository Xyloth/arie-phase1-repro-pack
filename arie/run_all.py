"""Run the full minimal pipeline: download -> process -> baseline."""

from __future__ import annotations

import argparse

from arie.baseline import train_evaluate
from arie.datasets import DATASET_ID, download_cipa_blinova, process_cipa_blinova


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CiPA baseline pipeline end-to-end.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of consecutive seeds to run (default: 3).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if the raw file exists.",
    )
    parser.add_argument(
        "--force-process",
        action="store_true",
        help="Force re-processing even if the processed file exists.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline training/evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    print(f"Pipeline: {DATASET_ID}")
    print(f"Seeds: {seeds}")

    raw_path, raw_skipped = download_cipa_blinova(force=args.force_download)
    if raw_skipped:
        print(f"Download: skipped (found {raw_path})")
    else:
        print(f"Download: completed -> {raw_path}")

    processed_path, processed_skipped = process_cipa_blinova(force=args.force_process)
    if processed_skipped:
        print(f"Process: skipped (found {processed_path})")
    else:
        print(f"Process: completed -> {processed_path}")

    if args.skip_baseline:
        print("Baseline: skipped")
    else:
        results = train_evaluate(seeds=seeds, plot_seed=seeds[0], processed_path=processed_path)
        print(
            "Baseline: balanced_accuracy mean="
            f"{results['balanced_accuracy_mean']:.4f} std={results['balanced_accuracy_std']:.4f}"
        )

    print("Artifacts: results/baseline_metrics.json")
    print("Artifacts: figures/baseline_confusion_matrix.png")


if __name__ == "__main__":
    main()
