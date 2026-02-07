#!/usr/bin/env python3
"""Train and evaluate a baseline ML model on the processed CiPA dataset."""

from __future__ import annotations

import argparse

from arie.data import load_processed_dataset
from arie.baseline import train_evaluate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline CiPA classifier.")
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
        "--n-splits",
        type=int,
        default=5,
        help="Number of stratified group folds (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    df = load_processed_dataset()
    df = df.dropna(subset=["risk_class"]).copy()

    print("Dataset summary")
    print(f"- rows: {len(df)}")
    print(f"- unique drugs: {df['drug_name'].nunique()}")
    print(f"- unique risk_class: {df['risk_class'].nunique()}")
    print(f"- seeds: {seeds}")
    print(f"- split: StratifiedGroupKFold(n_splits={args.n_splits}, shuffle=True)")

    results = train_evaluate(
        seeds=seeds,
        n_splits=args.n_splits,
        plot_seed=seeds[0],
    )

    for run in results["runs"]:
        print(f"\nSeed {run['seed']} (fold {run['fold_index']})")
        print(
            f"- train drugs: {run['n_drugs_train']} | test drugs: {run['n_drugs_test']} | "
            f"overlap: {run['drug_overlap_count']}"
        )
        print(f"- train class counts: {run['train_class_counts']}")
        print(f"- test class counts: {run['test_class_counts']}")
        if run["missing_classes_in_test"]:
            print(f"WARNING: missing classes in test: {run['missing_classes_in_test']}")
        print(f"- balanced accuracy: {run['balanced_accuracy']:.4f}")

    print("\nSummary")
    print(
        f"- balanced accuracy mean: {results['balanced_accuracy_mean']:.4f} | "
        f"std: {results['balanced_accuracy_std']:.4f}"
    )
    print("Metrics saved to results/baseline_metrics.json")
    print("Plot saved to figures/baseline_confusion_matrix.png")


if __name__ == "__main__":
    main()
