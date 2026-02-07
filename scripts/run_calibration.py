#!/usr/bin/env python3
"""Run calibrated classifier evaluation on the CiPA dataset."""

from __future__ import annotations

import argparse

from arie.calibration import evaluate_calibration_grid
from arie.data import load_processed_dataset
from arie.datasets import DATASET_ID


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run calibrated evaluation for CiPA risk_class.")
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
    parser.add_argument(
        "--calib-splits",
        type=int,
        default=3,
        help="Number of stratified group folds for calibration split (default: 3).",
    )
    parser.add_argument(
        "--min-calib-per-class",
        type=int,
        default=50,
        help="Minimum calibration samples per class for isotonic (default: 50).",
    )
    return parser.parse_args()


def _print_split_diagnostics(split_diagnostics: dict) -> None:
    print("\nSplit diagnostics (all folds)")
    for seed in sorted(split_diagnostics.keys()):
        folds = split_diagnostics[seed]
        for fold in folds:
            print(
                f"- seed {seed} fold {fold['fold_index']}: "
                f"train/test overlap={fold['train_test_overlap']} | "
                f"train/calib overlap={fold['train_calib_overlap']} | "
                f"calib/test overlap={fold['calib_test_overlap']}"
            )
            if fold["missing_classes_in_test"]:
                print(f"  WARNING: missing classes in test: {fold['missing_classes_in_test']}")
            if fold["missing_classes_in_calib"]:
                print(f"  WARNING: missing classes in calib: {fold['missing_classes_in_calib']}")


def _print_comparison_table(table: list) -> None:
    print("\nComparison table (mean across seeds)")
    header = (
        "model      calib     status   bal_acc  log_loss  brier   ece"
    )
    print(header)
    print("-" * len(header))
    for row in table:
        if row["status"] != "OK":
            print(
                f"{row['model']:<10} {row['calibration']:<8} {row['status']:<8} "
                f"{row.get('reason', 'n/a')}"
            )
            continue
        print(
            f"{row['model']:<10} {row['calibration']:<8} {row['status']:<8} "
            f"{row['balanced_accuracy_mean']:.4f}  {row['log_loss_mean']:.4f}  "
            f"{row['brier_mean']:.4f}  {row['ece_mean']:.4f}"
        )


def main() -> None:
    args = _parse_args()
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    df = load_processed_dataset().dropna(subset=["risk_class"]).copy()

    print(f"Calibration run for {DATASET_ID}")
    print(f"- rows: {len(df)}")
    print(f"- unique drugs: {df['drug_name'].nunique()}")
    print(f"- unique risk_class: {df['risk_class'].nunique()}")
    print(f"- seeds: {seeds}")
    print(
        f"- split: StratifiedGroupKFold(n_splits={args.n_splits}, shuffle=True) "
        f"with inner n_splits={args.calib_splits}"
    )
    print("- models: log_reg, hist_gb")
    print("- calibration methods: none, sigmoid, isotonic")

    results = evaluate_calibration_grid(
        seeds=seeds,
        n_splits=args.n_splits,
        calibration_n_splits=args.calib_splits,
        min_calib_samples_per_class=args.min_calib_per_class,
    )

    _print_split_diagnostics(results["split_diagnostics"])
    _print_comparison_table(results["comparison_table"])

    print("\nSelected default")
    if results["default_choice"] is None:
        print("- none (no valid combos)")
    else:
        choice = results["default_choice"]
        print(f"- model: {choice['model']}")
        print(f"- calibration: {choice['calibration']}")
        print(f"- criteria: {choice['criteria']}")
        if results.get("default_selected"):
            sel = results["default_selected"]["summary"]
            print(
                f"- selected balanced_accuracy: {sel['balanced_accuracy']['mean']:.4f} "
                f"(std {sel['balanced_accuracy']['std']:.4f})"
            )
        if results.get("default_uncalibrated_reference"):
            uncal = results["default_uncalibrated_reference"]["summary"]
            print(
                f"- uncalibrated reference balanced_accuracy: {uncal['balanced_accuracy']['mean']:.4f} "
                f"(std {uncal['balanced_accuracy']['std']:.4f})"
            )

    print("\nMetrics saved to results/calibration_metrics.json")
    if results.get("predictions_path"):
        print(f"Predictions saved to {results['predictions_path']}")


if __name__ == "__main__":
    main()
