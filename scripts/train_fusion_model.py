#!/usr/bin/env python3
"""Train ML-only, mechanistic-only, and fused models with group-safe splits."""

from __future__ import annotations

import argparse
import warnings

from arie.fusion import train_fusion_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fusion models with mechanistic features.")
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


def _print_metrics(cfg: dict) -> None:
    print(
        f"- {cfg['name']}: bal_acc {cfg['balanced_accuracy_mean']:.4f} ± {cfg['balanced_accuracy_std']:.4f} | "
        f"macro_f1 {cfg['macro_f1_mean']:.4f} ± {cfg['macro_f1_std']:.4f} | "
        f"log_loss {cfg['log_loss_mean']:.4f} ± {cfg['log_loss_std']:.4f}"
    )
    raw = cfg["metrics"]["raw"]
    cal = cfg["metrics"]["calibrated"]
    print(
        f"  raw: log_loss {raw['log_loss']['mean']:.4f} ± {raw['log_loss']['std']:.4f} | "
        f"brier {raw['brier']['mean']:.4f} ± {raw['brier']['std']:.4f} | "
        f"ece {raw['ece']['mean']:.4f} ± {raw['ece']['std']:.4f}"
    )
    print(
        f"  cal: log_loss {cal['log_loss']['mean']:.4f} ± {cal['log_loss']['std']:.4f} | "
        f"brier {cal['brier']['mean']:.4f} ± {cal['brier']['std']:.4f} | "
        f"ece {cal['ece']['mean']:.4f} ± {cal['ece']['std']:.4f}"
    )


def main() -> None:
    args = _parse_args()
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    warnings.filterwarnings(
        "ignore",
        message="The y_prob values do not sum to one",
    )
    warnings.filterwarnings(
        "ignore",
        message="y_pred contains classes not in y_true",
    )

    print("Fusion training")
    print(f"- seeds: {seeds}")
    print(f"- splits: {args.n_splits}")

    outputs = train_fusion_model(seeds=seeds, n_splits=args.n_splits)

    join = outputs["join_summary"]
    print("\nJoin summary")
    print(f"- matched parents: {join['matched_parents']} / {join['cipa_unique_parents']}")
    print(f"- missing parents: {join['missing_parents']}")

    print("\nOverlap checks")
    for cfg in outputs["results"]["configs"]:
        print(f"- {cfg['name']}:")
        for run in cfg["runs"]:
            overlaps = run["overlap_counts"]
            print(f"  seed {run['seed']}: overlap counts per fold = {overlaps}")

    print("\nMetrics summary (mean ± std)")
    for cfg in outputs["results"]["configs"]:
        _print_metrics(cfg)

    print("\nFeature inventory")
    for cfg in outputs["results"]["configs"]:
        print(f"- {cfg['name']}:")
        print(f"  model: {cfg['model_type']}")
        print(f"  categorical ({cfg['n_categorical']}): {cfg['categorical_features']}")
        print(f"  numeric ({cfg['n_numeric']}): {cfg['numeric_features']}")
        shape = cfg.get("preprocessed_shape", {})
        if shape:
            print(f"  preprocessed shape: ({shape['n_rows']}, {shape['n_features']})")

    if outputs["results"].get("probability_self_check"):
        chk = outputs["results"]["probability_self_check"]
        print("\nProbability self-check")
        print(f"- checked rows: {chk['checked_rows']}")
        print(f"- mismatches: {chk['mismatches']}")

    print("\nAblation checks")
    for name, check in outputs["results"]["ablation_checks"].items():
        print(
            f"- {name}: pass={check['pass']} | bal_diff={check['balanced_accuracy_diff']:.6f} | "
            f"f1_diff={check['macro_f1_diff']:.6f}"
        )

    print("\nDeltas vs ML-only")
    for name, delta in outputs["results"]["deltas_vs_ml_only"].items():
        print(f"- {name}: {delta}")

    late = outputs["results"]["late_fusion"]
    print("\nLate fusion alpha sweep")
    for row in late["alpha_sweep"]:
        print(
            f"- alpha {row['alpha']}: bal_acc {row['balanced_accuracy']['mean']:.4f} | "
            f"log_loss {row['log_loss']['mean']:.4f}"
        )
    print(
        f"Selected alpha: {late['selected']['alpha']} (bal_acc {late['selected']['balanced_accuracy']['mean']:.4f}, "
        f"log_loss {late['selected']['log_loss']['mean']:.4f})"
    )

    print("\nAbstention comparison (balanced accuracy)")
    for cov in late["abstention_coverages"]:
        ml = late["abstention_ml_only"][str(cov)]["mean"]
        lf = late["abstention_late_fused"][str(cov)]["mean"]
        print(f"- coverage {cov:.2f}: ml_only {ml:.4f} | late_fused {lf:.4f} | delta {lf-ml:.4f}")

    print("\nArtifacts")
    print("- results/fusion_metrics.json")
    print("- results/fusion_join_summary.json")
    print("- results/fusion_predictions.csv")
    print("- figures/fusion_confusion_matrix.png")


if __name__ == "__main__":
    main()
