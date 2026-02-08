#!/usr/bin/env python3
"""Row-level mechanistic fusion diagnostic run."""

from __future__ import annotations

import argparse

import pandas as pd

from arie.fusion_rowlevel import run_rowlevel_diagnostic
from arie.data import load_processed_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run row-level mechanistic fusion diagnostics.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--enable-block-pred",
        action="store_true",
        help="Compute herg_block_pred using concentration_level (units assumed; use with caution).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    df = load_processed_dataset()
    print("Concentration evidence (first 10 rows):")
    print(df[["drug_name", "concentration_level"]].head(10).to_string(index=False))
    conc = df["concentration_level"]
    print(f"Concentration min={conc.min()} max={conc.max()} unique_count={conc.nunique()}")
    print(f"Unique values (up to 10): {sorted(conc.dropna().unique().tolist())[:10]}")

    report = run_rowlevel_diagnostic(
        seeds=seeds,
        n_splits=args.n_splits,
        enable_block_pred=args.enable_block_pred,
    )

    print("\nRow-level diagnostic metrics (mean ± std)")
    for name, cfg in report["metrics"]["configs"].items():
        m = cfg["metrics"]
        print(
            f"- {name}: bal_acc {m['balanced_accuracy']['mean']:.4f} ± {m['balanced_accuracy']['std']:.4f} | "
            f"log_loss {m['log_loss']['mean']:.4f} ± {m['log_loss']['std']:.4f} | "
            f"brier {m['brier']['mean']:.4f} ± {m['brier']['std']:.4f} | "
            f"ece {m['ece']['mean']:.4f} ± {m['ece']['std']:.4f}"
        )

    print(f"\nnh_imputed_rate: {report['nh_imputed_rate']:.3f}")
    print(f"fusion_beats_ml: {report['fusion_beats_ml']}")

    if report.get("feature_importance"):
        print("\nMechanistic feature importance (permutation, seed 42 fold 0)")
        for name, val in report["feature_importance"]["mechanistic_importances"]:
            print(f"- {name}: {val:.6f}")

    print("\nArtifacts")
    print("- results/fusion_rowlevel_diagnostic.json")


if __name__ == "__main__":
    main()
