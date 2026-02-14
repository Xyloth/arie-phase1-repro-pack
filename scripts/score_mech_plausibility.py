#!/usr/bin/env python3
"""Score mechanistic plausibility for ML predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arie.mech_plausibility import OUTPUT_SCORES_PATH, OUTPUT_SUMMARY_PATH, score_mech_plausibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score mechanistic plausibility per prediction.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/calibration_predictions.csv"),
        help="Path to predictions CSV (default: results/calibration_predictions.csv).",
    )
    parser.add_argument(
        "--enable-identity-alias",
        action="store_true",
        help="Enable identity-changing aliases (default: off).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds used in predictions (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scored, summary = score_mech_plausibility(
        predictions_path=args.predictions,
        enable_identity_alias=args.enable_identity_alias,
        n_splits=args.n_splits,
    )

    OUTPUT_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUTPUT_SCORES_PATH, index=False)

    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)

    OUTPUT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    print("Mechanistic plausibility scoring complete.")
    print(f"- scored rows: {len(scored)}")
    print(f"- output scores: {OUTPUT_SCORES_PATH}")
    print(f"- summary: {OUTPUT_SUMMARY_PATH}")

    # Sanity checks
    pred_rows = summary.get("rows_total")
    scored_rows = summary.get("rows_scored")
    print(f"- counts align: {pred_rows} predictions vs {scored_rows} scored rows")

    overlaps = summary.get("overlap_checks", [])
    max_overlap = max((entry["overlap_count"] for entry in overlaps), default=0)
    print(f"- max train/test overlap (should be 0): {max_overlap}")

    print("Definitions:")
    print(f"- mech_support: {summary.get('mech_support_definition')}")
    print(f"- mech_support_higher_is: {summary.get('mech_support_higher_is')}")
    print(f"- mech_disagreement: {summary.get('mech_disagreement_definition')}")
    print(f"- mech_plausibility: {summary.get('mech_plausibility_definition')}")

    metrics = summary.get("metrics_by_seed", {})
    if metrics:
        print("Per-seed metrics (AUC vs error=1):")
        for seed, stats in metrics.items():
            print(
                f"- seed {seed}: auc_support={stats.get('roc_auc_support_vs_error')}, "
                f"auc_disagree={stats.get('roc_auc_disagreement_vs_error')}, "
                f"auc_missing={stats.get('roc_auc_missing_frac_vs_error')}, "
                f"auc_plaus={stats.get('roc_auc_plausibility_vs_error')}, "
                f"corr_support_disagree={stats.get('corr_support_vs_disagreement')}"
            )
        print("Missingness bins (AUC vs error=1):")
        for seed, stats in metrics.items():
            print(f"- seed {seed}:")
            for row in stats.get("missingness_bins", []):
                print(
                    f"  bin {row.get('bin')}: n={row.get('n_rows')} "
                    f"auc_support={row.get('auc_support')} auc_disagree={row.get('auc_disagreement')}"
                )

    conflict_count = summary.get("label_conflict_count", 0)
    if conflict_count:
        print(f"- label conflicts: {conflict_count} (see summary JSON)")
    else:
        print("- label conflicts: 0 (consistent per drug)")


if __name__ == "__main__":
    main()
