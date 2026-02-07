#!/usr/bin/env python3
"""Run confidence-based abstention analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arie.abstention import compute_abstention_confidence_curve


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run confidence-based abstention analysis.")
    parser.add_argument(
        "--rates",
        type=str,
        default="",
        help="Comma-separated abstention rates (default: use 0 to 0.5 by 0.05).",
    )
    parser.add_argument(
        "--calibration-metrics",
        type=str,
        default="results/calibration_metrics.json",
        help="Path to calibration_metrics.json for 0%% sanity check.",
    )
    return parser.parse_args()


def _parse_rates(args: argparse.Namespace) -> list[float]:
    if not args.rates.strip():
        return []
    return [float(x.strip()) for x in args.rates.split(",") if x.strip()]


def main() -> None:
    args = _parse_args()
    rates = _parse_rates(args)

    results = compute_abstention_confidence_curve(rates=rates or None)

    summary = results["summary"]
    print("Abstention summary (selected rates)")
    for rate_str in ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]:
        if rate_str not in summary:
            continue
        row = summary[rate_str]
        print(
            f"- rate {rate_str}: coverage {row['coverage_mean']:.3f} | "
            f"bal_acc {row['balanced_accuracy_mean']:.4f} | macro_f1 {row['macro_f1_mean']:.4f}"
        )

    # Sanity check at 0% abstention against calibration metrics
    calibration_path = Path(args.calibration_metrics)
    if calibration_path.exists() and "0.0" in summary:
        with calibration_path.open() as f:
            cal = json.load(f)
        default_selected = cal.get("default_selected", {}).get("summary", {})
        expected = default_selected.get("balanced_accuracy", {}).get("mean")
        if expected is not None:
            observed = summary["0.0"]["balanced_accuracy_mean"]
            diff = observed - expected
            print(
                f"Sanity check @0% abstention: observed {observed:.4f} vs "
                f"expected {expected:.4f} (diff {diff:+.4f})"
            )
        else:
            print("Sanity check: no default_selected balanced_accuracy found in calibration_metrics.json")
    else:
        print("Sanity check skipped: calibration_metrics.json missing or 0.0 rate not present")

    print("Outputs written:")
    print(f"- results/abstention_confidence_curve.csv")
    print(f"- results/abstention_confidence_summary.json")


if __name__ == "__main__":
    main()
