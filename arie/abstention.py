"""Confidence-based abstention analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT / "results" / "calibration_predictions.csv"
CURVE_PATH = ROOT / "results" / "abstention_confidence_curve.csv"
SUMMARY_PATH = ROOT / "results" / "abstention_confidence_summary.json"


def _default_rates() -> List[float]:
    return [round(x, 2) for x in np.arange(0.0, 0.51, 0.05)]


def compute_abstention_confidence_curve(
    predictions_path: Path | None = None,
    rates: Iterable[float] | None = None,
    curve_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict:
    predictions_path = predictions_path or PREDICTIONS_PATH
    curve_path = curve_path or CURVE_PATH
    summary_path = summary_path or SUMMARY_PATH

    df = pd.read_csv(predictions_path)

    required_cols = {"seed", "fold", "confidence", "y_true", "y_pred"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in predictions: {sorted(missing)}")

    if "split_role" in df.columns:
        df = df[df["split_role"] == "test"].copy()

    rates = list(rates or _default_rates())
    rates = sorted(set(float(r) for r in rates))

    labels = sorted(df["y_true"].unique())

    rows = []
    for (seed, fold), group in df.groupby(["seed", "fold"], sort=True):
        group_sorted = group.sort_values("confidence", ascending=True)
        n_total = len(group_sorted)
        if n_total == 0:
            continue

        for rate in rates:
            drop_n = int(np.floor(rate * n_total))
            kept = group_sorted.iloc[drop_n:]
            n_kept = len(kept)
            coverage = n_kept / n_total if n_total else 0.0

            if n_kept == 0:
                bal_acc = float("nan")
                macro_f1 = float("nan")
            else:
                bal_acc = balanced_accuracy_score(kept["y_true"], kept["y_pred"])
                macro_f1 = f1_score(kept["y_true"], kept["y_pred"], labels=labels, average="macro")

            rows.append(
                {
                    "seed": int(seed),
                    "fold": int(fold),
                    "abstention_rate": rate,
                    "coverage": coverage,
                    "n_total": int(n_total),
                    "n_kept": int(n_kept),
                    "balanced_accuracy": float(bal_acc),
                    "macro_f1": float(macro_f1),
                }
            )

    curve_df = pd.DataFrame(rows)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(curve_path, index=False)

    summary = {}
    for rate in rates:
        subset = curve_df[curve_df["abstention_rate"] == rate]
        if subset.empty:
            continue
        summary[str(rate)] = {
            "coverage_mean": float(subset["coverage"].mean()),
            "coverage_std": float(subset["coverage"].std(ddof=1)) if len(subset) > 1 else 0.0,
            "balanced_accuracy_mean": float(subset["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(subset["balanced_accuracy"].std(ddof=1)) if len(subset) > 1 else 0.0,
            "macro_f1_mean": float(subset["macro_f1"].mean()),
            "macro_f1_std": float(subset["macro_f1"].std(ddof=1)) if len(subset) > 1 else 0.0,
            "n_groups": int(len(subset)),
        }

    baseline = summary.get("0.0")
    best = None
    for rate, stats in summary.items():
        if best is None or stats["balanced_accuracy_mean"] > best["balanced_accuracy_mean"]:
            best = {
                "abstention_rate": float(rate),
                "coverage_mean": stats["coverage_mean"],
                "balanced_accuracy_mean": stats["balanced_accuracy_mean"],
            }

    def _coverage_entry(target_rate: float) -> dict | None:
        key = f"{target_rate:.1f}"
        if key not in summary:
            return None
        entry = summary[key]
        delta = None
        if baseline is not None:
            delta = entry["balanced_accuracy_mean"] - baseline["balanced_accuracy_mean"]
        return {
            "abstention_rate": target_rate,
            "coverage_mean": entry["coverage_mean"],
            "balanced_accuracy_mean": entry["balanced_accuracy_mean"],
            "delta_vs_baseline": delta,
        }

    summary_payload = {
        "predictions_path": str(predictions_path),
        "curve_path": str(curve_path),
        "rates": rates,
        "baseline": {
            "abstention_rate": 0.0,
            "coverage_mean": baseline["coverage_mean"] if baseline else None,
            "balanced_accuracy_mean": baseline["balanced_accuracy_mean"] if baseline else None,
            "macro_f1_mean": baseline["macro_f1_mean"] if baseline else None,
        },
        "best_balanced_accuracy": best,
        "coverage_targets": {
            "0.90": _coverage_entry(0.10),
            "0.70": _coverage_entry(0.30),
        },
        "summary": summary,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    return summary_payload
