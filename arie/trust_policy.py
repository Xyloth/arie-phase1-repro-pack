"""Unified trust policy combining confidence and OOD scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT / "results" / "calibration_predictions.csv"
OOD_SCORES_PATH = ROOT / "results" / "ood_knn_scores.csv"
CURVE_PATH = ROOT / "results" / "abstention_trust_policy_curve.csv"
SUMMARY_PATH = ROOT / "results" / "abstention_trust_policy_summary.json"
SCORES_PATH = ROOT / "results" / "trust_policy_scores.csv"


def default_rates() -> List[float]:
    return [round(x, 2) for x in np.arange(0.0, 0.51, 0.05)]


def default_alphas() -> List[float]:
    return [0.0, 0.25, 0.5, 0.75, 1.0]


def load_predictions(predictions_path: Path | None = None) -> pd.DataFrame:
    predictions_path = predictions_path or PREDICTIONS_PATH
    df = pd.read_csv(predictions_path)
    if "split_role" in df.columns:
        df = df[df["split_role"] == "test"].copy()
    required_cols = {"row_id", "seed", "fold", "confidence", "y_true", "y_pred"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in predictions: {sorted(missing)}")
    return df


def load_ood_scores(ood_path: Path | None = None) -> pd.DataFrame:
    ood_path = ood_path or OOD_SCORES_PATH
    df = pd.read_csv(ood_path)
    required_cols = {"row_id", "seed", "fold", "ood_knn_distance"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in OOD scores: {sorted(missing)}")
    return df


def align_and_merge(
    predictions: pd.DataFrame, ood: pd.DataFrame
) -> Tuple[pd.DataFrame, dict]:
    pred_keys = set(
        zip(predictions["row_id"].astype(int), predictions["seed"], predictions["fold"])
    )
    ood_keys = set(zip(ood["row_id"].astype(int), ood["seed"], ood["fold"]))
    sym_diff = pred_keys.symmetric_difference(ood_keys)
    alignment = {
        "pred_rows": int(len(pred_keys)),
        "ood_rows": int(len(ood_keys)),
        "sym_diff": int(len(sym_diff)),
    }
    if sym_diff:
        raise RuntimeError("Alignment check failed: predictions and OOD scores do not match.")

    merged = predictions.merge(
        ood,
        on=["row_id", "seed", "fold"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_ood"),
    )
    merged = merged.copy()
    merged["ood_score"] = merged["ood_knn_distance"].astype(float)
    return merged, alignment


def add_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _pct_rank(series: pd.Series) -> pd.Series:
        return series.rank(pct=True, method="average")

    df["conf_percentile"] = df.groupby(["seed", "fold"])["confidence"].transform(_pct_rank)
    df["ood_percentile"] = df.groupby(["seed", "fold"])["ood_score"].transform(_pct_rank)
    df["untrust_conf"] = 1.0 - df["conf_percentile"]
    df["untrust_ood"] = df["ood_percentile"]
    return df


def compute_policy_curve(
    df: pd.DataFrame,
    rates: Iterable[float],
    policy: str,
    alpha: float | None = None,
) -> pd.DataFrame:
    if policy not in {"confidence", "ood", "combined"}:
        raise ValueError(f"Unknown policy: {policy}")

    rates = sorted(set(float(r) for r in rates))
    labels = sorted(df["y_true"].unique())

    rows = []
    for (seed, fold), group in df.groupby(["seed", "fold"], sort=True):
        if policy == "confidence":
            score = group["untrust_conf"]
        elif policy == "ood":
            score = group["untrust_ood"]
        else:
            if alpha is None:
                raise ValueError("alpha is required for combined policy")
            score = alpha * group["ood_percentile"] + (1 - alpha) * (1 - group["conf_percentile"])

        group_sorted = group.assign(_score=score).sort_values("_score", ascending=False)
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
                    "policy": policy,
                    "abstention_rate": rate,
                    "coverage": coverage,
                    "n_total": int(n_total),
                    "n_kept": int(n_kept),
                    "balanced_accuracy": float(bal_acc),
                    "macro_f1": float(macro_f1),
                    "alpha": float(alpha) if alpha is not None else np.nan,
                }
            )

    return pd.DataFrame(rows)


def summarize_curve(curve_df: pd.DataFrame, rates: Iterable[float]) -> dict:
    rates = sorted(set(float(r) for r in rates))
    summary = {}
    for rate in rates:
        subset = curve_df[curve_df["abstention_rate"] == rate]
        if subset.empty:
            continue
        summary[str(rate)] = {
            "coverage_mean": float(subset["coverage"].mean()),
            "coverage_std": float(subset["coverage"].std(ddof=1)) if len(subset) > 1 else 0.0,
            "balanced_accuracy_mean": float(subset["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(subset["balanced_accuracy"].std(ddof=1))
            if len(subset) > 1
            else 0.0,
            "macro_f1_mean": float(subset["macro_f1"].mean()),
            "macro_f1_std": float(subset["macro_f1"].std(ddof=1)) if len(subset) > 1 else 0.0,
            "n_groups": int(len(subset)),
        }
    return summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
