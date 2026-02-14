"""Trust policy evaluation with mechanistic plausibility."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from arie.trust_policy import (
    OOD_SCORES_PATH,
    PREDICTIONS_PATH,
    add_percentiles,
    load_ood_scores,
    load_predictions,
)

ROOT = Path(__file__).resolve().parents[1]
MECH_SCORES_PATH = ROOT / "results" / "mech_plausibility_scores.csv"
MECH_SUMMARY_PATH = ROOT / "results" / "mech_plausibility_summary.json"

TRUST_POLICY_SUMMARY_PATH = ROOT / "results" / "abstention_trust_policy_summary.json"

CURVE_PATH = ROOT / "results" / "abstention_trust_policy_mech_curve.csv"
SUMMARY_PATH = ROOT / "results" / "abstention_trust_policy_mech_summary.json"
SCORES_PATH = ROOT / "results" / "trust_policy_scores_with_mech.csv"

COVERAGE_TARGETS = [0.90, 0.70, 0.50]
ABSTENTION_RATES = [0.10, 0.30, 0.50]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_mech_scores(path: Path | None = None) -> pd.DataFrame:
    path = path or MECH_SCORES_PATH
    df = pd.read_csv(path)
    if "split_role" in df.columns:
        df = df[df["split_role"] == "test"].copy()
    required = {
        "row_id",
        "seed",
        "fold",
        "mech_support",
        "mech_missing_frac",
        "mech_plausibility",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required mech columns: {sorted(missing)}")
    return df


def alignment_check(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    left_keys = set(zip(left["row_id"].astype(int), left["seed"], left["fold"]))
    right_keys = set(zip(right["row_id"].astype(int), right["seed"], right["fold"]))
    sym_diff = left_keys.symmetric_difference(right_keys)
    return {
        "left_rows": int(len(left_keys)),
        "right_rows": int(len(right_keys)),
        "sym_diff": int(len(sym_diff)),
    }


def load_selected_alpha() -> float:
    if TRUST_POLICY_SUMMARY_PATH.exists():
        payload = json.loads(TRUST_POLICY_SUMMARY_PATH.read_text())
        alpha = payload.get("selected_alpha")
        if alpha is not None:
            return float(alpha)
    # Fallback if summary missing
    return 0.5


def compute_variant_curves(
    df: pd.DataFrame,
    variants: Iterable[str],
    alpha: float,
    eps: float,
) -> pd.DataFrame:
    rows = []
    for (seed, fold), group in df.groupby(["seed", "fold"], sort=True):
        n_total = len(group)
        if n_total == 0:
            continue

        # Base untrust scores
        untrust_conf = group["untrust_conf"]
        untrust_ood = group["untrust_ood"]
        untrust_combined = group["untrust_combined"]

        for variant in variants:
            if variant == "confidence":
                untrust = untrust_conf
            elif variant == "ood":
                untrust = untrust_ood
            elif variant == "combined":
                untrust = untrust_combined
            elif variant == "combined_plus_mech":
                untrust = 1.0 - group["score_combined_plus_mech"]
            elif variant == "combined_plus_mech_eps":
                untrust = 1.0 - group["score_combined_plus_mech_eps"]
            else:
                raise ValueError(f"Unknown variant: {variant}")

            ranked = group.assign(_score=untrust).sort_values("_score", ascending=False)
            for rate, coverage_target in zip(ABSTENTION_RATES, COVERAGE_TARGETS):
                drop_n = int(np.floor(rate * n_total))
                kept = ranked.iloc[drop_n:]
                n_kept = len(kept)
                coverage = n_kept / n_total if n_total else 0.0
                if n_kept == 0:
                    bal_acc = float("nan")
                else:
                    bal_acc = float(
                        balanced_accuracy_score(kept["y_true"], kept["y_pred"])
                    )

                rows.append(
                    {
                        "seed": int(seed),
                        "fold": int(fold),
                        "coverage_target": float(coverage_target),
                        "coverage": float(coverage),
                        "variant": variant,
                        "balanced_accuracy": bal_acc,
                        "n_total": int(n_total),
                        "n_kept": int(n_kept),
                    }
                )
    return pd.DataFrame(rows)


def summarize_curves(curve_df: pd.DataFrame) -> dict:
    summary = {}
    for coverage_target in COVERAGE_TARGETS:
        cov_key = f"{coverage_target:.2f}"
        summary[cov_key] = {}
        for variant, group in curve_df[curve_df["coverage_target"] == coverage_target].groupby(
            "variant"
        ):
            summary[cov_key][variant] = {
                "balanced_accuracy_mean": float(group["balanced_accuracy"].mean()),
                "balanced_accuracy_std": float(
                    group["balanced_accuracy"].std(ddof=1)
                )
                if len(group) > 1
                else 0.0,
                "coverage_mean": float(group["coverage"].mean()),
                "coverage_std": float(group["coverage"].std(ddof=1))
                if len(group) > 1
                else 0.0,
                "n_groups": int(len(group)),
            }
    return summary


def compute_deltas(summary: dict, baseline_variant: str) -> dict:
    deltas = {}
    for cov_key, variants in summary.items():
        base = variants.get(baseline_variant, {}).get("balanced_accuracy_mean")
        deltas[cov_key] = {}
        for variant, stats in variants.items():
            bal = stats.get("balanced_accuracy_mean")
            delta = None
            if base is not None and bal is not None:
                delta = bal - base
            deltas[cov_key][variant] = {
                "balanced_accuracy_mean": bal,
                "delta_vs_combined": delta,
            }
    return deltas


def best_variants(summary: dict) -> dict:
    winners = {}
    for cov_key, variants in summary.items():
        best_variant = None
        best_score = None
        for variant, stats in variants.items():
            score = stats.get("balanced_accuracy_mean")
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_variant = variant
        winners[cov_key] = {
            "variant": best_variant,
            "balanced_accuracy_mean": best_score,
        }
    return winners


def run_trust_policy_mech(
    eps: float = 0.05,
) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    predictions = load_predictions(PREDICTIONS_PATH)
    ood_scores = load_ood_scores(OOD_SCORES_PATH)
    mech_scores = load_mech_scores(MECH_SCORES_PATH)

    align_pred_ood = alignment_check(predictions, ood_scores)
    align_pred_mech = alignment_check(predictions, mech_scores)
    if align_pred_mech["sym_diff"]:
        raise RuntimeError("Alignment check failed: predictions vs mech scores mismatch.")

    merged = predictions.merge(
        ood_scores,
        on=["row_id", "seed", "fold"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_ood"),
    )
    merged["ood_score"] = merged["ood_knn_distance"].astype(float)
    merged = merged.merge(
        mech_scores[
            [
                "row_id",
                "seed",
                "fold",
                "mech_support",
                "mech_missing_frac",
                "mech_plausibility",
            ]
        ],
        on=["row_id", "seed", "fold"],
        how="inner",
        validate="one_to_one",
    )

    merged = add_percentiles(merged)

    alpha = load_selected_alpha()
    merged["untrust_combined"] = alpha * merged["ood_percentile"] + (1 - alpha) * (
        1 - merged["conf_percentile"]
    )
    merged["score_combined"] = 1.0 - merged["untrust_combined"]
    merged["score_combined_plus_mech"] = (
        merged["score_combined"] * merged["mech_plausibility"]
    )
    merged["score_combined_plus_mech_eps"] = merged["score_combined"] * (
        eps + merged["mech_plausibility"]
    )

    merged["error"] = (merged["y_pred"] != merged["y_true"]).astype(int)

    variants = [
        "confidence",
        "ood",
        "combined",
        "combined_plus_mech",
        "combined_plus_mech_eps",
    ]

    curve_df = compute_variant_curves(merged, variants, alpha=alpha, eps=eps)
    summary = summarize_curves(curve_df)
    deltas = compute_deltas(summary, baseline_variant="combined")
    winners = best_variants(summary)

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "alignment": {
            "predictions_vs_ood": align_pred_ood,
            "predictions_vs_mech": align_pred_mech,
            "join_keys": ["row_id", "seed", "fold"],
        },
        "alpha": alpha,
        "eps": eps,
        "coverage_targets": COVERAGE_TARGETS,
        "summary": summary,
        "deltas_vs_combined": deltas,
        "best_variants": winners,
        "mech_scores_path": str(MECH_SCORES_PATH),
        "mech_scores_sha256": _sha256_file(MECH_SCORES_PATH),
        "mech_summary_path": str(MECH_SUMMARY_PATH),
        "mech_summary_sha256": _sha256_file(MECH_SUMMARY_PATH)
        if MECH_SUMMARY_PATH.exists()
        else None,
        "trust_policy_scores_path": str(SCORES_PATH),
        "curve_path": str(CURVE_PATH),
    }

    return merged, summary_payload, curve_df


__all__ = [
    "run_trust_policy_mech",
    "CURVE_PATH",
    "SUMMARY_PATH",
    "SCORES_PATH",
]
