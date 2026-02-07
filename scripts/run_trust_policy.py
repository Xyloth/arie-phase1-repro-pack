#!/usr/bin/env python3
"""Run unified trust policy abstention analysis (confidence + OOD)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from arie.trust_policy import (
    CURVE_PATH,
    OOD_SCORES_PATH,
    PREDICTIONS_PATH,
    SCORES_PATH,
    SUMMARY_PATH,
    align_and_merge,
    add_percentiles,
    compute_policy_curve,
    default_alphas,
    default_rates,
    load_ood_scores,
    load_predictions,
    summarize_curve,
    write_json,
)


def main() -> None:
    predictions = load_predictions(PREDICTIONS_PATH)
    ood_scores = load_ood_scores(OOD_SCORES_PATH)

    merged, alignment = align_and_merge(predictions, ood_scores)
    print(
        "Alignment check (predictions vs OOD scores): "
        f"pred n={alignment['pred_rows']} | "
        f"ood n={alignment['ood_rows']} | sym_diff={alignment['sym_diff']}"
    )

    merged = add_percentiles(merged)

    rates = default_rates()
    alphas = default_alphas()

    confidence_curve = compute_policy_curve(merged, rates, policy="confidence")
    ood_curve = compute_policy_curve(merged, rates, policy="ood")

    combined_curves = {}
    combined_summaries = {}
    for alpha in alphas:
        curve = compute_policy_curve(merged, rates, policy="combined", alpha=alpha)
        combined_curves[alpha] = curve
        combined_summaries[alpha] = summarize_curve(curve, rates)

    def _score(summary: dict) -> tuple:
        def _val(rate: str) -> float:
            return summary.get(rate, {}).get("balanced_accuracy_mean", float("nan"))

        return (_val("0.3"), _val("0.1"), _val("0.5"))

    selected_alpha = None
    best_score = None
    for alpha in alphas:
        score = _score(combined_summaries[alpha])
        if best_score is None or score > best_score:
            best_score = score
            selected_alpha = alpha

    assert selected_alpha is not None

    combined_curve = combined_curves[selected_alpha]

    curve_df = pd.concat([confidence_curve, ood_curve, combined_curve], ignore_index=True)
    CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(CURVE_PATH, index=False)

    summary_conf = summarize_curve(confidence_curve, rates)
    summary_ood = summarize_curve(ood_curve, rates)
    summary_comb = summarize_curve(combined_curve, rates)

    baseline = summary_conf.get("0.0", {})
    baseline_bal = baseline.get("balanced_accuracy_mean")

    def _policy_entry(summary: dict, rate: str) -> dict:
        entry = summary.get(rate, {})
        bal = entry.get("balanced_accuracy_mean")
        delta = None if baseline_bal is None or bal is None else bal - baseline_bal
        return {
            "balanced_accuracy_mean": bal,
            "delta_vs_baseline": delta,
            "coverage_mean": entry.get("coverage_mean"),
        }

    comparison = {}
    for rate, coverage in [("0.1", "0.90"), ("0.3", "0.70"), ("0.5", "0.50")]:
        comparison[coverage] = {
            "confidence": _policy_entry(summary_conf, rate),
            "ood": _policy_entry(summary_ood, rate),
            "combined": _policy_entry(summary_comb, rate),
        }

    # Optional per-sample scores for selected alpha
    scores_df = merged.copy()
    scores_df["error"] = (scores_df["y_pred"] != scores_df["y_true"]).astype(int)
    scores_df["untrust"] = selected_alpha * scores_df["ood_percentile"] + (
        1 - selected_alpha
    ) * (1 - scores_df["conf_percentile"])
    scores_df["policy_rank"] = scores_df.groupby(["seed", "fold"])["untrust"].rank(
        ascending=False, method="first"
    )
    scores_df_out = scores_df[
        [
            "row_id",
            "seed",
            "fold",
            "y_true",
            "y_pred",
            "error",
            "confidence",
            "ood_score",
            "ood_percentile",
            "conf_percentile",
            "untrust",
            "policy_rank",
        ]
    ]
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores_df_out.to_csv(SCORES_PATH, index=False)

    ood_summary_path = Path("results/abstention_ood_knn_summary.json")
    ood_meta = {}
    if ood_summary_path.exists():
        ood_meta = json.loads(ood_summary_path.read_text())

    alpha_sweep = []
    for alpha in alphas:
        summary = combined_summaries[alpha]
        alpha_sweep.append(
            {
                "alpha": alpha,
                "bal_acc_0.90": summary.get("0.1", {}).get("balanced_accuracy_mean"),
                "bal_acc_0.70": summary.get("0.3", {}).get("balanced_accuracy_mean"),
                "bal_acc_0.50": summary.get("0.5", {}).get("balanced_accuracy_mean"),
            }
        )

    summary_payload = {
        "rates": rates,
        "alphas": alphas,
        "selected_alpha": selected_alpha,
        "selection_criteria": "maximize bal_acc at 0.70 coverage, then 0.90, then 0.50",
        "alpha_sweep": alpha_sweep,
        "baseline": {
            "abstention_rate": 0.0,
            "balanced_accuracy_mean": baseline_bal,
            "coverage_mean": baseline.get("coverage_mean"),
            "macro_f1_mean": baseline.get("macro_f1_mean"),
        },
        "policy_comparison": comparison,
        "confidence_definition": "confidence = max predicted probability",
        "ood_metadata": {
            "selected_method": ood_meta.get("selected_method"),
            "selected_k": ood_meta.get("selected_k"),
            "score_definition": ood_meta.get("score_definition"),
            "drop_policy": ood_meta.get("drop_policy"),
        },
        "summary": {
            "confidence": summary_conf,
            "ood": summary_ood,
            "combined": summary_comb,
        },
        "curve_path": str(CURVE_PATH),
        "scores_path": str(SCORES_PATH),
    }

    write_json(SUMMARY_PATH, summary_payload)

    print("Alpha sweep (balanced accuracy):")
    for row in alpha_sweep:
        print(
            f"- alpha {row['alpha']:.2f}: "
            f"@0.90 {row['bal_acc_0.90']:.4f} | "
            f"@0.70 {row['bal_acc_0.70']:.4f} | "
            f"@0.50 {row['bal_acc_0.50']:.4f}"
        )
    print(f"Selected alpha: {selected_alpha}")
    print("Coverage comparison (balanced accuracy, delta vs baseline):")
    for coverage in ["0.90", "0.70", "0.50"]:
        row = comparison[coverage]
        print(
            f"- coverage {coverage}: "
            f"conf {row['confidence']['balanced_accuracy_mean']:.4f} "
            f"({row['confidence']['delta_vs_baseline']:+.4f}) | "
            f"ood {row['ood']['balanced_accuracy_mean']:.4f} "
            f"({row['ood']['delta_vs_baseline']:+.4f}) | "
            f"combined {row['combined']['balanced_accuracy_mean']:.4f} "
            f"({row['combined']['delta_vs_baseline']:+.4f})"
        )

    print("Outputs written:")
    print(f"- {CURVE_PATH}")
    print(f"- {SUMMARY_PATH}")
    print(f"- {SCORES_PATH}")


if __name__ == "__main__":
    main()
