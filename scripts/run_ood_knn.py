#!/usr/bin/env python3
"""Run kNN distance-based applicability domain analysis and abstention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from arie.ood import (
    compute_abstention_ood_curve,
    compute_knn_ood_scores,
    CURVE_PATH,
    OOD_SCORES_PATH,
    PREDICTIONS_PATH,
    SUMMARY_PATH,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run kNN OOD scoring and abstention.")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of neighbors for kNN distance (default: use best from k-sweep, currently 10).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cosine_sparse",
        choices=[
            "pca_euclidean_margin",
            "pca_euclidean_class",
            "pca_euclidean",
            "cosine_sparse_margin",
            "cosine_sparse_class",
            "cosine_sparse",
        ],
        help="OOD distance method (default: cosine_sparse).",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=30,
        help="Number of PCA/SVD components if using pca_euclidean (default: 30).",
    )
    parser.add_argument(
        "--calibration-metrics",
        type=str,
        default="results/calibration_metrics.json",
        help="Path to calibration_metrics.json for 0%% sanity check.",
    )
    return parser.parse_args()


def _raw_score_definition(method: str, k: int, n_components: int) -> str:
    is_margin = method.endswith("_margin")
    class_conditional = method.endswith("_class") or is_margin
    base_method = method.replace("_class", "").replace("_margin", "")

    if base_method == "cosine_sparse":
        metric = "cosine distance"
        space = "sparse one-hot feature space"
    elif base_method == "pca_euclidean":
        metric = "euclidean distance"
        space = f"SVD/PCA space (n_components={n_components})"
    else:
        metric = "distance"
        space = "feature space"

    if is_margin:
        return (
            f"mean {metric} to kNN within predicted class minus mean {metric} "
            f"to kNN in full training set (k={k}, train-only, {space})"
        )
    if class_conditional:
        return f"mean {metric} to kNN within predicted class (k={k}, train-only, {space})"
    return f"mean {metric} to kNN in full training set (k={k}, train-only, {space})"


def main() -> None:
    args = _parse_args()

    pred = pd.read_csv(PREDICTIONS_PATH)
    if "split_role" in pred.columns:
        pred = pred[pred["split_role"] == "test"].copy()
    k_values = [3, 5, 10]
    k_sweep = []
    results_by_k = {}

    def _evaluate_k(k: int) -> dict:
        ood_distances, overlaps, method_meta = compute_knn_ood_scores(
            k=k, method=args.method, n_components=args.pca_components
        )

        # Alignment check between predictions and OOD distances
        print(f"Alignment check (predictions vs OOD distances) [k={k}]")
        pred_groups = pred.groupby(["seed", "fold"])
        ood_groups = ood_distances.groupby(["seed", "fold"])
        for (seed, fold), pred_group in pred_groups:
            pred_ids = set(pred_group["row_id"].astype(int).tolist())
            ood_group = (
                ood_groups.get_group((seed, fold)) if (seed, fold) in ood_groups.groups else None
            )
            ood_ids = set() if ood_group is None else set(ood_group["row_id"].astype(int).tolist())
            diff = pred_ids.symmetric_difference(ood_ids)
            print(
                f"- seed {seed} fold {fold}: pred n={len(pred_ids)} | "
                f"ood n={len(ood_ids)} | sym_diff={len(diff)}"
            )
            if diff:
                raise RuntimeError("Split alignment mismatch between predictions and OOD scoring.")

        ood = pred.merge(
            ood_distances, on=["row_id", "seed", "fold"], how="left", validate="one_to_one"
        )
        if ood["ood_knn_distance"].isna().any():
            raise RuntimeError("Missing OOD distances after merge; check split alignment.")

        # Define direction: higher score => more OOD / lower trust
        ood["error"] = (ood["y_pred"] != ood["y_true"]).astype(int)
        ood["ood_knn_raw"] = ood["ood_knn_distance"].copy()
        correct_mean_raw = ood[ood["error"] == 0]["ood_knn_raw"].mean()
        incorrect_mean_raw = ood[ood["error"] == 1]["ood_knn_raw"].mean()
        flipped = False
        if incorrect_mean_raw < correct_mean_raw:
            ood["ood_knn_distance"] = -ood["ood_knn_distance"]
            flipped = True
            flip_reason = "incorrect_mean_raw < correct_mean_raw (raw scores inverted)"
        else:
            flip_reason = "incorrect_mean_raw >= correct_mean_raw (no flip needed)"

        method_meta = {
            **method_meta,
            "direction": "higher=more_ood",
            "flipped": flipped,
        }

        summary = compute_abstention_ood_curve(ood, method_meta=method_meta, write_outputs=False)
        raw_score_definition = _raw_score_definition(args.method, k, args.pca_components)

        return {
            "ood": ood,
            "summary": summary,
            "method_meta": method_meta,
            "overlaps": overlaps,
            "raw_score_definition": raw_score_definition,
            "flip_reason": flip_reason,
        }

    for k in k_values:
        result = _evaluate_k(k)
        summary = result["summary"]

        def _get_bal(rate: str) -> float:
            return summary["summary"].get(rate, {}).get("balanced_accuracy_mean", float("nan"))

        k_sweep.append(
            {
                "k": k,
                "bal_acc_0": _get_bal("0.0"),
                "bal_acc_0.1": _get_bal("0.1"),
                "bal_acc_0.3": _get_bal("0.3"),
                "bal_acc_0.5": _get_bal("0.5"),
            }
        )

        results_by_k[k] = result

    # Select default k: best at 70% coverage (0.3 abstention), tie-breaker 0.5
    def _score(entry: dict) -> tuple:
        return (entry["bal_acc_0.3"], entry["bal_acc_0.5"])

    k_sweep_sorted = sorted(k_sweep, key=_score, reverse=True)
    best_k = k_sweep_sorted[0]["k"]

    print("\nK-sweep summary (bal_acc at key rates)")
    for entry in k_sweep:
        print(
            f"- k={entry['k']}: @0% {entry['bal_acc_0']:.4f} | "
            f"@10% {entry['bal_acc_0.1']:.4f} | @30% {entry['bal_acc_0.3']:.4f} | "
            f"@50% {entry['bal_acc_0.5']:.4f}"
        )
    print(f"Selected k: {best_k} (best @70% coverage, then @50%)")

    selected_k = args.k if args.k is not None else best_k
    if selected_k not in results_by_k:
        print(f"Selected k={selected_k} not in k-sweep list; evaluating separately.")
        results_by_k[selected_k] = _evaluate_k(selected_k)

    # Use selected k for outputs
    selected = results_by_k[selected_k]
    ood = selected["ood"]
    method_meta = selected["method_meta"]
    overlaps = selected["overlaps"]
    raw_score_definition = selected["raw_score_definition"]
    flip_reason = selected["flip_reason"]

    print(f"Selected method: {args.method}")
    print(f"Selected k for outputs: {selected_k} (best from sweep: {best_k})")
    print(
        "OOD score direction: higher=more_ood (abstain highest first) "
        f"[flipped={method_meta.get('flipped', False)}]"
    )
    final_score_definition = (
        "ood_score = raw_score * (-1 if flipped else +1) to enforce higher=more_ood"
    )
    print(f"Score definition: raw={raw_score_definition} | final={final_score_definition}")

    OOD_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ood.to_csv(OOD_SCORES_PATH, index=False)

    # OOD usefulness diagnostics (numeric)
    print("\nOOD diagnostics by seed (higher = more OOD)")
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    diagnostics = []
    for seed, group in ood.groupby("seed"):
        correct_mean = group[group["error"] == 0]["ood_knn_distance"].mean()
        incorrect_mean = group[group["error"] == 1]["ood_knn_distance"].mean()
        corr, _ = spearmanr(group["ood_knn_distance"], group["error"])
        try:
            auc = roc_auc_score(group["error"], group["ood_knn_distance"])
        except ValueError:
            auc = float("nan")
        diagnostics.append(
            {
                "seed": int(seed),
                "mean_correct": float(correct_mean),
                "mean_incorrect": float(incorrect_mean),
                "spearman_r": float(corr),
                "roc_auc": float(auc),
            }
        )
        print(
            f"- seed {seed}: mean_correct={correct_mean:.4f} | mean_incorrect={incorrect_mean:.4f} | "
            f"spearman r={corr:.4f} | roc_auc={auc:.4f}"
        )

    overlap_issues = [o for o in overlaps if o["train_test_overlap"] != 0]
    if overlap_issues:
        print("WARNING: train/test drug overlap detected in kNN splits")
        for issue in overlap_issues:
            print(f"- seed {issue['seed']} fold {issue['fold']}: overlap {issue['train_test_overlap']}")
    else:
        print("Train/test drug overlap check: 0 for all folds")

    summary = compute_abstention_ood_curve(ood, method_meta=method_meta, write_outputs=True)

    # Sanity check at 0% abstention against calibration metrics
    calibration_path = Path(args.calibration_metrics)
    if calibration_path.exists() and "0.0" in summary["summary"]:
        with calibration_path.open() as f:
            cal = json.load(f)
        default_selected = cal.get("default_selected", {}).get("summary", {})
        expected = default_selected.get("balanced_accuracy", {}).get("mean")
        if expected is not None:
            observed = summary["summary"]["0.0"]["balanced_accuracy_mean"]
            diff = observed - expected
            print(
                f"Sanity check @0% abstention: observed {observed:.4f} vs "
                f"expected {expected:.4f} (diff {diff:+.4f})"
            )
        else:
            print("Sanity check: no default_selected balanced_accuracy found in calibration_metrics.json")
    else:
        print("Sanity check skipped: calibration_metrics.json missing or 0.0 rate not present")

    print("Abstention summary (selected rates)")
    for rate_str in ["0.0", "0.1", "0.3", "0.5"]:
        if rate_str not in summary["summary"]:
            continue
        row = summary["summary"][rate_str]
        print(
            f"- rate {rate_str}: coverage {row['coverage_mean']:.3f} | "
            f"bal_acc {row['balanced_accuracy_mean']:.4f} | macro_f1 {row['macro_f1_mean']:.4f}"
        )

    # Compare to confidence-based abstention
    confidence_path = Path("results/abstention_confidence_summary.json")
    confidence_compare = {}
    if confidence_path.exists():
        with confidence_path.open() as f:
            conf = json.load(f)
        for rate_str, coverage in [("0.1", "0.90"), ("0.3", "0.70"), ("0.5", "0.50")]:
            conf_bal = conf.get("summary", {}).get(rate_str, {}).get("balanced_accuracy_mean")
            ood_bal = summary["summary"].get(rate_str, {}).get("balanced_accuracy_mean")
            confidence_compare[coverage] = {
                "confidence_bal_acc": conf_bal,
                "ood_bal_acc": ood_bal,
            }
        print("\nOOD vs confidence (balanced accuracy)")
        for cov, vals in confidence_compare.items():
            print(
                f"- coverage {cov}: confidence {vals['confidence_bal_acc']:.4f} | "
                f"ood {vals['ood_bal_acc']:.4f}"
            )

    # Update summary JSON with diagnostics + k sweep + comparison
    summary["diagnostics"] = {
        "direction": "higher=more_ood (abstain highest)",
        "per_seed": diagnostics,
    }
    summary["k_sweep"] = k_sweep
    summary["best_k_sweep"] = best_k
    summary["selected_k"] = selected_k
    summary["selected_method"] = args.method
    summary["score_definition"] = {
        "raw_score": raw_score_definition,
        "final_ood_score": final_score_definition,
        "flipped": bool(method_meta.get("flipped", False)),
        "flip_reason": flip_reason,
    }
    summary["drop_policy"] = "abstain highest ood_score first (higher=more_ood)"
    summary["comparison_vs_confidence"] = confidence_compare
    summary["method"] = {**method_meta, "k": selected_k}

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Outputs written:")
    print(f"- {OOD_SCORES_PATH}")
    print(f"- {CURVE_PATH}")
    print(f"- {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
