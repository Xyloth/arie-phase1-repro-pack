#!/usr/bin/env python3
"""Train a mechanistic-only baseline model for CiPA risk_class."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from arie.data import load_processed_dataset
from arie.datasets import DATASET_ID
from arie.mechanistic_features import (
    NUMERIC_FEATURE_COLS,
    build_mechanistic_feature_table,
    load_mechanistic_feature_table,
    summarize_mechanistic_feature_join,
)
from arie.names import normalize_compound

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "mechanistic_baseline_metrics.json"
FIGURE_PATH = ROOT / "figures" / "mechanistic_baseline_confusion_matrix.png"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mechanistic-only baseline model.")
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
        "--force-features",
        action="store_true",
        help="Rebuild mechanistic features even if present.",
    )
    return parser.parse_args()


def _counts(series: pd.Series, classes: list[str]) -> dict:
    return series.value_counts().reindex(classes, fill_value=0).to_dict()


def _resolve_drug_label(counts: pd.Series) -> str:
    if counts.empty:
        return ""
    max_count = counts.max()
    top = sorted([cls for cls, val in counts.items() if val == max_count])
    if len(top) == 1:
        return top[0]
    risk_order = {"L": 0, "M": 1, "H": 2}
    return sorted(top, key=lambda x: risk_order.get(x, -1))[-1]


def main() -> None:
    args = _parse_args()
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

    features_path, features_skipped = build_mechanistic_feature_table(force=args.force_features)
    if features_skipped:
        print(f"Found existing mechanistic feature table at {features_path}.")
    else:
        print(f"Wrote mechanistic feature table to {features_path}.")

    join_summary = summarize_mechanistic_feature_join()
    print("\nJoin summary")
    print(f"- CiPA rows: {join_summary['cipa_rows']}")
    print(f"- CiPA unique parent drugs: {join_summary['cipa_unique_drugs_parent']}")
    print(f"- Mechanistic parent drugs: {join_summary['mechanistic_unique_drugs_parent']}")
    print(f"- Matched parent drugs: {join_summary['matched_drugs_parent']}")
    print(f"- Missing rate (IC50 mean): {join_summary['missing_rate_ic50_mean']:.3f}")
    print(f"- Missing rate (nH mean): {join_summary['missing_rate_nh_mean']:.3f}")

    cipa = load_processed_dataset().dropna(subset=["risk_class"]).copy()
    cipa["drug_name_parent"] = cipa["drug_name"].apply(
        lambda x: normalize_compound(x)["drug_name_parent"]
    )

    drug_rows = []
    conflicts = {}
    for parent, group in cipa.groupby("drug_name_parent"):
        counts = group["risk_class"].value_counts()
        if len(counts) > 1:
            conflicts[parent] = counts.to_dict()
        label = _resolve_drug_label(counts)
        drug_rows.append(
            {
                "drug_name_parent": parent,
                "risk_class": label,
                "n_rows": int(len(group)),
            }
        )
    drug_df = pd.DataFrame(drug_rows).sort_values("drug_name_parent").reset_index(drop=True)

    if conflicts:
        print("\nRisk class conflicts detected (parent -> counts):")
        for parent, counts in conflicts.items():
            print(f"- {parent}: {counts}")
    else:
        print("\nRisk class is consistent across parent drugs.")

    class_counts = drug_df["risk_class"].value_counts().to_dict()
    print(f"Drug-level class counts: {class_counts}")

    features = load_mechanistic_feature_table()
    merged = drug_df.merge(features, on="drug_name_parent", how="left")

    feature_cols = NUMERIC_FEATURE_COLS
    X = merged[feature_cols]
    y = merged["risk_class"]
    groups = merged["drug_name_parent"]

    classes = sorted(y.unique())

    dataset_info = {
        "id": DATASET_ID,
        "cipa_path": str(ROOT / "data" / "processed" / f"{DATASET_ID}.csv"),
        "mechanistic_features_path": str(features_path),
        "cipa_sha256": _sha256_file(ROOT / "data" / "processed" / f"{DATASET_ID}.csv"),
        "mechanistic_features_sha256": _sha256_file(features_path),
        "rows": int(len(merged)),
        "evaluation_unit": "drug",
        "columns": list(merged.columns),
    }

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=args.seed,
    )

    pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )

    fold_runs = []
    plot_seed = seeds[0]
    plot_true = []
    plot_pred = []

    for seed in seeds:
        np.random.seed(seed)
        splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])
            overlap = train_groups.intersection(test_groups)

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            bal_acc = balanced_accuracy_score(y_test, y_pred)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="y_pred contains classes not in y_true"
                )
                macro_f1 = f1_score(
                    y_test,
                    y_pred,
                    average="macro",
                    labels=classes,
                    zero_division=0,
                )

            fold_runs.append(
                {
                    "seed": int(seed),
                    "fold_index": int(fold_idx),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_drugs_train": int(groups.iloc[train_idx].nunique()),
                    "n_drugs_test": int(groups.iloc[test_idx].nunique()),
                    "drug_overlap_count": int(len(overlap)),
                    "train_class_counts": _counts(y_train, classes),
                    "test_class_counts": _counts(y_test, classes),
                    "balanced_accuracy": float(bal_acc),
                    "macro_f1": float(macro_f1),
                }
            )

            if seed == plot_seed:
                plot_true.extend(y_test.tolist())
                plot_pred.extend(y_pred.tolist())

    bal_scores = [run["balanced_accuracy"] for run in fold_runs]
    f1_scores = [run["macro_f1"] for run in fold_runs]

    results = {
        "dataset": dataset_info,
        "target": "risk_class",
        "split_method": f"StratifiedGroupKFold(n_splits={args.n_splits}, shuffle=True)",
        "group_column": "drug_name_parent",
        "seeds": seeds,
        "feature_columns": feature_cols,
        "runs": fold_runs,
        "drug_label_conflicts": conflicts,
        "drug_label_resolution": "majority vote; ties resolved by risk order L<M<H",
        "balanced_accuracy_mean": float(np.mean(bal_scores)),
        "balanced_accuracy_std": float(np.std(bal_scores, ddof=1)) if len(bal_scores) > 1 else 0.0,
        "macro_f1_mean": float(np.mean(f1_scores)),
        "macro_f1_std": float(np.std(f1_scores, ddof=1)) if len(f1_scores) > 1 else 0.0,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    if plot_true and plot_pred:
        FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        display = ConfusionMatrixDisplay.from_predictions(
            plot_true,
            plot_pred,
            normalize="true",
            values_format=".2f",
            cmap="Blues",
        )
        display.ax_.set_title(f"Mechanistic Baseline (Seed {plot_seed})")
        display.figure_.tight_layout()
        display.figure_.savefig(FIGURE_PATH, dpi=150)

    print("\nMechanistic baseline results")
    print(
        f"- balanced accuracy: {results['balanced_accuracy_mean']:.4f} "
        f"(std {results['balanced_accuracy_std']:.4f})"
    )
    print(
        f"- macro F1: {results['macro_f1_mean']:.4f} "
        f"(std {results['macro_f1_std']:.4f})"
    )
    print(f"Metrics saved to {RESULTS_PATH}")
    if plot_true and plot_pred:
        print(f"Confusion matrix saved to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
