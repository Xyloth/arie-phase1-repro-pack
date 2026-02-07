"""Baseline ML model for CiPA Blinova 2018 dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from arie.data import load_processed_dataset

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "baseline_metrics.json"
FIGURE_PATH = ROOT / "figures" / "baseline_confusion_matrix.png"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_pipeline(random_state: int) -> Pipeline:
    categorical_features = ["cell_type", "platform", "ead_type", "site"]
    numeric_features = ["concentration_level", "ead", "dd_fpdc"]

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_features),
            ("num", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def _stratified_group_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seed: int,
    n_splits: int,
) -> tuple[list[int], list[int], int, set[str]]:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    all_classes = set(y.unique())
    first_split = None
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        if first_split is None:
            first_split = (train_idx, test_idx, fold_idx)
        test_classes = set(y.iloc[test_idx].unique())
        if all_classes.issubset(test_classes):
            return list(train_idx), list(test_idx), fold_idx, set()

    assert first_split is not None
    train_idx, test_idx, fold_idx = first_split
    missing = all_classes.difference(set(y.iloc[test_idx].unique()))
    return list(train_idx), list(test_idx), fold_idx, missing


def train_evaluate(
    seeds: list[int],
    processed_path: Path | None = None,
    results_path: Path | None = None,
    figure_path: Path | None = None,
    n_splits: int = 5,
    plot_seed: int | None = None,
) -> dict:
    processed_path = processed_path or (ROOT / "data" / "processed" / "cipa_blinova_2018.csv")
    df = load_processed_dataset(processed_path)
    df = df.dropna(subset=["risk_class"]).copy()

    target_col = "risk_class"
    group_col = "drug_name"

    feature_cols = [
        "cell_type",
        "platform",
        "ead_type",
        "concentration_level",
        "ead",
        "dd_fpdc",
        "site",
    ]

    X = df[feature_cols]
    y = df[target_col]
    groups = df[group_col]

    dataset_info = {
        "path": str(processed_path),
        "sha256": _sha256_file(processed_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
    }

    runs = []
    plot_seed = plot_seed if plot_seed is not None else seeds[0]
    plot_data = None

    for seed in seeds:
        np.random.seed(seed)
        train_idx, test_idx, fold_idx, missing_classes = _stratified_group_split(
            X, y, groups, seed=seed, n_splits=n_splits
        )

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        train_drugs = set(groups.iloc[train_idx])
        test_drugs = set(groups.iloc[test_idx])
        drug_overlap = train_drugs.intersection(test_drugs)

        pipeline = _build_pipeline(random_state=seed)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        run = {
            "seed": int(seed),
            "fold_index": int(fold_idx),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_drugs_train": int(groups.iloc[train_idx].nunique()),
            "n_drugs_test": int(groups.iloc[test_idx].nunique()),
            "drug_overlap_count": int(len(drug_overlap)),
            "train_class_counts": y_train.value_counts().to_dict(),
            "test_class_counts": y_test.value_counts().to_dict(),
            "missing_classes_in_test": sorted(missing_classes),
            "balanced_accuracy": float(balanced_acc),
        }
        runs.append(run)

        if seed == plot_seed:
            plot_data = (y_test, y_pred, seed)

    balanced_scores = [run["balanced_accuracy"] for run in runs]
    results = {
        "dataset": dataset_info,
        "target": target_col,
        "metric": "balanced_accuracy",
        "split_method": f"StratifiedGroupKFold(n_splits={n_splits}, shuffle=True)",
        "seeds": seeds,
        "runs": runs,
        "balanced_accuracy_mean": float(np.mean(balanced_scores)),
        "balanced_accuracy_std": float(np.std(balanced_scores, ddof=1)) if len(balanced_scores) > 1 else 0.0,
    }

    results_path = results_path or RESULTS_PATH
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    if plot_data is not None:
        figure_path = figure_path or FIGURE_PATH
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        y_test, y_pred, seed = plot_data
        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            normalize="true",
            values_format=".2f",
            cmap="Blues",
        )
        display.ax_.set_title(f"Baseline Random Forest (Seed {seed})")
        display.figure_.tight_layout()
        display.figure_.savefig(figure_path, dpi=150)

    return results
