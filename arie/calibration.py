"""Calibrated classifier evaluation for CiPA Blinova 2018."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from arie.data import load_processed_dataset
from arie.datasets import DATASET_ID

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "calibration_metrics.json"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_pipeline(model, *, dense_output: bool) -> Pipeline:
    categorical_features = ["cell_type", "platform", "ead_type", "site"]
    numeric_features = ["concentration_level", "ead", "dd_fpdc"]

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output)),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_features),
            ("num", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def _expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: np.ndarray,
    n_bins: int = 10,
) -> float:
    confidences = probs.max(axis=1)
    pred_indices = probs.argmax(axis=1)
    preds = classes[pred_indices]
    correct = (preds == y_true)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if not np.any(mask):
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * (mask.sum() / len(y_true))
    return float(ece)


def _brier_multiclass(y_true: np.ndarray, probs: np.ndarray, classes: np.ndarray) -> float:
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    indices = np.array([class_to_index[cls] for cls in y_true])
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(indices)), indices] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def _build_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seeds: List[int],
    n_splits: int,
    calibration_n_splits: int,
    calibration_seed_offset: int,
) -> Tuple[Dict[int, List[dict]], Dict[int, List[dict]]]:
    all_classes = sorted(y.unique())
    split_map: Dict[int, List[dict]] = {}
    diagnostics: Dict[int, List[dict]] = {}

    for seed in seeds:
        outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds: List[dict] = []
        diag_folds: List[dict] = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X, y, groups)):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            groups_train = groups.iloc[train_idx]

            inner_seed = seed + calibration_seed_offset + fold_idx
            inner = StratifiedGroupKFold(
                n_splits=calibration_n_splits,
                shuffle=True,
                random_state=inner_seed,
            )

            chosen = None
            for calib_fold_idx, (train_sub_rel, calib_rel) in enumerate(inner.split(X_train, y_train, groups_train)):
                calib_classes = set(y_train.iloc[calib_rel].unique())
                missing_calib = sorted(set(all_classes).difference(calib_classes))
                if not missing_calib:
                    chosen = (train_sub_rel, calib_rel, calib_fold_idx, missing_calib)
                    break
                if chosen is None:
                    chosen = (train_sub_rel, calib_rel, calib_fold_idx, missing_calib)

            assert chosen is not None
            train_sub_rel, calib_rel, calib_fold_idx, missing_calib = chosen

            train_sub_idx = train_idx[train_sub_rel]
            calib_idx = train_idx[calib_rel]

            train_drugs = set(groups.iloc[train_sub_idx])
            calib_drugs = set(groups.iloc[calib_idx])
            test_drugs = set(groups.iloc[test_idx])

            def _counts(series: pd.Series) -> dict:
                return series.value_counts().reindex(all_classes, fill_value=0).to_dict()

            fold_info = {
                "seed": seed,
                "fold_index": fold_idx,
                "calibration_fold_index": calib_fold_idx,
                "train_idx": train_sub_idx,
                "calib_idx": calib_idx,
                "test_idx": test_idx,
                "n_train": int(len(train_sub_idx)),
                "n_calib": int(len(calib_idx)),
                "n_test": int(len(test_idx)),
                "n_drugs_train": int(groups.iloc[train_sub_idx].nunique()),
                "n_drugs_calib": int(groups.iloc[calib_idx].nunique()),
                "n_drugs_test": int(groups.iloc[test_idx].nunique()),
                "train_test_overlap": int(len(train_drugs.intersection(test_drugs))),
                "train_calib_overlap": int(len(train_drugs.intersection(calib_drugs))),
                "calib_test_overlap": int(len(calib_drugs.intersection(test_drugs))),
                "train_class_counts": _counts(y.iloc[train_sub_idx]),
                "calib_class_counts": _counts(y.iloc[calib_idx]),
                "test_class_counts": _counts(y.iloc[test_idx]),
                "missing_classes_in_test": sorted(
                    set(all_classes).difference(set(y.iloc[test_idx].unique()))
                ),
                "missing_classes_in_calib": missing_calib,
            }
            folds.append(fold_info)

            diag_folds.append(
                {
                    "seed": seed,
                    "fold_index": fold_idx,
                    "calibration_fold_index": calib_fold_idx,
                    "n_train": fold_info["n_train"],
                    "n_calib": fold_info["n_calib"],
                    "n_test": fold_info["n_test"],
                    "n_drugs_train": fold_info["n_drugs_train"],
                    "n_drugs_calib": fold_info["n_drugs_calib"],
                    "n_drugs_test": fold_info["n_drugs_test"],
                    "train_test_overlap": fold_info["train_test_overlap"],
                    "train_calib_overlap": fold_info["train_calib_overlap"],
                    "calib_test_overlap": fold_info["calib_test_overlap"],
                    "train_class_counts": fold_info["train_class_counts"],
                    "calib_class_counts": fold_info["calib_class_counts"],
                    "test_class_counts": fold_info["test_class_counts"],
                    "missing_classes_in_test": fold_info["missing_classes_in_test"],
                    "missing_classes_in_calib": fold_info["missing_classes_in_calib"],
                }
            )
        split_map[seed] = folds
        diagnostics[seed] = diag_folds

    return split_map, diagnostics


def evaluate_calibration_grid(
    seeds: List[int],
    processed_path: Path | None = None,
    results_path: Path | None = None,
    n_splits: int = 5,
    calibration_n_splits: int = 3,
    calibration_methods: List[str] | None = None,
    min_calib_samples_per_class: int = 50,
    calibration_seed_offset: int = 1000,
) -> dict:
    processed_path = processed_path or (ROOT / "data" / "processed" / f"{DATASET_ID}.csv")
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
        "id": DATASET_ID,
        "path": str(processed_path),
        "sha256": _sha256_file(processed_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
    }

    calibration_methods = calibration_methods or ["none", "sigmoid", "isotonic"]

    split_map, split_diagnostics = _build_splits(
        X,
        y,
        groups,
        seeds=seeds,
        n_splits=n_splits,
        calibration_n_splits=calibration_n_splits,
        calibration_seed_offset=calibration_seed_offset,
    )

    min_calib_count = min(
        min(fold["calib_class_counts"].values())
        for folds in split_diagnostics.values()
        for fold in folds
    )

    missing_calib_folds = sum(
        1 for folds in split_diagnostics.values() for fold in folds if fold["missing_classes_in_calib"]
    )
    isotonic_allowed = (min_calib_count >= min_calib_samples_per_class) and (missing_calib_folds == 0)

    model_builders = {
        "log_reg": {
            "builder": lambda seed: LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="saga",
                tol=1e-3,
                n_jobs=1,
                random_state=seed,
            ),
            "dense_output": False,
        },
        "hist_gb": {
            "builder": lambda seed: HistGradientBoostingClassifier(
                loss="log_loss",
                max_iter=200,
                learning_rate=0.1,
                max_depth=6,
                class_weight="balanced",
                random_state=seed,
            ),
            "dense_output": True,
        },
    }

    combos = []
    for model_name in model_builders:
        for method in calibration_methods:
            combos.append((model_name, method))

    combo_results: Dict[str, dict] = {}
    for model_name, method in combos:
        key = f"{model_name}__{method}"
        status = "OK"
        reason = None
        if method == "isotonic" and not isotonic_allowed:
            status = "SKIP"
            reason = (
                f"min_calib_samples_per_class_observed={min_calib_count} "
                f"< {min_calib_samples_per_class}; "
                f"missing_calib_folds={missing_calib_folds}"
            )
        combo_results[key] = {
            "model": model_name,
            "calibration": method,
            "status": status,
            "reason": reason,
            "per_seed": [],
            "summary": {},
        }

    for model_name, method in combos:
        key = f"{model_name}__{method}"
        if combo_results[key]["status"] != "OK":
            continue

        try:
            for seed in seeds:
                fold_metrics = []
                for fold in split_map[seed]:
                    np.random.seed(seed)

                    if method == "isotonic" and min(fold["calib_class_counts"].values()) < min_calib_samples_per_class:
                        raise ValueError(
                            "Calibration set too small for isotonic regression; "
                            "increase calibration set size or lower threshold."
                        )

                    train_idx = fold["train_idx"]
                    calib_idx = fold["calib_idx"]
                    test_idx = fold["test_idx"]

                    X_train = X.iloc[train_idx]
                    y_train = y.iloc[train_idx]

                    X_calib = X.iloc[calib_idx]
                    y_calib = y.iloc[calib_idx]

                    X_test = X.iloc[test_idx]
                    y_test = y.iloc[test_idx]

                    model_cfg = model_builders[model_name]
                    model = model_cfg["builder"](seed)
                    base_model = _build_pipeline(model, dense_output=model_cfg["dense_output"])
                    base_model.fit(X_train, y_train)

                    if method == "none":
                        probs = base_model.predict_proba(X_test)
                        y_pred = base_model.predict(X_test)
                        classes = base_model.named_steps["model"].classes_
                    else:
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message="The `cv='prefit'` option is deprecated",
                                category=FutureWarning,
                            )
                            calibrator = CalibratedClassifierCV(
                                estimator=base_model,
                                method=method,
                                cv="prefit",
                            )
                            calibrator.fit(X_calib, y_calib)

                        probs = calibrator.predict_proba(X_test)
                        y_pred = calibrator.predict(X_test)
                        classes = calibrator.classes_

                    metrics = {
                        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                        "ece": _expected_calibration_error(y_test.to_numpy(), probs, classes),
                        "brier": _brier_multiclass(y_test.to_numpy(), probs, classes),
                        "log_loss": float(log_loss(y_test, probs, labels=classes)),
                    }

                    fold_metrics.append(
                        {
                            "seed": int(seed),
                            "fold_index": fold["fold_index"],
                            **metrics,
                        }
                    )

                metric_keys = ["balanced_accuracy", "ece", "brier", "log_loss"]
                seed_summary = {}
                for key_metric in metric_keys:
                    values = [m[key_metric] for m in fold_metrics]
                    seed_summary[key_metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    }

                combo_results[key]["per_seed"].append(
                    {
                        "seed": int(seed),
                        "fold_metrics": fold_metrics,
                        "summary": seed_summary,
                    }
                )

            # Overall summary across seeds
            metric_keys = ["balanced_accuracy", "ece", "brier", "log_loss"]
            overall_summary = {}
            for key_metric in metric_keys:
                values = [
                    seed_entry["summary"][key_metric]["mean"]
                    for seed_entry in combo_results[key]["per_seed"]
                ]
                overall_summary[key_metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }

            combo_results[key]["summary"] = overall_summary
        except Exception as exc:  # pragma: no cover - defensive
            combo_results[key]["status"] = "FAIL"
            combo_results[key]["reason"] = str(exc)
            combo_results[key]["per_seed"] = []
            combo_results[key]["summary"] = {}

    comparison_table = []
    for combo in combo_results.values():
        row = {
            "model": combo["model"],
            "calibration": combo["calibration"],
            "status": combo["status"],
        }
        if combo["status"] == "OK":
            row.update(
                {
                    "balanced_accuracy_mean": combo["summary"]["balanced_accuracy"]["mean"],
                    "balanced_accuracy_std": combo["summary"]["balanced_accuracy"]["std"],
                    "log_loss_mean": combo["summary"]["log_loss"]["mean"],
                    "log_loss_std": combo["summary"]["log_loss"]["std"],
                    "brier_mean": combo["summary"]["brier"]["mean"],
                    "brier_std": combo["summary"]["brier"]["std"],
                    "ece_mean": combo["summary"]["ece"]["mean"],
                    "ece_std": combo["summary"]["ece"]["std"],
                }
            )
        else:
            row["reason"] = combo.get("reason")
        comparison_table.append(row)

    eligible = [row for row in comparison_table if row["status"] == "OK"]
    default_choice = None
    if eligible:
        eligible.sort(
            key=lambda r: (
                -r["balanced_accuracy_mean"],
                r["log_loss_mean"],
            )
        )
        best = eligible[0]
        default_choice = {
            "model": best["model"],
            "calibration": best["calibration"],
            "criteria": "max balanced_accuracy then min log_loss",
        }

        assert any(
            row["model"] == default_choice["model"]
            and row["calibration"] == default_choice["calibration"]
            and row["status"] == "OK"
            for row in comparison_table
        ), "Default choice must be present in comparison table with status OK."

    predictions_path = None
    prediction_columns = None
    default_selected = None
    default_uncalibrated_reference = None

    if default_choice is not None:
        default_key = f"{default_choice['model']}__{default_choice['calibration']}"
        if combo_results.get(default_key, {}).get("status") != "OK":
            raise RuntimeError("Default choice not available for prediction output.")

        default_selected = {
            "per_seed": combo_results[default_key]["per_seed"],
            "summary": combo_results[default_key]["summary"],
            "method": default_choice["calibration"],
        }

        uncalibrated_per_seed = []

        preds = []
        classes_ref = None
        for seed in seeds:
            fold_uncalibrated_metrics = []
            for fold in split_map[seed]:
                np.random.seed(seed)

                train_idx = fold["train_idx"]
                calib_idx = fold["calib_idx"]
                test_idx = fold["test_idx"]

                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]

                X_calib = X.iloc[calib_idx]
                y_calib = y.iloc[calib_idx]

                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]

                model_cfg = model_builders[default_choice["model"]]
                model = model_cfg["builder"](seed)
                base_model = _build_pipeline(model, dense_output=model_cfg["dense_output"])
                base_model.fit(X_train, y_train)

                base_probs = base_model.predict_proba(X_test)
                base_pred = base_model.predict(X_test)
                base_classes = list(base_model.named_steps["model"].classes_)

                if default_choice["calibration"] == "none":
                    probs = base_probs
                    y_pred = base_pred
                    classes = list(base_classes)
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="The `cv='prefit'` option is deprecated",
                            category=FutureWarning,
                        )
                        calibrator = CalibratedClassifierCV(
                            estimator=base_model,
                            method=default_choice["calibration"],
                            cv="prefit",
                        )
                        calibrator.fit(X_calib, y_calib)

                    probs = calibrator.predict_proba(X_test)
                    y_pred = calibrator.predict(X_test)
                    classes = list(calibrator.classes_)

                if classes_ref is None:
                    classes_ref = classes
                elif classes != classes_ref:
                    raise RuntimeError("Class order mismatch across folds; cannot write predictions.")

                if base_classes != classes_ref:
                    index_map = [base_classes.index(cls) for cls in classes_ref]
                    base_probs = base_probs[:, index_map]

                fold_uncalibrated_metrics.append(
                    {
                        "seed": int(seed),
                        "fold_index": fold["fold_index"],
                        "balanced_accuracy": float(balanced_accuracy_score(y_test, base_pred)),
                        "ece": _expected_calibration_error(y_test.to_numpy(), base_probs, np.array(classes_ref)),
                        "brier": _brier_multiclass(y_test.to_numpy(), base_probs, np.array(classes_ref)),
                        "log_loss": float(log_loss(y_test, base_probs, labels=classes_ref)),
                    }
                )

                subset = df.iloc[test_idx]
                rows = subset[
                    ["drug_name", "cell_type", "platform", "concentration_level", "site"]
                ].copy()
                rows.insert(0, "row_id", subset.index.to_numpy())
                rows.insert(1, "seed", seed)
                rows.insert(2, "fold", fold["fold_index"])
                rows.insert(3, "split_role", "test")
                rows["y_true"] = y_test.to_numpy()
                rows["y_pred"] = y_pred
                rows["confidence"] = probs.max(axis=1)

                for col_idx, cls in enumerate(classes_ref):
                    rows[f"prob_{cls}"] = probs[:, col_idx]

                preds.append(rows)

            metric_keys = ["balanced_accuracy", "ece", "brier", "log_loss"]
            seed_summary = {}
            for key_metric in metric_keys:
                values = [m[key_metric] for m in fold_uncalibrated_metrics]
                seed_summary[key_metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }

            uncalibrated_per_seed.append(
                {
                    "seed": int(seed),
                    "fold_metrics": fold_uncalibrated_metrics,
                    "summary": seed_summary,
                }
            )

        pred_df = pd.concat(preds, ignore_index=True)
        pred_df = pred_df.sort_values(["seed", "fold", "row_id"]).reset_index(drop=True)

        predictions_path = ROOT / "results" / "calibration_predictions.csv"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(predictions_path, index=False)
        prediction_columns = list(pred_df.columns)

        metric_keys = ["balanced_accuracy", "ece", "brier", "log_loss"]
        overall_summary = {}
        for key_metric in metric_keys:
            values = [seed_entry["summary"][key_metric]["mean"] for seed_entry in uncalibrated_per_seed]
            overall_summary[key_metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }

        default_uncalibrated_reference = {
            "per_seed": uncalibrated_per_seed,
            "summary": overall_summary,
        }

    results = {
        "dataset": dataset_info,
        "target": target_col,
        "calibration_methods": calibration_methods,
        "min_calib_samples_per_class": min_calib_samples_per_class,
        "min_calib_class_count_observed": int(min_calib_count),
        "missing_calib_folds": int(missing_calib_folds),
        "split_method": (
            f"outer=StratifiedGroupKFold(n_splits={n_splits}, shuffle=True); "
            f"inner=StratifiedGroupKFold(n_splits={calibration_n_splits}, shuffle=True)"
        ),
        "seeds": seeds,
        "split_diagnostics": split_diagnostics,
        "comparison_table": comparison_table,
        "default_choice": default_choice,
        "default_selected": default_selected,
        "default_uncalibrated_reference": default_uncalibrated_reference,
        "predictions_path": str(predictions_path) if predictions_path else None,
        "prediction_columns": prediction_columns,
        "combos": combo_results,
    }

    results_path = results_path or RESULTS_PATH
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    return results
