"""Mechanistic + ML fusion model training and evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from arie.data import load_processed_dataset
from arie.datasets import DATASET_ID
from arie.mechanistic_features import (
    FEATURES_PATH as MECH_FEATURES_PATH,
    NUMERIC_FEATURE_COLS as MECH_NUMERIC_COLS,
    PROVENANCE_COUNT_COLS as MECH_PROVENANCE_COLS,
    load_mechanistic_feature_table,
)
from arie.names import normalize_compound

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "fusion_metrics.json"
JOIN_SUMMARY_PATH = ROOT / "results" / "fusion_join_summary.json"
PREDICTIONS_PATH = ROOT / "results" / "fusion_predictions.csv"
FIGURE_PATH = ROOT / "figures" / "fusion_confusion_matrix.png"

ML_CATEGORICAL_COLS = ["cell_type", "platform", "ead_type", "site"]
ML_NUMERIC_COLS = ["concentration_level", "ead", "dd_fpdc"]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_pipeline(
    categorical_features: List[str],
    numeric_features: List[str],
    seed: int,
) -> Pipeline:
    transformers = []
    if categorical_features:
        cat_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_features))

    if numeric_features:
        num_pipeline = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])
        transformers.append(("num", num_pipeline, numeric_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    model = HistGradientBoostingClassifier(
        random_state=seed,
        max_iter=200,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=30,
        l2_regularization=0.1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def _compute_preprocessed_shape(
    data: pd.DataFrame,
    categorical_features: List[str],
    numeric_features: List[str],
) -> Tuple[int, int]:
    X = data[categorical_features + numeric_features]
    pipeline = _build_pipeline(categorical_features, numeric_features, seed=0)
    preprocessor = pipeline.named_steps["preprocess"]
    transformed = preprocessor.fit_transform(X)
    return transformed.shape[0], transformed.shape[1]


def _align_proba(
    prob: np.ndarray,
    model_classes: np.ndarray,
    class_order: List[str],
) -> np.ndarray:
    aligned = np.zeros((prob.shape[0], len(class_order)), dtype=float)
    class_index = {cls: idx for idx, cls in enumerate(model_classes)}
    for j, cls in enumerate(class_order):
        if cls in class_index:
            aligned[:, j] = prob[:, class_index[cls]]
    return aligned


def _temperature_scale(proba: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return proba
    scaled = np.power(proba, 1.0 / temperature)
    scaled_sum = scaled.sum(axis=1, keepdims=True)
    scaled = np.divide(scaled, scaled_sum, out=np.zeros_like(scaled), where=scaled_sum > 0)
    zero_rows = scaled_sum.squeeze() == 0
    if np.any(zero_rows):
        scaled[zero_rows] = 1.0 / scaled.shape[1]
    return scaled


def _normalize_probs(proba: np.ndarray) -> np.ndarray:
    sums = proba.sum(axis=1, keepdims=True)
    normalized = np.divide(proba, sums, out=np.zeros_like(proba), where=sums > 0)
    zero_rows = sums.squeeze() == 0
    if np.any(zero_rows):
        normalized[zero_rows] = 1.0 / normalized.shape[1]
    return normalized


def _fit_temperature(
    proba: np.ndarray,
    y_true: np.ndarray,
    class_order: List[str],
    grid: List[float],
) -> float:
    best_t = 1.0
    best_loss = float("inf")
    for temp in grid:
        scaled = _temperature_scale(proba, temp)
        loss = float(log_loss(y_true, scaled, labels=class_order))
        if loss < best_loss:
            best_loss = loss
            best_t = temp
    return best_t


def _expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: List[str],
    n_bins: int = 10,
) -> float:
    confidences = probs.max(axis=1)
    pred_indices = probs.argmax(axis=1)
    preds = np.array([classes[idx] for idx in pred_indices])
    correct = preds == y_true

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


def _brier_multiclass(y_true: np.ndarray, probs: np.ndarray, classes: List[str]) -> float:
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    indices = np.array([class_to_index[cls] for cls in y_true])
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(indices)), indices] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def _build_outer_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seeds: List[int],
    n_splits: int,
) -> Dict[int, List[dict]]:
    split_map: Dict[int, List[dict]] = {}
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
            folds.append(
                {
                    "fold_index": fold_idx,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )
        split_map[seed] = folds
    return split_map


def _select_calibration_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    seed: int,
    fold_idx: int,
    n_splits: int = 3,
    seed_offset: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    all_classes = sorted(y_train.unique())
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed + seed_offset + fold_idx,
    )
    chosen = None
    for train_sub_idx, calib_idx in splitter.split(X_train, y_train, groups_train):
        calib_classes = set(y_train.iloc[calib_idx].unique())
        missing = sorted(set(all_classes).difference(calib_classes))
        if not missing:
            chosen = (train_sub_idx, calib_idx, missing)
            break
        if chosen is None:
            chosen = (train_sub_idx, calib_idx, missing)
    if chosen is None:
        raise RuntimeError("Failed to generate calibration split.")
    return chosen[0], chosen[1], chosen[2]


def build_joined_dataset(
    mechanistic_features_path: Path | None = None,
) -> Tuple[pd.DataFrame, dict]:
    mechanistic_features_path = mechanistic_features_path or MECH_FEATURES_PATH
    cipa = load_processed_dataset().dropna(subset=["risk_class"]).copy()
    cipa = cipa.reset_index(drop=True)
    cipa["row_id"] = cipa.index.astype(int)
    cipa["drug_name_parent"] = cipa["drug_name"].apply(
        lambda x: normalize_compound(x)["drug_name_parent"]
    )

    mech = load_mechanistic_feature_table(mechanistic_features_path)
    merged = cipa.merge(mech, on="drug_name_parent", how="left")

    cipa_parents = sorted(cipa["drug_name_parent"].unique())
    mech_parents = sorted(mech["drug_name_parent"].unique())
    matched = sorted(set(cipa_parents).intersection(set(mech_parents)))
    missing = sorted(set(cipa_parents).difference(set(mech_parents)))

    missing_rates = {
        col: float(merged[col].isna().mean())
        for col in MECH_NUMERIC_COLS + MECH_PROVENANCE_COLS
        if col in merged.columns
    }

    summary = {
        "dataset_id": DATASET_ID,
        "cipa_rows": int(len(cipa)),
        "cipa_unique_parents": int(len(cipa_parents)),
        "mechanistic_unique_parents": int(len(mech_parents)),
        "matched_parents": int(len(matched)),
        "match_rate": float(len(matched) / len(cipa_parents)) if cipa_parents else 0.0,
        "missing_parents": missing,
        "missing_rates": missing_rates,
        "mechanistic_features_path": str(mechanistic_features_path),
        "mechanistic_features_sha256": _sha256_file(mechanistic_features_path),
    }

    JOIN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOIN_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return merged, summary


def _metric_summary(values: List[float]) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def _evaluate_config(
    name: str,
    data: pd.DataFrame,
    categorical_features: List[str],
    numeric_features: List[str],
    seeds: List[int],
    n_splits: int,
    class_order: List[str],
    outer_splits: Dict[int, List[dict]],
    calibrate: bool = False,
    collect_predictions: bool = False,
) -> Tuple[dict, List[dict]]:
    X = data[categorical_features + numeric_features]
    y = data["risk_class"]
    groups = data["drug_name_parent"]
    row_ids = data["row_id"].to_numpy()

    runs = []
    prediction_rows: List[dict] = []

    for seed in seeds:
        seed_true_raw: List[str] = []
        seed_pred_raw: List[str] = []
        seed_proba_raw: List[np.ndarray] = []

        seed_true_cal: List[str] = []
        seed_pred_cal: List[str] = []
        seed_proba_cal: List[np.ndarray] = []

        overlap_counts: List[int] = []
        missing_train: List[List[str]] = []
        missing_test: List[List[str]] = []
        missing_calib: List[List[str]] = []
        calib_skipped: List[bool] = []

        for fold in outer_splits[seed]:
            fold_idx = fold["fold_index"]
            train_idx = fold["train_idx"]
            test_idx = fold["test_idx"]

            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])
            overlap_counts.append(len(train_groups.intersection(test_groups)))

            train_classes = set(y.iloc[train_idx].unique())
            test_classes = set(y.iloc[test_idx].unique())
            missing_train.append(sorted(set(class_order).difference(train_classes)))
            missing_test.append(sorted(set(class_order).difference(test_classes)))

            if calibrate:
                train_sub_rel, calib_rel, missing = _select_calibration_split(
                    X_train=X.iloc[train_idx],
                    y_train=y.iloc[train_idx],
                    groups_train=groups.iloc[train_idx],
                    seed=seed,
                    fold_idx=fold_idx,
                )
                train_fit_idx = train_idx[train_sub_rel]
                calib_idx = train_idx[calib_rel]
                missing_calib.append(missing)
            else:
                train_fit_idx = train_idx
                calib_idx = None
                missing_calib.append([])

            pipeline = _build_pipeline(categorical_features, numeric_features, seed)
            pipeline.fit(X.iloc[train_fit_idx], y.iloc[train_fit_idx])

            y_pred_raw = pipeline.predict(X.iloc[test_idx])
            raw_proba = pipeline.predict_proba(X.iloc[test_idx])
            proba_raw = _align_proba(raw_proba, pipeline.classes_, class_order)
            proba_raw = _normalize_probs(proba_raw)

            if calibrate and calib_idx is not None:
                calib_raw = pipeline.predict_proba(X.iloc[calib_idx])
                calib_raw = _align_proba(calib_raw, pipeline.classes_, class_order)
                calib_raw = _normalize_probs(calib_raw)
                calib_raw = np.clip(calib_raw, 1e-6, 1 - 1e-6)
                temp_grid = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
                temperature = _fit_temperature(
                    calib_raw,
                    y.iloc[calib_idx].to_numpy(),
                    class_order,
                    temp_grid,
                )
                proba_cal = _temperature_scale(proba_raw, temperature)
                proba_cal = _normalize_probs(proba_cal)
                y_pred_cal = np.array([class_order[idx] for idx in np.argmax(proba_cal, axis=1)])
                calib_skipped.append(False)
            else:
                proba_cal = proba_raw
                y_pred_cal = y_pred_raw
                calib_skipped.append(True)

            seed_true_raw.extend(y.iloc[test_idx].tolist())
            seed_pred_raw.extend(y_pred_raw.tolist())
            seed_proba_raw.append(proba_raw)

            seed_true_cal.extend(y.iloc[test_idx].tolist())
            seed_pred_cal.extend(y_pred_cal.tolist())
            seed_proba_cal.append(proba_cal)

            if collect_predictions:
                for idx, row_id, true_label, pred_raw, pred_cal, probs_raw, probs_cal in zip(
                    test_idx,
                    row_ids[test_idx],
                    y.iloc[test_idx],
                    y_pred_raw,
                    y_pred_cal,
                    proba_raw,
                    proba_cal,
                ):
                    record = {
                        "row_id": int(row_id),
                        "seed": int(seed),
                        "fold": int(fold_idx),
                        "split_role": "test",
                        "drug_name_parent": data.iloc[idx]["drug_name_parent"],
                        "y_true": true_label,
                        "y_pred": pred_cal,
                        "y_pred_raw": pred_raw,
                        "y_pred_cal": pred_cal,
                        "confidence": float(np.max(probs_cal)),
                        "confidence_raw": float(np.max(probs_raw)),
                        "confidence_cal": float(np.max(probs_cal)),
                    }
                    for cls, val in zip(class_order, probs_raw):
                        record[f"prob_raw_{cls}"] = float(val)
                    for cls, val in zip(class_order, probs_cal):
                        record[f"prob_cal_{cls}"] = float(val)
                    prediction_rows.append(record)

        y_true_raw = np.array(seed_true_raw)
        y_pred_raw = np.array(seed_pred_raw)
        proba_raw = np.vstack(seed_proba_raw) if seed_proba_raw else None

        y_true_cal = np.array(seed_true_cal)
        y_pred_cal = np.array(seed_pred_cal)
        proba_cal = np.vstack(seed_proba_cal) if seed_proba_cal else None

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="y_pred contains classes not in y_true",
            )
            raw_macro_f1 = float(
                f1_score(
                    y_true_raw,
                    y_pred_raw,
                    average="macro",
                    labels=class_order,
                    zero_division=0,
                )
            )
            cal_macro_f1 = float(
                f1_score(
                    y_true_cal,
                    y_pred_cal,
                    average="macro",
                    labels=class_order,
                    zero_division=0,
                )
            )

        run = {
            "seed": int(seed),
            "folds": int(n_splits),
            "overlap_counts": overlap_counts,
            "missing_classes_in_train": missing_train,
            "missing_classes_in_test": missing_test,
            "missing_classes_in_calib": missing_calib,
            "calibration_skipped": calib_skipped,
            "raw": {
                "balanced_accuracy": float(balanced_accuracy_score(y_true_raw, y_pred_raw)),
                "macro_f1": raw_macro_f1,
                "log_loss": float(log_loss(y_true_raw, proba_raw, labels=class_order)),
                "brier": _brier_multiclass(y_true_raw, proba_raw, class_order),
                "ece": _expected_calibration_error(y_true_raw, proba_raw, class_order),
            },
            "calibrated": {
                "balanced_accuracy": float(balanced_accuracy_score(y_true_cal, y_pred_cal)),
                "macro_f1": cal_macro_f1,
                "log_loss": float(log_loss(y_true_cal, proba_cal, labels=class_order)),
                "brier": _brier_multiclass(y_true_cal, proba_cal, class_order),
                "ece": _expected_calibration_error(y_true_cal, proba_cal, class_order),
            },
        }
        runs.append(run)

    raw_bal = [r["raw"]["balanced_accuracy"] for r in runs]
    raw_f1 = [r["raw"]["macro_f1"] for r in runs]
    raw_log = [r["raw"]["log_loss"] for r in runs]
    raw_brier = [r["raw"]["brier"] for r in runs]
    raw_ece = [r["raw"]["ece"] for r in runs]

    cal_bal = [r["calibrated"]["balanced_accuracy"] for r in runs]
    cal_f1 = [r["calibrated"]["macro_f1"] for r in runs]
    cal_log = [r["calibrated"]["log_loss"] for r in runs]
    cal_brier = [r["calibrated"]["brier"] for r in runs]
    cal_ece = [r["calibrated"]["ece"] for r in runs]

    pre_shape = _compute_preprocessed_shape(
        data=data,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )

    summary = {
        "name": name,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "raw_input_columns": categorical_features + numeric_features,
        "n_categorical": len(categorical_features),
        "n_numeric": len(numeric_features),
        "preprocessed_shape": {"n_rows": int(pre_shape[0]), "n_features": int(pre_shape[1])},
        "model_type": "HistGradientBoostingClassifier",
        "classes": class_order,
        "calibration": {
            "enabled": calibrate,
            "method": "temperature_scaling" if calibrate else None,
            "calib_splits": 3 if calibrate else None,
        },
        "runs": runs,
        "metrics": {
            "raw": {
                "balanced_accuracy": _metric_summary(raw_bal),
                "macro_f1": _metric_summary(raw_f1),
                "log_loss": _metric_summary(raw_log),
                "brier": _metric_summary(raw_brier),
                "ece": _metric_summary(raw_ece),
            },
            "calibrated": {
                "balanced_accuracy": _metric_summary(cal_bal),
                "macro_f1": _metric_summary(cal_f1),
                "log_loss": _metric_summary(cal_log),
                "brier": _metric_summary(cal_brier),
                "ece": _metric_summary(cal_ece),
            },
        },
        "balanced_accuracy_mean": _metric_summary(cal_bal)["mean"] if calibrate else _metric_summary(raw_bal)["mean"],
        "balanced_accuracy_std": _metric_summary(cal_bal)["std"] if calibrate else _metric_summary(raw_bal)["std"],
        "macro_f1_mean": _metric_summary(cal_f1)["mean"] if calibrate else _metric_summary(raw_f1)["mean"],
        "macro_f1_std": _metric_summary(cal_f1)["std"] if calibrate else _metric_summary(raw_f1)["std"],
        "log_loss_mean": _metric_summary(cal_log)["mean"] if calibrate else _metric_summary(raw_log)["mean"],
        "log_loss_std": _metric_summary(cal_log)["std"] if calibrate else _metric_summary(raw_log)["std"],
    }

    return summary, prediction_rows


def _late_fusion(
    ml_pred: pd.DataFrame,
    mech_pred: pd.DataFrame,
    class_order: List[str],
    alphas: List[float],
) -> dict:
    required = {"row_id", "seed", "fold", "y_true"}
    if not required.issubset(ml_pred.columns) or not required.issubset(mech_pred.columns):
        raise ValueError("Prediction tables missing required columns for late fusion.")

    ml_prob_cols = [f"prob_raw_{cls}" for cls in class_order]
    mech_prob_cols = [f"prob_cal_{cls}" for cls in class_order]

    ml_pred = ml_pred[["row_id", "seed", "fold", "y_true", *ml_prob_cols]].copy()
    mech_pred = mech_pred[["row_id", "seed", "fold", "y_true", *mech_prob_cols]].copy()

    merged = ml_pred.merge(
        mech_pred,
        on=["row_id", "seed", "fold", "y_true"],
        how="inner",
    )

    results = []
    for alpha in alphas:
        seed_metrics = []
        for seed in sorted(merged["seed"].unique()):
            subset = merged[merged["seed"] == seed]
            ml_probs = subset[ml_prob_cols].to_numpy()
            mech_probs = subset[mech_prob_cols].to_numpy()
            fused_probs = (1 - alpha) * ml_probs + alpha * mech_probs
            fused_probs = fused_probs / fused_probs.sum(axis=1, keepdims=True)
            y_true = subset["y_true"].to_numpy()
            y_pred = np.array([class_order[idx] for idx in np.argmax(fused_probs, axis=1)])

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="y_pred contains classes not in y_true",
                )
                macro_f1 = float(
                    f1_score(
                        y_true,
                        y_pred,
                        average="macro",
                        labels=class_order,
                        zero_division=0,
                    )
                )

            seed_metrics.append(
                {
                    "seed": int(seed),
                    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                    "log_loss": float(log_loss(y_true, fused_probs, labels=class_order)),
                    "macro_f1": macro_f1,
                }
            )
        bal = [m["balanced_accuracy"] for m in seed_metrics]
        ll = [m["log_loss"] for m in seed_metrics]
        f1 = [m["macro_f1"] for m in seed_metrics]
        results.append(
            {
                "alpha": alpha,
                "balanced_accuracy": _metric_summary(bal),
                "log_loss": _metric_summary(ll),
                "macro_f1": _metric_summary(f1),
                "per_seed": seed_metrics,
            }
        )

    best = None
    for row in results:
        if best is None:
            best = row
            continue
        if row["balanced_accuracy"]["mean"] > best["balanced_accuracy"]["mean"]:
            best = row
        elif row["balanced_accuracy"]["mean"] == best["balanced_accuracy"]["mean"]:
            if row["log_loss"]["mean"] < best["log_loss"]["mean"]:
                best = row

    return {
        "alpha_sweep": results,
        "selected": best,
    }


def _abstention_at_coverages(
    pred: pd.DataFrame,
    class_order: List[str],
    coverages: List[float],
    prob_cols: List[str],
) -> dict:
    rows = []
    for (seed, fold), group in pred.groupby(["seed", "fold"]):
        probs = group[prob_cols].to_numpy()
        confidence = probs.max(axis=1)
        order = np.argsort(confidence)
        y_true = group["y_true"].to_numpy()
        for cov in coverages:
            keep_n = int(np.ceil(cov * len(group)))
            keep_idx = order[-keep_n:]
            y_keep = y_true[keep_idx]
            probs_keep = probs[keep_idx]
            y_pred = np.array([class_order[idx] for idx in np.argmax(probs_keep, axis=1)])
            bal = float(balanced_accuracy_score(y_keep, y_pred)) if len(y_keep) else float("nan")
            rows.append({"seed": seed, "fold": fold, "coverage": cov, "balanced_accuracy": bal})
    summary = {}
    for cov in coverages:
        subset = [r["balanced_accuracy"] for r in rows if r["coverage"] == cov]
        summary[str(cov)] = _metric_summary(subset)
    return summary


def train_fusion_model(
    seeds: List[int],
    n_splits: int = 5,
    mechanistic_features_path: Path | None = None,
) -> dict:
    data, join_summary = build_joined_dataset(mechanistic_features_path=mechanistic_features_path)
    class_order = sorted(data["risk_class"].unique())

    mech_features = MECH_NUMERIC_COLS + MECH_PROVENANCE_COLS

    def _apply_ablation(base: pd.DataFrame, mode: str) -> pd.DataFrame:
        altered = base.copy()
        if mode == "zero_mech":
            for col in mech_features:
                if col in altered.columns:
                    altered[col] = 0.0
        elif mode == "zero_ml":
            for col in ML_NUMERIC_COLS:
                if col in altered.columns:
                    altered[col] = 0.0
            for col in ML_CATEGORICAL_COLS:
                if col in altered.columns:
                    altered[col] = "__missing__"
        return altered

    outer_splits = _build_outer_splits(
        X=data[ML_CATEGORICAL_COLS + ML_NUMERIC_COLS + mech_features],
        y=data["risk_class"],
        groups=data["drug_name_parent"],
        seeds=seeds,
        n_splits=n_splits,
    )

    ml_only_summary, ml_preds = _evaluate_config(
        name="ml_only",
        data=data,
        categorical_features=ML_CATEGORICAL_COLS,
        numeric_features=ML_NUMERIC_COLS,
        seeds=seeds,
        n_splits=n_splits,
        class_order=class_order,
        outer_splits=outer_splits,
        calibrate=False,
        collect_predictions=True,
    )

    mech_only_summary, mech_preds = _evaluate_config(
        name="mech_only",
        data=data,
        categorical_features=[],
        numeric_features=mech_features,
        seeds=seeds,
        n_splits=n_splits,
        class_order=class_order,
        outer_splits=outer_splits,
        calibrate=True,
        collect_predictions=True,
    )

    fused_summary, fused_predictions = _evaluate_config(
        name="fused",
        data=data,
        categorical_features=ML_CATEGORICAL_COLS,
        numeric_features=ML_NUMERIC_COLS + mech_features,
        seeds=seeds,
        n_splits=n_splits,
        class_order=class_order,
        outer_splits=outer_splits,
        calibrate=True,
        collect_predictions=True,
    )

    fused_minus_mech_summary, _ = _evaluate_config(
        name="fused_minus_mech",
        data=data,
        categorical_features=ML_CATEGORICAL_COLS,
        numeric_features=ML_NUMERIC_COLS,
        seeds=seeds,
        n_splits=n_splits,
        class_order=class_order,
        outer_splits=outer_splits,
        calibrate=False,
        collect_predictions=False,
    )

    fused_minus_ml_summary, _ = _evaluate_config(
        name="fused_minus_ml",
        data=data,
        categorical_features=[],
        numeric_features=mech_features,
        seeds=seeds,
        n_splits=n_splits,
        class_order=class_order,
        outer_splits=outer_splits,
        calibrate=True,
        collect_predictions=False,
    )

    def _delta(metric: str, base: dict, other: dict) -> float | None:
        if base.get(metric) is None or other.get(metric) is None:
            return None
        return float(other[metric] - base[metric])

    def _ablation_check(base: dict, other: dict, label: str, tol: float = 1e-6) -> dict:
        bal_diff = abs(
            other["metrics"]["raw"]["balanced_accuracy"]["mean"]
            - base["metrics"]["raw"]["balanced_accuracy"]["mean"]
        )
        f1_diff = abs(
            other["metrics"]["raw"]["macro_f1"]["mean"]
            - base["metrics"]["raw"]["macro_f1"]["mean"]
        )
        return {
            "name": label,
            "balanced_accuracy_diff": float(bal_diff),
            "macro_f1_diff": float(f1_diff),
            "tolerance": tol,
            "pass": bool(bal_diff <= tol and f1_diff <= tol),
        }

    ml_pred_df = pd.DataFrame(ml_preds)
    mech_pred_df = pd.DataFrame(mech_preds)

    alpha_sweep = _late_fusion(
        ml_pred_df,
        mech_pred_df,
        class_order=class_order,
        alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
    )

    coverages = [0.9, 0.7, 0.5]
    ml_abstention = _abstention_at_coverages(
        pred=ml_pred_df,
        class_order=class_order,
        coverages=coverages,
        prob_cols=[f"prob_raw_{cls}" for cls in class_order],
    )

    selected_alpha = alpha_sweep["selected"]["alpha"]
    fused_probs_cols = [f"prob_raw_{cls}" for cls in class_order]
    mech_probs_cols = [f"prob_cal_{cls}" for cls in class_order]
    late_fused = ml_pred_df[["row_id", "seed", "fold", "y_true", *fused_probs_cols]].merge(
        mech_pred_df[["row_id", "seed", "fold", "y_true", *mech_probs_cols]],
        on=["row_id", "seed", "fold", "y_true"],
        how="inner",
    )
    late_probs = (1 - selected_alpha) * late_fused[fused_probs_cols].to_numpy() + selected_alpha * late_fused[
        mech_probs_cols
    ].to_numpy()
    late_probs = late_probs / late_probs.sum(axis=1, keepdims=True)
    late_fused_pred = late_fused.copy()
    for idx, cls in enumerate(class_order):
        late_fused_pred[f"prob_{cls}"] = late_probs[:, idx]

    late_abstention = _abstention_at_coverages(
        pred=late_fused_pred,
        class_order=class_order,
        coverages=coverages,
        prob_cols=[f"prob_{cls}" for cls in class_order],
    )

    results = {
        "dataset": {
            "id": DATASET_ID,
            "path": str(ROOT / "data" / "processed" / f"{DATASET_ID}.csv"),
            "sha256": _sha256_file(ROOT / "data" / "processed" / f"{DATASET_ID}.csv"),
            "rows": int(len(data)),
            "columns": list(data.columns),
        },
        "mechanistic_features": {
            "path": str(mechanistic_features_path or MECH_FEATURES_PATH),
            "sha256": _sha256_file(mechanistic_features_path or MECH_FEATURES_PATH),
            "columns": mech_features,
        },
        "class_order": class_order,
        "seeds": seeds,
        "split_method": f"StratifiedGroupKFold(n_splits={n_splits}, shuffle=True)",
        "group_column": "drug_name_parent",
        "join_summary_path": str(JOIN_SUMMARY_PATH),
        "configs": [
            ml_only_summary,
            mech_only_summary,
            fused_summary,
            fused_minus_mech_summary,
            fused_minus_ml_summary,
        ],
        "ablation_checks": {
            "fused_minus_mech": _ablation_check(ml_only_summary, fused_minus_mech_summary, "fused_minus_mech"),
            "fused_minus_ml": _ablation_check(mech_only_summary, fused_minus_ml_summary, "fused_minus_ml"),
        },
        "deltas_vs_ml_only": {
            "mech_only": {
                "balanced_accuracy": _delta(
                    "balanced_accuracy_mean", ml_only_summary, mech_only_summary
                ),
                "macro_f1": _delta("macro_f1_mean", ml_only_summary, mech_only_summary),
                "log_loss": _delta("log_loss_mean", ml_only_summary, mech_only_summary),
            },
            "fused": {
                "balanced_accuracy": _delta("balanced_accuracy_mean", ml_only_summary, fused_summary),
                "macro_f1": _delta("macro_f1_mean", ml_only_summary, fused_summary),
                "log_loss": _delta("log_loss_mean", ml_only_summary, fused_summary),
            },
        },
        "late_fusion": {
            "alpha_sweep": alpha_sweep["alpha_sweep"],
            "selected": alpha_sweep["selected"],
            "abstention_coverages": coverages,
            "abstention_ml_only": ml_abstention,
            "abstention_late_fused": late_abstention,
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    pred_df = pd.DataFrame(fused_predictions).sort_values(["seed", "fold", "row_id"])
    pred_df.to_csv(PREDICTIONS_PATH, index=False)

    if not pred_df.empty:
        prob_cols = [f"prob_cal_{cls}" for cls in class_order]
        sample = pred_df.sample(n=min(20, len(pred_df)), random_state=seeds[0])
        prob_matrix = sample[prob_cols].to_numpy()
        argmax_labels = [class_order[idx] for idx in np.argmax(prob_matrix, axis=1)]
        mismatch = int(np.sum(sample["y_pred"].to_numpy() != np.array(argmax_labels)))

        results["probability_self_check"] = {
            "checked_rows": int(len(sample)),
            "mismatches": mismatch,
            "class_order": class_order,
        }

        FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        display = ConfusionMatrixDisplay.from_predictions(
            pred_df["y_true"],
            pred_df["y_pred"],
            normalize="true",
            values_format=".2f",
            cmap="Blues",
        )
        display.ax_.set_title("Fused Model (HistGradientBoosting)")
        display.figure_.tight_layout()
        display.figure_.savefig(FIGURE_PATH, dpi=150)

    return {
        "results": results,
        "join_summary": join_summary,
        "predictions_path": str(PREDICTIONS_PATH),
        "figure_path": str(FIGURE_PATH),
    }
