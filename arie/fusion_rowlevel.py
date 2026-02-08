"""Row-level mechanistic feature fusion diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from arie.data import load_processed_dataset
from arie.mechanistic_features import load_mechanistic_feature_table
from arie.names import normalize_compound

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "fusion_rowlevel_diagnostic.json"

ML_CATEGORICAL_COLS = ["cell_type", "platform", "ead_type", "site"]
ML_NUMERIC_COLS = ["concentration_level", "ead", "dd_fpdc"]

BASE_MECH_FEATURE_COLS = [
    "herg_ic50_uM",
    "herg_nh",
    "nh_imputed",
]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_pipeline(categorical_features: List[str], numeric_features: List[str], seed: int) -> Pipeline:
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

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def _align_proba(prob: np.ndarray, model_classes: np.ndarray, class_order: List[str]) -> np.ndarray:
    aligned = np.zeros((prob.shape[0], len(class_order)), dtype=float)
    class_index = {cls: idx for idx, cls in enumerate(model_classes)}
    for j, cls in enumerate(class_order):
        if cls in class_index:
            aligned[:, j] = prob[:, class_index[cls]]
    return aligned


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


def _metric_summary(values: List[float]) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def build_rowlevel_dataset(enable_block_pred: bool = False) -> Tuple[pd.DataFrame, dict]:
    cipa = load_processed_dataset().dropna(subset=["risk_class"]).copy()
    cipa = cipa.reset_index(drop=True)
    cipa["row_id"] = cipa.index.astype(int)
    cipa["drug_name_parent"] = cipa["drug_name"].apply(
        lambda x: normalize_compound(x)["drug_name_parent"]
    )

    mech = load_mechanistic_feature_table().copy()
    mech = mech[["drug_name_parent", "herg_ic50_uM_median", "herg_ic50_uM_mean", "herg_nh_median", "herg_nh_mean"]]

    merged = cipa.merge(mech, on="drug_name_parent", how="left")

    merged["herg_ic50_uM"] = merged["herg_ic50_uM_median"].fillna(merged["herg_ic50_uM_mean"])
    merged["herg_nh"] = merged["herg_nh_median"].fillna(merged["herg_nh_mean"])

    merged["nh_imputed"] = merged["herg_nh"].isna().astype(int)
    merged["herg_nh"] = merged["herg_nh"].fillna(1.0)

    if enable_block_pred:
        conc = merged["concentration_level"].astype(float)
        ic50 = merged["herg_ic50_uM"].astype(float)
        nh = merged["herg_nh"].astype(float)

        with np.errstate(divide="ignore", invalid="ignore"):
            block = 1.0 / (1.0 + np.power(ic50 / conc, nh))
        block[~np.isfinite(block)] = np.nan
        merged["herg_block_pred"] = block
    else:
        merged["herg_block_pred"] = np.nan

    missing_nh_rate = float(merged["nh_imputed"].mean())

    summary = {
        "rows": int(len(merged)),
        "unique_parents": int(merged["drug_name_parent"].nunique()),
        "missing_nh_rate": missing_nh_rate,
        "block_pred_enabled": enable_block_pred,
        "cipa_path": str(ROOT / "data" / "processed" / "cipa_blinova_2018.csv"),
        "cipa_sha256": _sha256_file(ROOT / "data" / "processed" / "cipa_blinova_2018.csv"),
    }

    return merged, summary


def evaluate_configs(
    data: pd.DataFrame,
    seeds: List[int],
    n_splits: int,
    mech_feature_cols: List[str],
) -> dict:
    class_order = sorted(data["risk_class"].unique())

    configs = {
        "ml_only": {
            "categorical": ML_CATEGORICAL_COLS,
            "numeric": ML_NUMERIC_COLS,
        },
        "mech_row": {
            "categorical": [],
            "numeric": mech_feature_cols,
        },
        "fused": {
            "categorical": ML_CATEGORICAL_COLS,
            "numeric": ML_NUMERIC_COLS + mech_feature_cols,
        },
    }

    results = {}
    for name, cfg in configs.items():
        X = data[cfg["categorical"] + cfg["numeric"]]
        y = data["risk_class"]
        groups = data["drug_name_parent"]

        seed_metrics = []
        overlap_counts = {}

        for seed in seeds:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            y_true_all = []
            y_pred_all = []
            prob_all = []
            overlaps = []

            for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
                train_groups = set(groups.iloc[train_idx])
                test_groups = set(groups.iloc[test_idx])
                overlaps.append(len(train_groups.intersection(test_groups)))

                pipeline = _build_pipeline(cfg["categorical"], cfg["numeric"], seed)
                pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
                y_pred = pipeline.predict(X.iloc[test_idx])
                prob = pipeline.predict_proba(X.iloc[test_idx])
                prob = _align_proba(prob, pipeline.classes_, class_order)

                y_true_all.extend(y.iloc[test_idx].tolist())
                y_pred_all.extend(y_pred.tolist())
                prob_all.append(prob)

            y_true_all = np.array(y_true_all)
            y_pred_all = np.array(y_pred_all)
            prob_all = np.vstack(prob_all)

            seed_metrics.append(
                {
                    "seed": int(seed),
                    "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
                    "macro_f1": float(
                        f1_score(y_true_all, y_pred_all, average="macro", labels=class_order, zero_division=0)
                    ),
                    "log_loss": float(log_loss(y_true_all, prob_all, labels=class_order)),
                    "brier": _brier_multiclass(y_true_all, prob_all, class_order),
                    "ece": _expected_calibration_error(y_true_all, prob_all, class_order),
                }
            )
            overlap_counts[int(seed)] = overlaps

        metrics = {
            "balanced_accuracy": _metric_summary([m["balanced_accuracy"] for m in seed_metrics]),
            "macro_f1": _metric_summary([m["macro_f1"] for m in seed_metrics]),
            "log_loss": _metric_summary([m["log_loss"] for m in seed_metrics]),
            "brier": _metric_summary([m["brier"] for m in seed_metrics]),
            "ece": _metric_summary([m["ece"] for m in seed_metrics]),
        }

        results[name] = {
            "features": cfg,
            "metrics": metrics,
            "per_seed": seed_metrics,
            "overlap_counts": overlap_counts,
        }

    return {
        "class_order": class_order,
        "configs": results,
    }


def permutation_importance_check(
    data: pd.DataFrame,
    seeds: List[int],
    n_splits: int,
    mech_feature_cols: List[str],
) -> dict:
    X_cols = ML_CATEGORICAL_COLS + ML_NUMERIC_COLS + mech_feature_cols
    X = data[X_cols]
    y = data["risk_class"]
    groups = data["drug_name_parent"]

    seed = seeds[0]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(iter(splitter.split(X, y, groups)))

    pipeline = _build_pipeline(ML_CATEGORICAL_COLS, ML_NUMERIC_COLS + mech_feature_cols, seed)
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])

    perm = permutation_importance(
        pipeline,
        X.iloc[test_idx],
        y.iloc[test_idx],
        n_repeats=5,
        random_state=seed,
        scoring="balanced_accuracy",
    )

    importances = dict(zip(X_cols, perm.importances_mean))
    mech_importances = {k: v for k, v in importances.items() if k in mech_feature_cols}
    sorted_mech = sorted(mech_importances.items(), key=lambda x: x[1], reverse=True)

    return {
        "seed": int(seed),
        "fold_index": 0,
        "mechanistic_importances": sorted_mech,
    }


def run_rowlevel_diagnostic(seeds: List[int], n_splits: int, enable_block_pred: bool = False) -> dict:
    data, data_summary = build_rowlevel_dataset(enable_block_pred=enable_block_pred)

    mech_feature_cols = BASE_MECH_FEATURE_COLS.copy()
    if enable_block_pred:
        print("WARNING: herg_block_pred uses concentration_level without unit evidence.")
        mech_feature_cols.append("herg_block_pred")
    else:
        print("WARNING: concentration units unknown; herg_block_pred disabled (NaN).")

    conc = data["concentration_level"]
    conc_stats = {
        "min": float(conc.min()),
        "max": float(conc.max()),
        "unique_count": int(conc.nunique()),
        "unique_values": sorted(conc.dropna().unique().tolist())[:10],
    }

    metrics = evaluate_configs(data, seeds=seeds, n_splits=n_splits, mech_feature_cols=mech_feature_cols)

    ml_bal = metrics["configs"]["ml_only"]["metrics"]["balanced_accuracy"]["mean"]
    fused_bal = metrics["configs"]["fused"]["metrics"]["balanced_accuracy"]["mean"]

    importance = None
    if fused_bal <= ml_bal:
        importance = permutation_importance_check(
            data,
            seeds=seeds,
            n_splits=n_splits,
            mech_feature_cols=mech_feature_cols,
        )

    report = {
        "concentration_stats": conc_stats,
        "nh_imputed_rate": data_summary["missing_nh_rate"],
        "block_pred_enabled": enable_block_pred,
        "block_pred_note": "disabled (units unknown)" if not enable_block_pred else "enabled (units assumed)",
        "metrics": metrics,
        "fusion_beats_ml": fused_bal > ml_bal,
        "feature_importance": importance,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report
