"""Mechanistic plausibility scoring for ML predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import warnings
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from arie.data import load_processed_dataset
from arie.names import normalize_compound

ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = ROOT / "data" / "processed" / "mechanistic_multichannel_features_cipa28.csv"
JOIN_SUMMARY_PATH = ROOT / "results" / "mechanistic_multichannel_feature_join_summary.json"

DEFAULT_PREDICTIONS_PATH = ROOT / "results" / "calibration_predictions.csv"
OUTPUT_SCORES_PATH = ROOT / "results" / "mech_plausibility_scores.csv"
OUTPUT_SUMMARY_PATH = ROOT / "results" / "mech_plausibility_summary.json"

CHANNELS = [
    "hERG",
    "Nav1.5_peak",
    "Nav1.5_late",
    "Cav1.2",
    "IKs",
    "IK1",
    "Kv4.3",
]

CLASS_ORDER = ["H", "L", "M"]


def _sha256_file(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class FoldInfo:
    seed: int
    fold: int
    train_parents: List[str]
    test_parents: List[str]


def _resolve_drug_label(counts: pd.Series) -> str:
    if counts.empty:
        return ""
    max_count = counts.max()
    top = sorted([cls for cls, val in counts.items() if val == max_count])
    if len(top) == 1:
        return top[0]
    risk_order = {"L": 0, "M": 1, "H": 2}
    return sorted(top, key=lambda x: risk_order.get(x, -1))[-1]


def load_predictions(path: Path, enable_identity_alias: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "drug_name_parent" not in df.columns:
        if "drug_name" not in df.columns:
            raise ValueError("Predictions file must include drug_name or drug_name_parent.")
        df["drug_name_parent"] = df["drug_name"].apply(
            lambda x: normalize_compound(x, enable_identity_alias=enable_identity_alias)[
                "drug_name_parent"
            ]
        )
    return df


def load_mechanistic_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing mechanistic feature table: {FEATURES_PATH}")
    return pd.read_csv(FEATURES_PATH)


def build_drug_labels(enable_identity_alias: bool = False) -> Tuple[pd.DataFrame, dict]:
    cipa = load_processed_dataset().dropna(subset=["risk_class"]).copy()
    cipa["drug_name_parent"] = cipa["drug_name"].apply(
        lambda x: normalize_compound(x, enable_identity_alias=enable_identity_alias)["drug_name_parent"]
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
    return drug_df, conflicts


def _feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col.startswith("ic50_uM_")]


def _channels_present(row: pd.Series, feature_cols: List[str]) -> List[str]:
    present = []
    for col in feature_cols:
        if pd.notna(row[col]):
            present.append(col.replace("ic50_uM_", ""))
    return present


def _missing_fraction(row: pd.Series, feature_cols: List[str]) -> float:
    total = len(feature_cols)
    if total == 0:
        return 1.0
    missing = int(row[feature_cols].isna().sum())
    return missing / total


def build_folds_from_predictions(
    predictions: pd.DataFrame,
    all_parents: List[str],
) -> List[FoldInfo]:
    folds: List[FoldInfo] = []
    for (seed, fold), group in predictions.groupby(["seed", "fold"]):
        test_parents = sorted(set(group["drug_name_parent"]))
        train_parents = sorted(set(all_parents) - set(test_parents))
        folds.append(FoldInfo(seed=int(seed), fold=int(fold), train_parents=train_parents, test_parents=test_parents))
    return sorted(folds, key=lambda f: (f.seed, f.fold))


def build_folds_stratified(
    drug_df: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> List[FoldInfo]:
    X = drug_df[["drug_name_parent"]]
    y = drug_df["risk_class"]
    groups = drug_df["drug_name_parent"]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        train_parents = sorted(groups.iloc[train_idx].unique())
        test_parents = sorted(groups.iloc[test_idx].unique())
        folds.append(FoldInfo(seed=seed, fold=fold_idx, train_parents=train_parents, test_parents=test_parents))
    return folds


def score_mech_plausibility(
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    enable_identity_alias: bool = False,
    n_splits: int = 5,
    seeds: List[int] | None = None,
) -> Tuple[pd.DataFrame, dict]:
    warnings.filterwarnings(
        "ignore",
        message="Skipping features without any observed values",
    )

    predictions = load_predictions(predictions_path, enable_identity_alias=enable_identity_alias)
    predictions = predictions.copy()
    predictions["split_role"] = predictions.get("split_role", "test")
    predictions_all_rows = len(predictions)
    predictions = predictions[predictions["split_role"] == "test"].copy()

    drug_labels, conflicts = build_drug_labels(enable_identity_alias=enable_identity_alias)
    feature_df = load_mechanistic_features()

    feature_cols = _feature_columns(feature_df)
    if not feature_cols:
        raise RuntimeError("No mechanistic feature columns found (ic50_uM_*).")

    feature_df["channels_present"] = feature_df.apply(
        lambda row: _channels_present(row, feature_cols), axis=1
    )
    feature_df["mech_missing_frac"] = feature_df.apply(
        lambda row: _missing_fraction(row, feature_cols), axis=1
    )

    merged = drug_labels.merge(feature_df, on="drug_name_parent", how="left")

    seeds = seeds or sorted(predictions["seed"].unique())
    folds_from_preds = build_folds_from_predictions(predictions, merged["drug_name_parent"].tolist())

    # Optional split alignment check against StratifiedGroupKFold.
    alignment_checks = []
    for seed in seeds:
        stratified_folds = build_folds_stratified(merged, n_splits=n_splits, seed=seed)
        stratified_map = {(f.seed, f.fold): set(f.test_parents) for f in stratified_folds}
        for f in [f for f in folds_from_preds if f.seed == seed]:
            expected = stratified_map.get((seed, f.fold), set())
            observed = set(f.test_parents)
            diff = expected.symmetric_difference(observed)
            alignment_checks.append(
                {
                    "seed": seed,
                    "fold": f.fold,
                    "expected_count": len(expected),
                    "observed_count": len(observed),
                    "symmetric_diff": len(diff),
                }
            )

    results_rows = []
    overlap_checks = []
    fallback_trains = 0

    for fold_info in folds_from_preds:
        seed = fold_info.seed
        fold = fold_info.fold

        train_mask = merged["drug_name_parent"].isin(fold_info.train_parents)
        test_mask = merged["drug_name_parent"].isin(fold_info.test_parents)

        train_df = merged[train_mask].copy()
        test_df = merged[test_mask].copy()

        train_parents = set(train_df["drug_name_parent"])
        test_parents = set(test_df["drug_name_parent"])
        overlap = train_parents.intersection(test_parents)
        overlap_checks.append(
            {
                "seed": seed,
                "fold": fold,
                "train_parents": len(train_parents),
                "test_parents": len(test_parents),
                "overlap_count": len(overlap),
            }
        )

        X_train = train_df[feature_cols]
        y_train = train_df["risk_class"]
        X_test = test_df[feature_cols]
        y_test = test_df["risk_class"]

        use_fallback = y_train.nunique() < 2
        if use_fallback:
            fallback_trains += 1

        if use_fallback:
            class_counts = y_train.value_counts()
            probs = []
            for _ in range(len(test_df)):
                row = {cls: 0.0 for cls in CLASS_ORDER}
                for cls in class_counts.index:
                    if cls in row:
                        row[cls] = 1.0
                probs.append(row)
            prob_df = pd.DataFrame(probs)
        else:
            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )
            pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", model),
                ]
            )
            pipeline.fit(X_train, y_train)
            probs = pipeline.predict_proba(X_test)
            classes = list(pipeline.named_steps["model"].classes_)

            prob_df = pd.DataFrame(probs, columns=classes, index=test_df.index)
            for cls in CLASS_ORDER:
                if cls not in prob_df.columns:
                    prob_df[cls] = 0.0
            prob_df = prob_df[CLASS_ORDER]

        test_df = test_df.reset_index(drop=True)
        prob_df = prob_df.reset_index(drop=True)

        test_df["p_mech_H"] = prob_df["H"]
        test_df["p_mech_L"] = prob_df["L"]
        test_df["p_mech_M"] = prob_df["M"]

        test_df["mech_support"] = test_df.apply(
            lambda row: row[f"p_mech_{row['risk_class']}"]
            if row["risk_class"] in CLASS_ORDER
            else np.nan,
            axis=1,
        )
        test_df["mech_disagreement"] = 1.0 - test_df["mech_support"]

        preds_fold = predictions[(predictions["seed"] == seed) & (predictions["fold"] == fold)].copy()
        preds_fold = preds_fold.merge(
            test_df[
                [
                    "drug_name_parent",
                    "p_mech_H",
                    "p_mech_L",
                    "p_mech_M",
                    "mech_missing_frac",
                    "channels_present",
                ]
            ],
            on="drug_name_parent",
            how="left",
            validate="many_to_one",
        )

        preds_fold["y_pred_ml"] = preds_fold["y_pred"]
        preds_fold["confidence_ml"] = preds_fold.get("confidence")

        preds_fold["mech_support"] = preds_fold.apply(
            lambda row: row.get(f"p_mech_{row['y_pred_ml']}") if row["y_pred_ml"] in CLASS_ORDER else np.nan,
            axis=1,
        )
        preds_fold["mech_disagreement"] = 1.0 - preds_fold["mech_support"]
        preds_fold["mech_plausibility"] = (1.0 - preds_fold["mech_missing_frac"]) * preds_fold["mech_support"]

        preds_fold["channels_present"] = preds_fold["channels_present"].apply(
            lambda val: "|".join(val) if isinstance(val, list) else (val or "")
        )

        results_rows.append(preds_fold)

    if results_rows:
        scored = pd.concat(results_rows, ignore_index=True)
    else:
        scored = pd.DataFrame()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions_path": str(predictions_path),
        "predictions_sha256": _sha256_file(predictions_path),
        "predictions_rows_total": int(predictions_all_rows),
        "predictions_rows_test": int(len(predictions)),
        "features_path": str(FEATURES_PATH),
        "identity_alias_enabled": enable_identity_alias,
        "seeds": sorted(predictions["seed"].unique().tolist()),
        "n_splits": int(n_splits),
        "mech_support_definition": "mech_support = p_mech[y_pred_ml], where p_mech comes from a mechanistic-only classifier trained on train drugs.",
        "mech_support_higher_is": "more_agreement_with_mechanistic_model (not necessarily more_correct)",
        "mech_disagreement_definition": "mech_disagreement = 1 - mech_support",
        "mech_plausibility_definition": "mech_plausibility = (1 - mech_missing_frac) * mech_support",
        "mech_plausibility_higher_is": "more_mechanistic_agreement_with_higher_coverage",
        "alignment_checks": alignment_checks,
        "overlap_checks": overlap_checks,
        "rows_total": int(len(predictions)),
        "rows_scored": int(len(scored)),
        "fallback_trains_due_to_single_class": int(fallback_trains),
        "feature_columns": feature_cols,
        "channel_coverage": {
            col.replace("ic50_uM_", ""): int(feature_df[col].notna().sum())
            for col in feature_cols
        },
        "label_conflicts": conflicts,
        "label_conflict_count": int(len(conflicts)),
        "label_conflict_parents": sorted(conflicts.keys()),
    }

    if JOIN_SUMMARY_PATH.exists():
        join_summary = json.loads(JOIN_SUMMARY_PATH.read_text())
        summary["feature_join_summary_alias_on"] = join_summary.get("alias_on")

    if not scored.empty:
        scored["error"] = (scored["y_true"] != scored["y_pred_ml"]).astype(int)
        scored["fully_missing"] = scored["mech_missing_frac"].fillna(1.0) >= 1.0

        metrics_by_seed = {}
        bin_edges = [0.0, 0.25, 0.5, 0.75, 1.0]
        bin_labels = ["<=0.25", "0.25-0.5", "0.5-0.75", ">0.75"]

        for seed, group in scored.groupby("seed"):
            usable = group[~group["fully_missing"] & group["mech_support"].notna()].copy()
            correct = usable[usable["error"] == 0]
            incorrect = usable[usable["error"] == 1]

            def _auc(series: pd.Series) -> float | None:
                try:
                    if usable["error"].nunique() < 2:
                        return None
                    return float(roc_auc_score(usable["error"], series))
                except Exception:
                    return None

            roc_auc = None
            try:
                if usable["error"].nunique() > 1:
                    roc_auc = float(roc_auc_score(usable["error"], usable["mech_disagreement"]))
            except Exception:
                roc_auc = None

            corr = None
            try:
                corr = float(usable["mech_support"].corr(usable["mech_disagreement"]))
            except Exception:
                corr = None

            bins = []
            usable["missing_bin"] = pd.cut(
                usable["mech_missing_frac"],
                bins=bin_edges,
                labels=bin_labels,
                include_lowest=True,
            )
            for label in bin_labels:
                subset = usable[usable["missing_bin"] == label]
                if subset.empty:
                    bins.append(
                        {
                            "bin": label,
                            "n_rows": 0,
                            "auc_support": None,
                            "auc_disagreement": None,
                        }
                    )
                    continue
                try:
                    auc_support = float(roc_auc_score(subset["error"], subset["mech_support"]))
                except Exception:
                    auc_support = None
                try:
                    auc_dis = float(roc_auc_score(subset["error"], subset["mech_disagreement"]))
                except Exception:
                    auc_dis = None
                bins.append(
                    {
                        "bin": label,
                        "n_rows": int(len(subset)),
                        "auc_support": auc_support,
                        "auc_disagreement": auc_dis,
                    }
                )

            metrics_by_seed[int(seed)] = {
                "n_rows": int(len(group)),
                "n_rows_usable": int(len(usable)),
                "mech_support_correct_mean": float(correct["mech_support"].mean()) if len(correct) else None,
                "mech_support_correct_median": float(correct["mech_support"].median()) if len(correct) else None,
                "mech_support_incorrect_mean": float(incorrect["mech_support"].mean())
                if len(incorrect)
                else None,
                "mech_support_incorrect_median": float(incorrect["mech_support"].median())
                if len(incorrect)
                else None,
                "roc_auc_support_vs_error": _auc(usable["mech_support"]),
                "roc_auc_disagreement_vs_error": roc_auc,
                "roc_auc_missing_frac_vs_error": _auc(usable["mech_missing_frac"]),
                "roc_auc_plausibility_vs_error": _auc(usable["mech_plausibility"]),
                "roc_auc_disagreement_plus_missing": _auc(
                    usable["mech_disagreement"] + usable["mech_missing_frac"]
                ),
                "corr_support_vs_disagreement": corr,
                "missingness_bins": bins,
            }
        summary["metrics_by_seed"] = metrics_by_seed

    # Keep output columns compact and deterministic.
    output_cols = [
        "row_id",
        "seed",
        "fold",
        "split_role",
        "drug_name_parent",
        "y_true",
        "y_pred_ml",
        "confidence_ml",
        "p_mech_H",
        "p_mech_L",
        "p_mech_M",
        "mech_support",
        "mech_disagreement",
        "mech_plausibility",
        "mech_missing_frac",
        "channels_present",
    ]
    missing = [col for col in output_cols if col not in scored.columns]
    if missing:
        raise RuntimeError(f"Missing required output columns: {missing}")
    scored = scored[output_cols]

    return scored, summary


__all__ = [
    "score_mech_plausibility",
    "OUTPUT_SCORES_PATH",
    "OUTPUT_SUMMARY_PATH",
]
