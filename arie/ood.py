"""kNN-based applicability domain (OOD) scoring and abstention analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD

from arie.data import load_processed_dataset

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT / "results" / "calibration_predictions.csv"
OOD_SCORES_PATH = ROOT / "results" / "ood_knn_scores.csv"
CURVE_PATH = ROOT / "results" / "abstention_ood_knn_curve.csv"
SUMMARY_PATH = ROOT / "results" / "abstention_ood_knn_summary.json"


FEATURE_COLS = [
    "cell_type",
    "platform",
    "ead_type",
    "concentration_level",
    "ead",
    "dd_fpdc",
    "site",
]


def _default_rates() -> List[float]:
    return [round(x, 2) for x in np.arange(0.0, 0.51, 0.05)]


def _build_preprocessor() -> ColumnTransformer:
    categorical_features = ["cell_type", "platform", "ead_type", "site"]
    numeric_features = ["concentration_level", "ead", "dd_fpdc"]

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_features),
            ("num", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
    )


def _class_conditional_knn_distances(
    X_train,
    X_test,
    y_train: pd.Series,
    pred: pd.DataFrame,
    seed: int,
    fold_idx: int,
    test_row_ids: np.ndarray,
    k: int,
    metric: str,
    global_nn: NearestNeighbors | None = None,
) -> np.ndarray:
    pred_fold = pred[(pred["seed"] == seed) & (pred["fold"] == fold_idx)]
    if pred_fold.empty:
        raise RuntimeError(f"No predictions found for seed={seed}, fold={fold_idx}.")

    pred_map = pred_fold.set_index("row_id")["y_pred"].to_dict()
    classes = sorted(y_train.unique())

    # Build per-class neighbor indices
    class_indices = {cls: np.where(y_train == cls)[0] for cls in classes}
    class_nn = {}
    for cls, idx in class_indices.items():
        if len(idx) == 0:
            continue
        nn = NearestNeighbors(n_neighbors=min(k, len(idx)), metric=metric, algorithm="brute")
        nn.fit(X_train[idx])
        class_nn[cls] = nn

    distances = np.zeros(X_test.shape[0], dtype=float)
    pred_labels = []
    for row_id in test_row_ids:
        if int(row_id) not in pred_map:
            raise RuntimeError(
                f"Missing predicted label for row_id={row_id} (seed={seed}, fold={fold_idx})."
            )
        pred_labels.append(pred_map[int(row_id)])
    pred_labels = np.array(pred_labels, dtype=object)

    for cls in np.unique(pred_labels):
        idx = np.where(pred_labels == cls)[0]
        if len(idx) == 0:
            continue
        nn = class_nn.get(cls)
        if nn is None:
            if global_nn is None:
                raise RuntimeError(f"No kNN index available for class {cls}.")
            dist, _ = global_nn.kneighbors(X_test[idx], n_neighbors=global_nn.n_neighbors, return_distance=True)
        else:
            dist, _ = nn.kneighbors(X_test[idx], n_neighbors=nn.n_neighbors, return_distance=True)
        distances[idx] = dist.mean(axis=1)

    return distances


def compute_knn_ood_scores(
    predictions_path: Path | None = None,
    processed_path: Path | None = None,
    seeds: List[int] | None = None,
    n_splits: int | None = None,
    k: int = 5,
    method: str = "cosine_sparse",
    n_components: int = 30,
) -> Tuple[pd.DataFrame, List[dict], dict]:
    predictions_path = predictions_path or PREDICTIONS_PATH
    pred = pd.read_csv(predictions_path)
    if "split_role" in pred.columns:
        pred = pred[pred["split_role"] == "test"].copy()

    seeds = seeds or sorted(pred["seed"].unique().tolist())
    if n_splits is None:
        n_splits = int(pred["fold"].max()) + 1

    df = load_processed_dataset(processed_path)
    df = df.dropna(subset=["risk_class"]).copy()

    X = df[FEATURE_COLS]
    y = df["risk_class"]
    groups = df["drug_name"]

    preprocessor = _build_preprocessor()

    rows = []
    overlap_checks = []
    is_margin = method.endswith("_margin")
    use_class_conditional = method.endswith("_class") or is_margin
    base_method = method.replace("_class", "").replace("_margin", "")
    method_meta = {
        "method": method,
        "base_method": base_method,
        "class_conditional": use_class_conditional,
        "k": k,
        "n_components": n_components if base_method == "pca_euclidean" else None,
    }

    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])
            overlap = train_groups.intersection(test_groups)
            overlap_checks.append(
                {
                    "seed": int(seed),
                    "fold": int(fold_idx),
                    "train_test_overlap": int(len(overlap)),
                }
            )

            X_train = preprocessor.fit_transform(X.iloc[train_idx])
            X_test = preprocessor.transform(X.iloc[test_idx])

            test_row_ids = df.iloc[test_idx].index.to_numpy()

            if base_method == "pca_euclidean":
                n_features = X_train.shape[1]
                n_comp = min(n_components, max(1, n_features - 1))
                svd = TruncatedSVD(n_components=n_comp, random_state=seed)
                X_train_dense = svd.fit_transform(X_train)
                X_test_dense = svd.transform(X_test)
                scaler = StandardScaler()
                X_train_dense = scaler.fit_transform(X_train_dense)
                X_test_dense = scaler.transform(X_test_dense)
                method_meta["n_components"] = n_comp

                global_nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute")
                global_nn.fit(X_train_dense)
                global_dist, _ = global_nn.kneighbors(X_test_dense, n_neighbors=k, return_distance=True)

                if use_class_conditional:
                    class_dist = _class_conditional_knn_distances(
                        X_train_dense,
                        X_test_dense,
                        y.iloc[train_idx],
                        pred,
                        seed,
                        fold_idx,
                        test_row_ids,
                        k=k,
                        metric="euclidean",
                        global_nn=global_nn,
                    )
                else:
                    class_dist = None

                if is_margin and class_dist is not None:
                    distances = class_dist - global_dist.mean(axis=1)
                elif class_dist is not None:
                    distances = class_dist
                else:
                    distances = global_dist
            elif base_method == "cosine_sparse":
                global_nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
                global_nn.fit(X_train)
                global_dist, _ = global_nn.kneighbors(X_test, n_neighbors=k, return_distance=True)

                if use_class_conditional:
                    class_dist = _class_conditional_knn_distances(
                        X_train,
                        X_test,
                        y.iloc[train_idx],
                        pred,
                        seed,
                        fold_idx,
                        test_row_ids,
                        k=k,
                        metric="cosine",
                        global_nn=global_nn,
                    )
                else:
                    class_dist = None

                if is_margin and class_dist is not None:
                    distances = class_dist - global_dist.mean(axis=1)
                elif class_dist is not None:
                    distances = class_dist
                else:
                    distances = global_dist
            else:
                raise ValueError(f"Unknown OOD method: {method}")

            if distances.ndim == 1:
                mean_dist = distances
            else:
                mean_dist = distances.mean(axis=1)

            for row_id, dist in zip(test_row_ids, mean_dist):
                rows.append(
                    {
                        "row_id": int(row_id),
                        "seed": int(seed),
                        "fold": int(fold_idx),
                        "ood_knn_distance": float(dist),
                        "ood_method": method,
                    }
                )

    ood_df = pd.DataFrame(rows)
    return ood_df, overlap_checks, method_meta


def compute_abstention_ood_curve(
    ood_df: pd.DataFrame,
    rates: Iterable[float] | None = None,
    curve_path: Path | None = None,
    summary_path: Path | None = None,
    method_meta: dict | None = None,
    write_outputs: bool = True,
) -> dict:
    curve_path = curve_path or CURVE_PATH
    summary_path = summary_path or SUMMARY_PATH

    rates = list(rates or _default_rates())
    rates = sorted(set(float(r) for r in rates))

    required_cols = {"seed", "fold", "ood_knn_distance", "y_true", "y_pred"}
    missing = required_cols.difference(ood_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for abstention: {sorted(missing)}")

    labels = sorted(ood_df["y_true"].unique())

    rows = []
    for (seed, fold), group in ood_df.groupby(["seed", "fold"], sort=True):
        group_sorted = group.sort_values("ood_knn_distance", ascending=False)
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
                    "abstention_rate": rate,
                    "coverage": coverage,
                    "n_total": int(n_total),
                    "n_kept": int(n_kept),
                    "balanced_accuracy": float(bal_acc),
                    "macro_f1": float(macro_f1),
                }
            )

    curve_df = pd.DataFrame(rows)
    if write_outputs:
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        curve_df.to_csv(curve_path, index=False)

    summary = {}
    for rate in rates:
        subset = curve_df[curve_df["abstention_rate"] == rate]
        if subset.empty:
            continue
        summary[str(rate)] = {
            "coverage_mean": float(subset["coverage"].mean()),
            "coverage_std": float(subset["coverage"].std(ddof=1)) if len(subset) > 1 else 0.0,
            "balanced_accuracy_mean": float(subset["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(subset["balanced_accuracy"].std(ddof=1)) if len(subset) > 1 else 0.0,
            "macro_f1_mean": float(subset["macro_f1"].mean()),
            "macro_f1_std": float(subset["macro_f1"].std(ddof=1)) if len(subset) > 1 else 0.0,
            "n_groups": int(len(subset)),
        }

    summary_payload = {
        "method": method_meta or {},
        "rates": rates,
        "summary": summary,
        "curve_path": str(curve_path),
    }

    if write_outputs:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    return summary_payload
