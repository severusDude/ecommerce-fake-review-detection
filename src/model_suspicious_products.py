import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
LABEL_MAP = {"original": 0, "fake": 1}
ID_TO_LABEL = {0: "original", 1: "fake"}
DEFAULT_SOURCE_WEIGHTS = {
    "human_seed": 1.0,
    "human_active": 1.0,
    "label_spreading": 0.75,
    "pseudo_xgb": 0.60,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train review-level fake probability model and rank suspicious Shopee products."
    )
    parser.add_argument("--labelled-path", default="data/labels/review_shopee_labelled.csv")
    parser.add_argument("--processed-path", default="data/processed/review_shopee_processed.csv")
    parser.add_argument("--selected-features-path", default="reports/strong_labels/selected_features.csv")
    parser.add_argument("--reports-dir", default="reports/modelling")
    parser.add_argument("--entity-column", default="item_id")
    parser.add_argument("--shop-column", default="shop_id")
    parser.add_argument("--title-column", default="product_title")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--high-risk-threshold", type=float, default=0.70)
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def project_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Project root with pyproject.toml and data/ not found.")


def xgb_classifier(seed: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Missing dependency: xgboost. Run `uv sync` first.") from exc

    return XGBClassifier(
        n_estimators=350,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
    )


def build_models(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", xgb_classifier(seed)),
            ]
        ),
    }


def read_csv(path: Path, id_columns: list[str] | None = None) -> pd.DataFrame:
    dtype = {column: str for column in id_columns or []}
    return pd.read_csv(path, dtype=dtype, keep_default_na=False)


def normalize_cmtid(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def normalize_label(value: object) -> int:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1
    raise ValueError(f"Invalid label value: {value!r}")


def require_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def selected_features(path: Path) -> tuple[list[str], pd.DataFrame]:
    feature_df = pd.read_csv(path)
    require_columns(feature_df, ["feature", "group", "consensus_rank"], "selected features")
    feature_df = feature_df.sort_values("consensus_rank").reset_index(drop=True)
    features = feature_df["feature"].astype(str).tolist()
    if len(features) != len(set(features)):
        raise ValueError("Selected feature file contains duplicate feature names.")
    return features, feature_df


def load_inputs(args: argparse.Namespace, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    labelled_path = root / args.labelled_path
    processed_path = root / args.processed_path
    selected_path = root / args.selected_features_path
    for path in [labelled_path, processed_path, selected_path]:
        require_path(path)

    features, feature_df = selected_features(selected_path)
    labelled = read_csv(labelled_path, id_columns=["cmtid"])
    processed = read_csv(processed_path, id_columns=["cmtid", args.entity_column, args.shop_column])

    labelled["cmtid"] = normalize_cmtid(labelled["cmtid"])
    processed["cmtid"] = normalize_cmtid(processed["cmtid"])
    if labelled["cmtid"].duplicated().any():
        raise ValueError("Labelled data has duplicate cmtid values.")
    if processed["cmtid"].duplicated().any():
        raise ValueError("Processed data has duplicate cmtid values.")

    require_columns(labelled, ["cmtid", "fakeornot", "label_source", "confidence", *features], "labelled data")
    require_columns(processed, ["cmtid", args.entity_column, args.shop_column, *features], "processed data")
    if args.title_column not in processed.columns:
        processed[args.title_column] = ""

    labelled["label"] = labelled["fakeornot"].map(normalize_label)
    label_counts = labelled["label"].value_counts().to_dict()
    if set(label_counts) != {0, 1}:
        raise ValueError("Labelled data must contain both original and fake labels.")
    if label_counts[0] != label_counts[1]:
        raise ValueError(f"Expected balanced labels, found {label_counts}.")

    return labelled, processed, features, feature_df


def source_sample_weight(labelled: pd.DataFrame) -> np.ndarray:
    source_weights = labelled["label_source"].map(DEFAULT_SOURCE_WEIGHTS).fillna(0.75).astype(float)
    confidence = pd.to_numeric(labelled["confidence"], errors="coerce").fillna(1.0).clip(0.25, 1.0)
    return (source_weights * confidence).to_numpy()


def fit_with_optional_weight(model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, weights: np.ndarray) -> Pipeline:
    model.fit(x_train, y_train, clf__sample_weight=weights)
    return model


def probability_for_positive(model: Pipeline, x_frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x_frame)[:, 1]
    if np.isnan(probabilities).any():
        raise ValueError("Model produced NaN probabilities.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Model probabilities outside [0, 1].")
    return probabilities


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"folds": [], "mean": {}, "std": {}}
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float, np.number))]
    mean = {key: float(np.mean([row[key] for row in rows])) for key in numeric_keys}
    std = {key: float(np.std([row[key] for row in rows], ddof=1)) for key in numeric_keys if len(rows) > 1}
    return {"folds": rows, "mean": mean, "std": std}


def evaluate_cv(
    model_name: str,
    model: Pipeline,
    x_frame: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    splitter: Any,
    seed: int,
    groups: pd.Series | None = None,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    rows: list[dict[str, Any]] = []
    oof_prob = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), -1)
    split_iter = splitter.split(x_frame, y, groups) if groups is not None else splitter.split(x_frame, y)

    for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
        fold_model = build_models(seed=seed)[model_name]
        x_train = x_frame.iloc[train_idx]
        x_test = x_frame.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        fit_with_optional_weight(fold_model, x_train, y_train, weights[train_idx])
        y_prob = probability_for_positive(fold_model, x_test)
        y_pred = (y_prob >= 0.5).astype(int)
        row = {"fold": fold, **metric_row(y_test.to_numpy(), y_pred, y_prob)}
        rows.append(row)
        oof_prob[test_idx] = y_prob
        oof_pred[test_idx] = y_pred

    if np.isnan(oof_prob).any() or (oof_pred < 0).any():
        return summarize_rows(rows), None, None
    return summarize_rows(rows), oof_prob, oof_pred


def cv_evaluations(
    labelled: pd.DataFrame,
    processed: pd.DataFrame,
    features: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    labelled_with_entity = labelled[["cmtid"]].merge(
        processed[["cmtid", args.entity_column]],
        on="cmtid",
        how="left",
        validate="one_to_one",
    )
    if labelled_with_entity[args.entity_column].isna().any():
        raise ValueError("Some labelled cmtid values are missing from processed data.")

    x_frame = labelled[features].apply(pd.to_numeric, errors="coerce")
    y = labelled["label"].astype(int)
    weights = source_sample_weight(labelled)
    models = build_models(args.seed)
    results: dict[str, Any] = {
        "stratified_kfold": {},
        "stratified_group_kfold_by_item": {},
    }
    xgb_oof_prob: np.ndarray | None = None
    xgb_oof_pred: np.ndarray | None = None

    for model_name, model in models.items():
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        summary, oof_prob, oof_pred = evaluate_cv(model_name, model, x_frame, y, weights, splitter, args.seed)
        results["stratified_kfold"][model_name] = summary
        if model_name == "xgboost":
            xgb_oof_prob = oof_prob
            xgb_oof_pred = oof_pred

    group_count = labelled_with_entity[args.entity_column].nunique()
    if group_count >= args.folds:
        groups = labelled_with_entity[args.entity_column].astype(str)
        for model_name, model in models.items():
            splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
            summary, _, _ = evaluate_cv(model_name, model, x_frame, y, weights, splitter, args.seed, groups)
            results["stratified_group_kfold_by_item"][model_name] = summary
    else:
        results["stratified_group_kfold_by_item"]["skipped"] = f"Only {group_count} groups for {args.folds} folds."

    if xgb_oof_prob is None or xgb_oof_pred is None:
        raise ValueError("XGBoost out-of-fold predictions were not produced.")
    return results, xgb_oof_prob, xgb_oof_pred


def mode_or_first(series: pd.Series) -> object:
    clean = series.dropna()
    if clean.empty:
        return ""
    modes = clean.mode()
    if not modes.empty:
        return modes.iloc[0]
    return clean.iloc[0]


def percentile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    if numeric.nunique(dropna=False) <= 1:
        return pd.Series(0.0, index=series.index)
    return numeric.rank(pct=True, method="average")


def add_reason_codes(product_scores: pd.DataFrame) -> pd.DataFrame:
    result = product_scores.copy()
    pressure_columns = [
        "mean_fake_probability",
        "high_risk_review_share",
        "review_count_pressure",
        "burst_pressure",
        "duplicate_pressure",
        "empty_pressure",
    ]
    labels = {
        "mean_fake_probability": "high mean fake probability",
        "high_risk_review_share": "many high-risk reviews",
        "review_count_pressure": "high review volume",
        "burst_pressure": "burst-like timing",
        "duplicate_pressure": "duplicate comment pattern",
        "empty_pressure": "empty comment pattern",
    }

    reasons: list[str] = []
    for _, row in result.iterrows():
        ranked = sorted(pressure_columns, key=lambda col: float(row[col]), reverse=True)
        selected = [labels[col] for col in ranked if float(row[col]) >= 0.70][:3]
        if not selected:
            selected = [labels[ranked[0]]]
        reasons.append("; ".join(selected))
    result["reason_codes"] = reasons
    return result


def product_scores(
    processed: pd.DataFrame,
    features: list[str],
    model: Pipeline,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoring = processed.copy()
    scoring_features = scoring[features].apply(pd.to_numeric, errors="coerce")
    scoring["fake_probability"] = probability_for_positive(model, scoring_features)
    scoring["is_high_risk_review"] = scoring["fake_probability"].ge(args.high_risk_threshold)

    optional_mean_features = [
        "product_duplicate_comment_rate",
        "product_empty_comment_rate",
        "shop_burst_ratio",
        "product_review_velocity_per_day",
        "product_mean_gap_hours",
        "product_active_days",
        "product_image_review_share_proxy",
    ]
    for column in optional_mean_features:
        if column not in scoring.columns:
            scoring[column] = 0.0
        scoring[column] = pd.to_numeric(scoring[column], errors="coerce").fillna(0)

    grouped = scoring.groupby(args.entity_column, dropna=False)
    product = grouped.agg(
        shop_id=(args.shop_column, mode_or_first),
        product_title=(args.title_column, mode_or_first),
        review_count=("cmtid", "size"),
        mean_fake_probability=("fake_probability", "mean"),
        median_fake_probability=("fake_probability", "median"),
        max_fake_probability=("fake_probability", "max"),
        high_risk_review_count=("is_high_risk_review", "sum"),
        high_risk_review_share=("is_high_risk_review", "mean"),
        product_duplicate_comment_rate=("product_duplicate_comment_rate", "mean"),
        product_empty_comment_rate=("product_empty_comment_rate", "mean"),
        shop_burst_ratio=("shop_burst_ratio", "mean"),
        product_review_velocity_per_day=("product_review_velocity_per_day", "mean"),
        product_mean_gap_hours=("product_mean_gap_hours", "mean"),
        product_active_days=("product_active_days", "mean"),
        product_image_review_share_proxy=("product_image_review_share_proxy", "mean"),
    ).reset_index()

    if product[args.entity_column].duplicated().any():
        raise ValueError("Product score output has duplicate entity rows.")

    product["review_count_pressure"] = percentile(product["review_count"])
    product["burst_pressure"] = percentile(product["shop_burst_ratio"])
    product["duplicate_pressure"] = percentile(product["product_duplicate_comment_rate"])
    product["empty_pressure"] = percentile(product["product_empty_comment_rate"])
    product["suspicious_score"] = (
        0.45 * product["mean_fake_probability"]
        + 0.25 * product["high_risk_review_share"]
        + 0.10 * product["review_count_pressure"]
        + 0.08 * product["burst_pressure"]
        + 0.07 * product["duplicate_pressure"]
        + 0.05 * product["empty_pressure"]
    ).round(4)
    product = add_reason_codes(product)
    product = product.sort_values(
        ["suspicious_score", "mean_fake_probability", "high_risk_review_share", "review_count"],
        ascending=False,
    ).reset_index(drop=True)
    product["rank"] = np.arange(1, len(product) + 1)
    top = product.head(args.top_n).copy()
    return product, top


def feature_importance_table(
    model: Pipeline,
    labelled: pd.DataFrame,
    features: list[str],
    feature_df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    x_frame = labelled[features].apply(pd.to_numeric, errors="coerce")
    y = labelled["label"].astype(int)
    clf = model.named_steps["clf"]
    xgb_importance = getattr(clf, "feature_importances_", np.zeros(len(features)))
    permutation = permutation_importance(
        model,
        x_frame,
        y,
        n_repeats=10,
        random_state=seed,
        scoring="average_precision",
        n_jobs=1,
    )
    result = pd.DataFrame(
        {
            "feature": features,
            "xgb_importance": xgb_importance,
            "permutation_importance_mean": permutation.importances_mean,
            "permutation_importance_std": permutation.importances_std,
        }
    )
    selected_context = feature_df.drop_duplicates("feature")
    result = result.merge(
        selected_context[
            [
                "feature",
                "group",
                "consensus_rank",
                "mutual_info",
                "rf_importance_mean",
                "permutation_importance_mean",
            ]
        ].rename(columns={"permutation_importance_mean": "selection_permutation_importance_mean"}),
        on="feature",
        how="left",
    )
    return result.sort_values(
        ["xgb_importance", "permutation_importance_mean", "consensus_rank"],
        ascending=[False, False, True],
    )


def save_outputs(
    args: argparse.Namespace,
    root: Path,
    labelled: pd.DataFrame,
    processed: pd.DataFrame,
    features: list[str],
    feature_df: pd.DataFrame,
    cv_results: dict[str, Any],
    xgb_oof_prob: np.ndarray,
    xgb_oof_pred: np.ndarray,
) -> None:
    reports_dir = root / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    x_frame = labelled[features].apply(pd.to_numeric, errors="coerce")
    y = labelled["label"].astype(int)
    weights = source_sample_weight(labelled)
    final_model = build_models(args.seed)["xgboost"]
    fit_with_optional_weight(final_model, x_frame, y, weights)

    product, top = product_scores(processed, features, final_model, args)
    importance = feature_importance_table(final_model, labelled, features, feature_df, args.seed)

    metrics = {
        "label_counts": {ID_TO_LABEL[int(k)]: int(v) for k, v in y.value_counts().sort_index().items()},
        "label_source_counts": {str(k): int(v) for k, v in labelled["label_source"].value_counts().items()},
        "source_weights": DEFAULT_SOURCE_WEIGHTS,
        "high_risk_threshold": args.high_risk_threshold,
        "selected_feature_count": len(features),
        "processed_review_count": int(len(processed)),
        "product_count": int(product[args.entity_column].nunique()),
        "review_level_cv": cv_results,
    }

    (reports_dir / "review_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (reports_dir / "review_classification_report.txt").write_text(
        classification_report(y, xgb_oof_pred, target_names=["original", "fake"], zero_division=0),
        encoding="utf-8",
    )
    pd.DataFrame(
        confusion_matrix(y, xgb_oof_pred),
        index=["true_original", "true_fake"],
        columns=["pred_original", "pred_fake"],
    ).to_csv(reports_dir / "review_confusion_matrix.csv")
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)
    product.to_csv(reports_dir / "product_suspicious_scores.csv", index=False)
    top.to_csv(reports_dir / "top_suspicious_products.csv", index=False)

    summary = {
        "objective": "Rank products for manual audit using review-level fake probabilities.",
        "primary_model": "XGBoost tabular classifier",
        "baseline_model": "Logistic Regression",
        "positive_class": "fake",
        "entity_column": args.entity_column,
        "inputs": {
            "labelled_path": args.labelled_path,
            "processed_path": args.processed_path,
            "selected_features_path": args.selected_features_path,
        },
        "outputs": {
            "metrics": str(reports_dir / "review_model_metrics.json"),
            "classification_report": str(reports_dir / "review_classification_report.txt"),
            "confusion_matrix": str(reports_dir / "review_confusion_matrix.csv"),
            "feature_importance": str(reports_dir / "feature_importance.csv"),
            "product_scores": str(reports_dir / "product_suspicious_scores.csv"),
            "top_products": str(reports_dir / "top_suspicious_products.csv"),
        },
        "top_product_preview": top[
            [
                "rank",
                args.entity_column,
                "shop_id",
                "review_count",
                "mean_fake_probability",
                "high_risk_review_share",
                "suspicious_score",
                "reason_codes",
            ]
        ].head(10).to_dict(orient="records"),
    }
    (reports_dir / "modelling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    root = project_root()
    labelled, processed, features, feature_df = load_inputs(args, root)
    cv_results, xgb_oof_prob, xgb_oof_pred = cv_evaluations(labelled, processed, features, args)
    save_outputs(args, root, labelled, processed, features, feature_df, cv_results, xgb_oof_prob, xgb_oof_pred)


if __name__ == "__main__":
    main()
