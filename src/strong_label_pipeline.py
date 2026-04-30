import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_STATE = 42
LABEL_MAP = {"original": 0, "fake": 1}
LABEL_NAME = {0: "original", 1: "fake"}
TARGET_FINAL_PER_CLASS = 250
TARGET_HUMAN_PER_CLASS = 125
TARGET_FINAL_TOTAL = 500
TARGET_HUMAN_TOTAL = 250
PSEUDO_LOW = 0.05
PSEUDO_HIGH = 0.95

FEATURE_GROUPS = {
    "comment_char_count": "review",
    "comment_word_count": "review",
    "comment_unique_word_ratio": "review",
    "empty_comment_flag": "review",
    "duplicate_comment_flag": "review",
    "repeated_char_run_max": "review",
    "rating_star_num": "review",
    "rating_deviation_from_product_mean": "review",
    "rating_abs_deviation_from_product_mean": "review",
    "edit_delay_hours": "review",
    "image_review_share_proxy": "review",
    "review_hour": "temporal",
    "review_dayofweek": "temporal",
    "user_prev_gap_hours": "temporal",
    "product_prev_gap_hours": "temporal",
    "shop_prev_gap_hours": "temporal",
    "user_reviews_same_day": "temporal",
    "product_reviews_same_day": "temporal",
    "shop_reviews_same_day": "temporal",
    "user_review_count": "user",
    "user_rating_mean": "user",
    "user_rating_std": "user",
    "user_active_days": "user",
    "user_review_span_days": "user",
    "user_review_velocity_per_day": "user",
    "user_mean_gap_hours": "user",
    "user_unique_products": "user",
    "user_unique_shops": "user",
    "user_empty_comment_rate": "user",
    "user_duplicate_comment_rate": "user",
    "product_review_count": "product",
    "product_rating_mean": "product",
    "product_rating_std": "product",
    "product_rating_var": "product",
    "product_five_star_share": "product",
    "product_active_days": "product",
    "product_review_span_days": "product",
    "product_review_velocity_per_day": "product",
    "product_mean_gap_hours": "product",
    "product_unique_users": "product",
    "product_unique_shops": "product",
    "product_empty_comment_rate": "product",
    "product_duplicate_comment_rate": "product",
    "product_image_review_share_proxy": "product",
    "product_max_reviews_per_day": "product",
    "product_burst_ratio": "product",
    "shop_review_count": "shop",
    "shop_rating_mean": "shop",
    "shop_rating_std": "shop",
    "shop_rating_var": "shop",
    "shop_five_star_share": "shop",
    "shop_active_days": "shop",
    "shop_review_span_days": "shop",
    "shop_review_velocity_per_day": "shop",
    "shop_mean_gap_hours": "shop",
    "shop_unique_products": "shop",
    "shop_unique_shops": "shop",
    "shop_empty_comment_rate": "shop",
    "shop_duplicate_comment_rate": "shop",
    "shop_image_review_share_proxy": "shop",
    "shop_max_reviews_per_day": "shop",
    "shop_burst_ratio": "shop",
}


def project_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError(
        "Project root with pyproject.toml and data/ not found.")


def xgb_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: xgboost. Install with `uv sync` or `pip install xgboost>=2.0`.") from exc

    return XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def read_processed(root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "review_shopee_processed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed CSV: {path}")
    df = pd.read_csv(path, dtype={"cmtid": str, "userid": str,
                     "item_id": str, "shop_id": str}, keep_default_na=False)
    df["cmtid"] = normalize_cmtid(df["cmtid"])
    if df["cmtid"].duplicated().any():
        raise ValueError("Processed data has duplicate cmtid.")
    return df


def normalize_cmtid(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def normalize_label(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if text in LABEL_MAP:
        return text
    if text in {"0", "0.0"}:
        return "original"
    if text in {"1", "1.0"}:
        return "fake"
    raise ValueError(f"Invalid label value: {value!r}")


def candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in FEATURE_GROUPS if column in df.columns]


def labeled_seed(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df[df["is_labeled"].astype(str).str.lower().eq("true")].copy()
    labeled["fakeornot"] = labeled["fakeornot"].map(normalize_label)
    labeled["label"] = labeled["fakeornot"].map(LABEL_MAP).astype(int)
    labels = labeled[["cmtid", "fakeornot", "label"]].copy()
    labels["label_source"] = "human_seed"
    labels["confidence"] = 1.0
    labels["human_round"] = 0
    labels["uncertainty_score"] = np.nan
    labels["pseudo_probability"] = np.nan
    return labels


def active_labels_path(root: Path) -> Path:
    return root / "data" / "labels" / "active_learning" / "human_labels.csv"


def pseudo_labels_path(root: Path) -> Path:
    return root / "data" / "labels" / "pseudo_xgb_labels.csv"


def spread_labels_path(root: Path) -> Path:
    return root / "data" / "labels" / "label_spreading_labels.csv"


def final_labels_path(root: Path) -> Path:
    return root / "data" / "labels" / "review_shopee_labelled.csv"


def reports_dir(root: Path) -> Path:
    return root / "reports" / "strong_labels"


def selected_features_path(root: Path) -> Path:
    return reports_dir(root) / "selected_features.csv"


def load_active_labels(root: Path) -> pd.DataFrame:
    path = active_labels_path(root)
    if not path.exists():
        return empty_label_frame()
    labels = pd.read_csv(path, dtype={"cmtid": str}, keep_default_na=False)
    return normalize_label_frame(labels, default_source="human_active")


def empty_label_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cmtid",
            "fakeornot",
            "label",
            "label_source",
            "confidence",
            "human_round",
            "uncertainty_score",
            "pseudo_probability",
        ]
    )


def normalize_label_frame(df: pd.DataFrame, default_source: str) -> pd.DataFrame:
    result = df.copy()
    if result.empty:
        return empty_label_frame()
    result["cmtid"] = normalize_cmtid(result["cmtid"])
    if "fakeornot" not in result.columns and "label" in result.columns:
        result["fakeornot"] = result["label"].map(normalize_label)
    result["fakeornot"] = result["fakeornot"].map(normalize_label)
    result["label"] = result["fakeornot"].map(LABEL_MAP).astype(int)
    if "label_source" not in result.columns:
        result["label_source"] = default_source
    if "confidence" not in result.columns:
        result["confidence"] = 1.0
    if "human_round" not in result.columns:
        result["human_round"] = np.nan
    if "uncertainty_score" not in result.columns:
        result["uncertainty_score"] = np.nan
    if "pseudo_probability" not in result.columns:
        result["pseudo_probability"] = np.nan
    result = result[
        ["cmtid", "fakeornot", "label", "label_source", "confidence",
            "human_round", "uncertainty_score", "pseudo_probability"]
    ].copy()
    conflicting = result.groupby("cmtid")["label"].nunique()
    conflicts = conflicting[conflicting > 1]
    if not conflicts.empty:
        raise ValueError(
            f"Conflicting labels for cmtid: {conflicts.index.tolist()[:10]}")
    return result.drop_duplicates("cmtid", keep="first")


def load_human_labels(root: Path, df: pd.DataFrame) -> pd.DataFrame:
    seed = labeled_seed(df)
    active = load_active_labels(root)
    combined = pd.concat([seed, active], ignore_index=True)
    combined = normalize_label_frame(combined, default_source="human_active")
    seed_ids = set(seed["cmtid"])
    active_seed_overlap = active[active["cmtid"].isin(seed_ids)]
    if not active_seed_overlap.empty:
        merged = active_seed_overlap.merge(
            seed[["cmtid", "label"]], on="cmtid", suffixes=("_active", "_seed"))
        if (merged["label_active"] != merged["label_seed"]).any():
            raise ValueError("Active labels conflict with seed labels.")
    return combined.drop_duplicates("cmtid", keep="first")


def validate_seed(df: pd.DataFrame) -> None:
    if len(df) != 4630:
        raise AssertionError(f"Expected 4,630 processed rows, got {len(df)}")
    seed = labeled_seed(df)
    counts = seed["fakeornot"].value_counts().to_dict()
    if len(seed) != 100:
        raise AssertionError(f"Expected 100 seed labels, got {len(seed)}")
    if counts != {"fake": 51, "original": 49}:
        raise AssertionError(
            f"Expected seed balance 51 fake / 49 original, got {counts}")


def feature_matrix(df: pd.DataFrame, features: list[str], imputer: SimpleImputer | None = None):
    x_raw = df[features].apply(pd.to_numeric, errors="coerce")
    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        values = imputer.fit_transform(x_raw)
    else:
        values = imputer.transform(x_raw)
    return pd.DataFrame(values, columns=features, index=df.index), imputer


def select_features(df: pd.DataFrame, labels: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    features = candidate_feature_columns(df)
    merged = labels[["cmtid", "label"]].merge(
        df[["cmtid", *features]], on="cmtid", how="left", validate="one_to_one")
    y = merged["label"].astype(int)
    if y.nunique() != 2:
        raise ValueError("Feature selection needs both classes.")
    x_raw = merged[features].apply(pd.to_numeric, errors="coerce")
    missing_rate = x_raw.isna().mean()
    missing_keep = missing_rate[missing_rate <= 0.40].index.tolist()
    x_missing = x_raw[missing_keep]
    imputer = SimpleImputer(strategy="median")
    x_imputed = pd.DataFrame(imputer.fit_transform(
        x_missing), columns=missing_keep, index=x_missing.index)
    variance = x_imputed.var(axis=0)
    variance_keep = variance[variance > 0].index.tolist()
    x_var = x_imputed[variance_keep]
    mi_values = mutual_info_classif(x_var, y, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi_values, index=x_var.columns, name="mutual_info")

    corr = x_var.corr(method="spearman").abs().fillna(0)
    to_drop: set[str] = set()
    columns = list(x_var.columns)
    for left_idx, left_col in enumerate(columns):
        if left_col in to_drop:
            continue
        for right_col in columns[left_idx + 1:]:
            if right_col in to_drop:
                continue
            if corr.loc[left_col, right_col] > 0.90:
                left_mi = mi_series.get(left_col, 0)
                right_mi = mi_series.get(right_col, 0)
                drop_col = right_col if left_mi >= right_mi else left_col
                to_drop.add(drop_col)
                if drop_col == left_col:
                    break

    x_pool = x_var.drop(columns=sorted(to_drop))
    model = xgb_classifier()
    weights = compute_sample_weight(class_weight="balanced", y=y)
    model.fit(x_pool, y, sample_weight=weights)

    selected = pd.DataFrame(
        {
            "feature": x_pool.columns,
            "group": [FEATURE_GROUPS.get(column, "") for column in x_pool.columns],
            "missing_rate": missing_rate.reindex(x_pool.columns).fillna(0).values,
            "variance": variance.reindex(x_pool.columns).fillna(0).values,
            "mutual_info": mi_series.reindex(x_pool.columns).fillna(0).values,
            "xgb_importance": model.feature_importances_,
        }
    )
    selected["mutual_info_rank"] = selected["mutual_info"].rank(
        ascending=False, method="average")
    selected["xgb_importance_rank"] = selected["xgb_importance"].rank(
        ascending=False, method="average")
    selected["consensus_rank_score"] = selected[[
        "mutual_info_rank", "xgb_importance_rank"]].mean(axis=1)
    selected["consensus_rank"] = selected["consensus_rank_score"].rank(
        ascending=True, method="first").astype(int)
    return selected.sort_values(["consensus_rank", "xgb_importance_rank"]).head(top_n).reset_index(drop=True)


def load_or_select_features(root: Path, df: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
    path = selected_features_path(root)
    if path.exists():
        selected = pd.read_csv(path)
        features = [feature for feature in selected["feature"].astype(
            str).tolist() if feature in df.columns]
        if features:
            return features
    selected = select_features(df, labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(path, index=False)
    return selected["feature"].tolist()


def train_model(df: pd.DataFrame, labels: pd.DataFrame, features: list[str]):
    train = labels[["cmtid", "label"]].merge(
        df[["cmtid", *features]], on="cmtid", how="left", validate="one_to_one")
    x, imputer = feature_matrix(train, features)
    y = train["label"].astype(int)
    model = xgb_classifier()
    weights = compute_sample_weight(class_weight="balanced", y=y)
    model.fit(x, y, sample_weight=weights)
    return model, imputer


def predict_unlabeled(df: pd.DataFrame, labels: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    model, imputer = train_model(df, labels, features)
    x_all, _ = feature_matrix(df, features, imputer=imputer)
    p_fake = model.predict_proba(x_all)[:, 1]
    scored = df[["cmtid", "comment", "product_title", "rating_star",
                 "review_datetime", "userid", "item_id", "shop_id"]].copy()
    scored["p_fake"] = p_fake
    scored["predicted_label"] = np.where(
        scored["p_fake"] >= 0.5, "fake", "original")
    scored["uncertainty_score"] = (scored["p_fake"] - 0.5).abs()
    labeled_ids = set(labels["cmtid"])
    return scored[~scored["cmtid"].isin(labeled_ids)].copy()


def command_baseline(args: argparse.Namespace) -> None:
    root = project_root()
    df = read_processed(root)
    validate_seed(df)
    seed = labeled_seed(df)
    out_dir = reports_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = select_features(df, seed, top_n=args.top_features)
    selected.to_csv(selected_features_path(root), index=False)
    features = selected["feature"].tolist()
    train = seed[["cmtid", "label"]].merge(
        df[["cmtid", *features]], on="cmtid", how="left", validate="one_to_one")
    x, _ = feature_matrix(train, features)
    y = train["label"].astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    importance_rows = []
    for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y), start=1):
        model = xgb_classifier()
        weights = compute_sample_weight(
            class_weight="balanced", y=y.iloc[train_idx])
        model.fit(x.iloc[train_idx], y.iloc[train_idx], sample_weight=weights)
        proba = model.predict_proba(x.iloc[valid_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        y_valid = y.iloc[valid_idx]
        rows.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(y_valid, pred),
                "precision": precision_score(y_valid, pred, zero_division=0),
                "recall": recall_score(y_valid, pred, zero_division=0),
                "f1": f1_score(y_valid, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_valid, proba),
                "pr_auc": average_precision_score(y_valid, proba),
            }
        )
        importance_rows.append(
            pd.Series(model.feature_importances_, index=features, name=fold))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "xgb_baseline_cv_metrics.csv", index=False)
    metrics.drop(columns="fold").agg(["mean", "std"]).to_csv(
        out_dir / "xgb_baseline_cv_summary.csv")
    seed["fakeornot"].value_counts().rename_axis("fakeornot").reset_index(name="count").to_csv(
        out_dir / "xgb_baseline_class_balance.csv", index=False
    )
    importance = pd.DataFrame(importance_rows).mean(
        axis=0).sort_values(ascending=False)
    importance.rename_axis("feature").reset_index(name="xgb_importance_mean").to_csv(
        out_dir / "xgb_feature_importance.csv", index=False
    )
    print(json.dumps({"reports_dir": str(out_dir),
          "selected_features": features}, indent=2))


def allocate_round_counts(deficits: dict[int, int], batch_size: int) -> dict[int, int]:
    total_deficit = sum(max(value, 0) for value in deficits.values())
    if total_deficit <= 0:
        return {0: 0, 1: 0}
    counts = {}
    remaining = batch_size
    for label in [0, 1]:
        share = deficits[label] / total_deficit if total_deficit else 0
        counts[label] = min(deficits[label] * 2, max(1 if deficits[label]
                            > 0 else 0, math.floor(batch_size * share)))
        remaining -= counts[label]
    while remaining > 0:
        label = max(
            deficits, key=lambda item: deficits[item] - counts[item] / 2)
        counts[label] += 1
        remaining -= 1
    return counts


def command_make_round(args: argparse.Namespace) -> None:
    root = project_root()
    df = read_processed(root)
    validate_seed(df)
    human = load_human_labels(root, df)
    features = load_or_select_features(root, df, human)
    counts = human["label"].value_counts().to_dict()
    deficits = {label: max(0, TARGET_HUMAN_PER_CLASS -
                           int(counts.get(label, 0))) for label in [0, 1]}
    if sum(deficits.values()) <= 0:
        print("Human label target already satisfied.")
        return

    batch_size = args.batch_size
    if args.round >= 3:
        batch_size = max(batch_size, sum(deficits.values()))
    scored = predict_unlabeled(df, human, features)
    allocation = allocate_round_counts(deficits, batch_size)
    selections = []
    for label in [0, 1]:
        predicted_name = LABEL_NAME[label]
        class_pool = scored[scored["predicted_label"].eq(
            predicted_name)].sort_values("uncertainty_score")
        selections.append(class_pool.head(allocation[label]))
    selected = pd.concat(
        selections, ignore_index=True).drop_duplicates("cmtid")
    if len(selected) < batch_size:
        extra = scored[~scored["cmtid"].isin(selected["cmtid"])].sort_values(
            "uncertainty_score").head(batch_size - len(selected))
        selected = pd.concat(
            [selected, extra], ignore_index=True).drop_duplicates("cmtid")

    out_dir = root / "data" / "labels" / "label_studio" / f"round_{args.round}"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / "candidates.csv"
    task_path = out_dir / "tasks.jsonl"
    selected.to_csv(candidate_path, index=False)
    with task_path.open("w", encoding="utf-8") as handle:
        for _, row in selected.iterrows():
            task = {
                "data": {
                    "cmtid": row["cmtid"],
                    "comment": row.get("comment", ""),
                    "product_title": row.get("product_title", ""),
                    "rating_star": row.get("rating_star", ""),
                    "review_datetime": row.get("review_datetime", ""),
                    "userid": row.get("userid", ""),
                    "item_id": row.get("item_id", ""),
                    "shop_id": row.get("shop_id", ""),
                    "model_predicted_label": row["predicted_label"],
                    "model_p_fake": float(row["p_fake"]),
                    "uncertainty_score": float(row["uncertainty_score"]),
                    "active_learning_round": int(args.round),
                }
            }
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(json.dumps({"tasks": str(task_path), "candidates": str(
        candidate_path), "count": int(len(selected))}, indent=2))


def load_label_studio_export(path: Path, round_number: int) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path, dtype={"cmtid": str}, keep_default_na=False)
        label_col = next((col for col in [
                         "fakeornot", "label", "choice", "choices"] if col in raw.columns), None)
        if label_col is None:
            raise ValueError(
                "CSV export needs one label column: fakeornot, label, choice, or choices.")
        labels = raw[["cmtid", label_col]].copy()
        labels = labels.rename(columns={label_col: "fakeornot"})
    else:
        text = path.read_text(encoding="utf-8")
        records = [json.loads(line) for line in text.splitlines() if line.strip(
        )] if path.suffix.lower() == ".jsonl" else json.loads(text)
        labels = []
        for record in records:
            data = record.get("data", {})
            cmtid = data.get("cmtid") or record.get("cmtid")
            choices = []
            for annotation in record.get("annotations", []):
                for result in annotation.get("result", []):
                    value = result.get("value", {})
                    choices.extend(value.get("choices", []))
                    if "labels" in value:
                        choices.extend(value["labels"])
            if not choices and "fakeornot" in record:
                choices = [record["fakeornot"]]
            if not cmtid or not choices:
                continue
            labels.append({"cmtid": cmtid, "fakeornot": choices[0]})
        labels = pd.DataFrame(labels)
    if labels.empty:
        raise ValueError(f"No labels found in export: {path}")
    labels["cmtid"] = normalize_cmtid(labels["cmtid"])
    labels["fakeornot"] = labels["fakeornot"].map(normalize_label)
    labels["label"] = labels["fakeornot"].map(LABEL_MAP).astype(int)
    labels["label_source"] = "human_active"
    labels["confidence"] = 1.0
    labels["human_round"] = round_number
    labels["uncertainty_score"] = np.nan
    labels["pseudo_probability"] = np.nan
    return labels


def round_candidate_metadata(root: Path) -> pd.DataFrame:
    paths = sorted((root / "data" / "labels" /
                   "label_studio").glob("round_*/candidates.csv"))
    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype={"cmtid": str}, keep_default_na=False)
        if "uncertainty_score" in frame.columns:
            frames.append(frame[["cmtid", "uncertainty_score"]])
    if not frames:
        return pd.DataFrame(columns=["cmtid", "uncertainty_score"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("cmtid", keep="last")


def command_ingest_labels(args: argparse.Namespace) -> None:
    root = project_root()
    labels = load_label_studio_export(Path(args.export_path), args.round)
    metadata = round_candidate_metadata(root)
    if not metadata.empty:
        labels = labels.drop(columns=["uncertainty_score"]).merge(
            metadata, on="cmtid", how="left")
    existing = load_active_labels(root)
    combined = pd.concat([existing, labels], ignore_index=True)
    combined = normalize_label_frame(combined, default_source="human_active")
    path = active_labels_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    print(json.dumps({"active_labels": str(path),
          "count": int(len(combined))}, indent=2))


def select_balanced_human(root: Path, df: pd.DataFrame) -> pd.DataFrame:
    human = load_human_labels(root, df)
    seed = human[human["label_source"].eq("human_seed")]
    active = human[human["label_source"].eq("human_active")].copy()
    selected_parts = []
    for label in [0, 1]:
        seed_part = seed[seed["label"].eq(label)]
        need = TARGET_HUMAN_PER_CLASS - len(seed_part)
        if need < 0:
            raise ValueError("Seed labels exceed human target for one class.")
        active_part = active[active["label"].eq(label)].copy()
        active_part["human_round_num"] = pd.to_numeric(
            active_part["human_round"], errors="coerce").fillna(999)
        active_part["uncertainty_sort"] = pd.to_numeric(
            active_part["uncertainty_score"], errors="coerce").fillna(1.0)
        active_part = active_part.sort_values(
            ["human_round_num", "uncertainty_sort", "cmtid"])
        if len(active_part) < need:
            raise ValueError(
                f"Need {need} more human {LABEL_NAME[label]} labels; only {len(active_part)} available.")
        selected_parts.append(seed_part)
        selected_parts.append(active_part.head(need).drop(
            columns=["human_round_num", "uncertainty_sort"]))
    selected = pd.concat(selected_parts, ignore_index=True)
    assert len(selected) == TARGET_HUMAN_TOTAL
    assert selected["label"].value_counts().to_dict(
    ) == {0: TARGET_HUMAN_PER_CLASS, 1: TARGET_HUMAN_PER_CLASS}
    return selected


def command_pseudo_label(args: argparse.Namespace) -> None:
    root = project_root()
    df = read_processed(root)
    validate_seed(df)
    human = select_balanced_human(root, df)
    features = load_or_select_features(root, df, human)
    scored = predict_unlabeled(df, human, features)
    accepted = scored[(scored["p_fake"] >= args.high_threshold) | (
        scored["p_fake"] <= args.low_threshold)].copy()
    accepted["label"] = (accepted["p_fake"] >= args.high_threshold).astype(int)
    accepted["fakeornot"] = accepted["label"].map(LABEL_NAME)
    accepted["label_source"] = "pseudo_xgb"
    accepted["confidence"] = np.where(accepted["label"].eq(
        1), accepted["p_fake"], 1 - accepted["p_fake"])
    accepted["human_round"] = np.nan
    accepted["pseudo_probability"] = accepted["p_fake"]
    parts = []
    for label in [0, 1]:
        part = accepted[accepted["label"].eq(label)].sort_values(
            "confidence", ascending=False).head(TARGET_FINAL_PER_CLASS - TARGET_HUMAN_PER_CLASS)
        parts.append(part)
    labels = pd.concat(parts, ignore_index=True)
    labels = labels[
        ["cmtid", "fakeornot", "label", "label_source", "confidence",
            "human_round", "uncertainty_score", "pseudo_probability"]
    ]
    path = pseudo_labels_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(path, index=False)
    print(json.dumps({"pseudo_labels": str(path),
          "count": int(len(labels))}, indent=2))


def load_optional_labels(path: Path, default_source: str) -> pd.DataFrame:
    if not path.exists():
        return empty_label_frame()
    return normalize_label_frame(pd.read_csv(path, dtype={"cmtid": str}, keep_default_na=False), default_source=default_source)


def command_label_spread(args: argparse.Namespace) -> None:
    root = project_root()
    df = read_processed(root)
    validate_seed(df)
    human = select_balanced_human(root, df)
    pseudo = load_optional_labels(
        pseudo_labels_path(root), default_source="pseudo_xgb")
    features = load_or_select_features(root, df, human)
    seed = pd.concat([human, pseudo], ignore_index=True).drop_duplicates(
        "cmtid", keep="first")
    x, _ = feature_matrix(df, features)
    x_scaled = StandardScaler().fit_transform(x)
    y = pd.Series(-1, index=df.index, dtype=int)
    seed_map = seed.set_index("cmtid")["label"].to_dict()
    y.loc[df["cmtid"].map(seed_map).notna()] = df.loc[df["cmtid"].map(
        seed_map).notna(), "cmtid"].map(seed_map).astype(int)
    model = LabelSpreading(
        kernel="knn", n_neighbors=args.n_neighbors, alpha=args.alpha, max_iter=1000)
    model.fit(x_scaled, y)
    probabilities = model.label_distributions_
    spread = df[["cmtid"]].copy()
    spread["p_original_spread"] = probabilities[:, 0]
    spread["p_fake_spread"] = probabilities[:, 1]
    spread["label"] = probabilities.argmax(axis=1)
    spread["fakeornot"] = spread["label"].map(LABEL_NAME)
    spread["confidence"] = probabilities.max(axis=1)
    spread["label_source"] = "label_spreading"
    spread["human_round"] = np.nan
    spread["uncertainty_score"] = np.nan
    spread["pseudo_probability"] = np.nan
    used_ids = set(seed["cmtid"])
    spread = spread[~spread["cmtid"].isin(used_ids)].copy()

    selected_parts = []
    current_counts = seed["label"].value_counts().to_dict()
    for label in [0, 1]:
        need = TARGET_FINAL_PER_CLASS - int(current_counts.get(label, 0))
        if need < 0:
            need = 0
        part = spread[spread["label"].eq(label)].sort_values(
            "confidence", ascending=False).head(need)
        if len(part) < need:
            raise ValueError(
                f"Label spreading found only {len(part)} rows for {LABEL_NAME[label]}, need {need}.")
        selected_parts.append(part)
    labels = pd.concat(selected_parts, ignore_index=True)
    labels = labels[
        ["cmtid", "fakeornot", "label", "label_source", "confidence",
            "human_round", "uncertainty_score", "pseudo_probability"]
    ]
    path = spread_labels_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(path, index=False)
    print(json.dumps({"label_spreading_labels": str(
        path), "count": int(len(labels))}, indent=2))


def command_build_final(args: argparse.Namespace) -> None:
    root = project_root()
    df = read_processed(root)
    validate_seed(df)
    human = select_balanced_human(root, df)
    pseudo = load_optional_labels(
        pseudo_labels_path(root), default_source="pseudo_xgb")
    spread = load_optional_labels(spread_labels_path(
        root), default_source="label_spreading")
    features = load_or_select_features(root, df, human)

    selected_parts = [human]
    used_ids = set(human["cmtid"])
    for label in [0, 1]:
        pseudo_part = pseudo[(pseudo["label"].eq(label)) & (
            ~pseudo["cmtid"].isin(used_ids))].copy()
        pseudo_part["confidence"] = pd.to_numeric(
            pseudo_part["confidence"], errors="coerce").fillna(0)
        pseudo_part = pseudo_part.sort_values("confidence", ascending=False)
        selected = pseudo_part.head(
            TARGET_FINAL_PER_CLASS - TARGET_HUMAN_PER_CLASS)
        selected_parts.append(selected)
        used_ids.update(selected["cmtid"])

    partial = pd.concat(selected_parts, ignore_index=True)
    current_counts = partial["label"].value_counts().to_dict()
    spread_parts = []
    for label in [0, 1]:
        need = TARGET_FINAL_PER_CLASS - int(current_counts.get(label, 0))
        spread_part = spread[(spread["label"].eq(label)) & (
            ~spread["cmtid"].isin(used_ids))].copy()
        spread_part["confidence"] = pd.to_numeric(
            spread_part["confidence"], errors="coerce").fillna(0)
        spread_part = spread_part.sort_values(
            "confidence", ascending=False).head(need)
        if len(spread_part) < need:
            raise ValueError(
                f"Final build needs {need} more {LABEL_NAME[label]} labels from label spreading.")
        spread_parts.append(spread_part)
        used_ids.update(spread_part["cmtid"])

    final = pd.concat([partial, *spread_parts], ignore_index=True)
    if final["cmtid"].duplicated().any():
        raise AssertionError("Final labels contain duplicate cmtid.")
    if len(final) != TARGET_FINAL_TOTAL:
        raise AssertionError(f"Expected 500 final labels, got {len(final)}")
    if final["label"].value_counts().to_dict() != {0: TARGET_FINAL_PER_CLASS, 1: TARGET_FINAL_PER_CLASS}:
        raise AssertionError(
            f"Final labels not balanced: {final['label'].value_counts().to_dict()}")
    pseudo_final = final[final["label_source"].eq("pseudo_xgb")]
    if not pseudo_final.empty:
        probs = pd.to_numeric(
            pseudo_final["pseudo_probability"], errors="coerce")
        if not ((probs >= PSEUDO_HIGH) | (probs <= PSEUDO_LOW)).all():
            raise AssertionError("Pseudo labels violate confidence threshold.")

    final = final.merge(df[["cmtid", *features]],
                        on="cmtid", how="left", validate="one_to_one")
    final = final[
        ["cmtid", "fakeornot", "label", "label_source", "confidence",
            "human_round", "uncertainty_score", "pseudo_probability", *features]
    ]
    path = final_labels_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(path, index=False)
    print(json.dumps({"final_labels": str(path), "count": int(
        len(final)), "features": features}, indent=2))


def command_validate(args: argparse.Namespace) -> None:
    root = project_root()
    df = read_processed(root)
    validate_seed(df)
    if final_labels_path(root).exists():
        final = pd.read_csv(final_labels_path(root), dtype={
                            "cmtid": str}, keep_default_na=False)
        if final["cmtid"].duplicated().any():
            raise AssertionError("Final labels contain duplicate cmtid.")
        if len(final) != TARGET_FINAL_TOTAL:
            raise AssertionError(
                f"Expected 500 final labels, got {len(final)}")
        counts = final["fakeornot"].value_counts().to_dict()
        if counts != {"original": TARGET_FINAL_PER_CLASS, "fake": TARGET_FINAL_PER_CLASS}:
            raise AssertionError(f"Final labels not balanced: {counts}")
    print("Validation passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Balanced 500-label active learning and semi-supervised pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser(
        "baseline", help="Run 5-fold XGBoost baseline on 100 seed labels.")
    baseline.add_argument("--top-features", type=int, default=15)
    baseline.set_defaults(func=command_baseline)

    make_round = subparsers.add_parser(
        "make-round", help="Export uncertain Label Studio tasks for one active-learning round.")
    make_round.add_argument("--round", type=int, required=True)
    make_round.add_argument("--batch-size", type=int, default=50)
    make_round.set_defaults(func=command_make_round)

    ingest = subparsers.add_parser(
        "ingest-labels", help="Import Label Studio JSON/JSONL/CSV export.")
    ingest.add_argument("--round", type=int, required=True)
    ingest.add_argument("--export-path", required=True)
    ingest.set_defaults(func=command_ingest_labels)

    pseudo = subparsers.add_parser(
        "pseudo-label", help="Create high-confidence XGBoost pseudo labels.")
    pseudo.add_argument("--low-threshold", type=float, default=PSEUDO_LOW)
    pseudo.add_argument("--high-threshold", type=float, default=PSEUDO_HIGH)
    pseudo.set_defaults(func=command_pseudo_label)

    spread = subparsers.add_parser(
        "label-spread", help="Create remaining labels with sklearn LabelSpreading.")
    spread.add_argument("--n-neighbors", type=int, default=15)
    spread.add_argument("--alpha", type=float, default=0.2)
    spread.set_defaults(func=command_label_spread)

    final = subparsers.add_parser(
        "build-final", help="Build final balanced 500 strong labels CSV.")
    final.set_defaults(func=command_build_final)

    validate = subparsers.add_parser("validate", help="Run acceptance checks.")
    validate.set_defaults(func=command_validate)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
