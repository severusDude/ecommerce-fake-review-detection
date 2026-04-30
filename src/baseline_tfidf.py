import argparse
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


LABEL_MAP = {"original": 0, "fake": 1}
ID_TO_LABEL = {0: "original", 1: "fake"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-IDF Logistic Regression baseline for Shopee fake review detection.")
    parser.add_argument("--data-path", default="data/labels/review_shopee_labelled.csv", help="Path to labeled CSV.")
    parser.add_argument("--processed-path", default="data/processed/review_shopee_processed.csv", help="Processed CSV used to recover comment text by cmtid.")
    parser.add_argument("--data-url", default="", help="Optional URL used when --data-path is missing.")
    parser.add_argument("--text-column", default="comment")
    parser.add_argument("--label-column", default="fakeornot")
    parser.add_argument("--reports-dir", default="reports/tfidf_baseline")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true", help="Fail if CSV does not exist locally.")
    return parser.parse_args()


def ensure_data(path: Path, url: str, no_download: bool) -> None:
    if path.exists():
        return
    if no_download or not url:
        raise FileNotFoundError(f"Missing dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " "))
    return text.strip()


def require_columns(df: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def label_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts().sort_index()
    return {ID_TO_LABEL[int(label)]: int(count) for label, count in counts.items()}


def merge_comment_text(df: pd.DataFrame, processed_path: Path, text_column: str) -> pd.DataFrame:
    require_columns(df, {"cmtid"}, "labelled data")
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed dataset: {processed_path}")

    labelled = df.copy()
    labelled["cmtid"] = labelled["cmtid"].fillna("").astype(str).str.strip()
    if labelled["cmtid"].duplicated().any():
        raise ValueError("Labelled data has duplicate cmtid values.")

    processed = pd.read_csv(processed_path, dtype={"cmtid": str}, keep_default_na=False)
    require_columns(processed, {"cmtid", text_column}, "processed data")
    processed["cmtid"] = processed["cmtid"].fillna("").astype(str).str.strip()
    if processed["cmtid"].duplicated().any():
        raise ValueError("Processed data has duplicate cmtid values.")

    merged = labelled.merge(processed[["cmtid", text_column]], on="cmtid", how="left", validate="one_to_one")
    if merged[text_column].isna().any():
        missing_count = int(merged[text_column].isna().sum())
        raise ValueError(f"{missing_count} labelled rows could not be matched to processed comments.")
    return merged


def load_dataset(path: Path, processed_path: Path, text_column: str, label_column: str) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(path, dtype={"cmtid": str}, keep_default_na=False)
    require_columns(df, {label_column, "label_source", "confidence"}, "labelled data")
    if text_column not in df.columns:
        df = merge_comment_text(df, processed_path, text_column)

    require_columns(df, {text_column, label_column}, "training data")
    df = df[[text_column, label_column, "label_source", "confidence"]].copy()
    df[text_column] = df[text_column].map(clean_text)
    df[label_column] = df[label_column].astype(str).str.strip().str.lower()
    df = df[df[label_column].isin(LABEL_MAP)]
    df["label"] = df[label_column].map(LABEL_MAP).astype(int)

    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both labels: fake and original.")
    counts = label_counts(df["label"])
    if counts.get("original") != counts.get("fake"):
        raise ValueError(f"Expected balanced labels, found {counts}.")

    metadata = {
        "data_path": str(path),
        "processed_path": str(processed_path),
        "text_column": text_column,
        "label_column": label_column,
        "empty_text_count": int(df[text_column].eq("").sum()),
        "label_counts": counts,
        "label_source_counts": {str(k): int(v) for k, v in df["label_source"].value_counts().items()},
    }
    return df, metadata


def split_dataset(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    relative_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val["label"],
        random_state=seed,
    )
    return train, val, test


def split_label_counts(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        "train": label_counts(train["label"]),
        "val": label_counts(val["label"]),
        "test": label_counts(test["label"]),
    }


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    reports_dir = Path(args.reports_dir)

    ensure_data(data_path, args.data_url, args.no_download)
    processed_path = Path(args.processed_path)

    df, dataset_metadata = load_dataset(data_path, processed_path, args.text_column, args.label_column)
    train, val, test = split_dataset(df, args.test_size, args.val_size, args.seed)

    train_for_fit = pd.concat([train, val], ignore_index=True)
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=30000)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)),
        ]
    )
    model.fit(train_for_fit[args.text_column], train_for_fit["label"])

    predictions = model.predict(test[args.text_column])
    metrics = {
        "accuracy": accuracy_score(test["label"], predictions),
        "precision": precision_score(test["label"], predictions, zero_division=0),
        "recall": recall_score(test["label"], predictions, zero_division=0),
        "f1": f1_score(test["label"], predictions, zero_division=0),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_total": int(len(df)),
        "label_counts": dataset_metadata["label_counts"],
        "split_label_counts": split_label_counts(train, val, test),
        "dataset": dataset_metadata,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (reports_dir / "classification_report.txt").write_text(
        classification_report(test["label"], predictions, target_names=["original", "fake"], zero_division=0),
        encoding="utf-8",
    )
    pd.DataFrame(confusion_matrix(test["label"], predictions), index=["true_original", "true_fake"], columns=["pred_original", "pred_fake"]).to_csv(
        reports_dir / "confusion_matrix.csv"
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
