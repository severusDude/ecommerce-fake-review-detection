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


DEFAULT_DATA_URL = "https://raw.githubusercontent.com/andrioktavianto/fake-review-shopee/master/train_review_only.csv"
LABEL_MAP = {"original": 0, "fake": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-IDF Logistic Regression baseline for Shopee fake review detection.")
    parser.add_argument("--data-path", default="data/train_review_only.csv", help="Path to labeled CSV.")
    parser.add_argument("--data-url", default=DEFAULT_DATA_URL, help="URL used when --data-path is missing.")
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
    if no_download:
        raise FileNotFoundError(f"Missing dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " "))
    return text.strip()


def load_dataset(path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {text_column, label_column} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[[text_column, label_column]].copy()
    df[text_column] = df[text_column].map(clean_text)
    df[label_column] = df[label_column].astype(str).str.strip().str.lower()
    df = df[df[text_column].ne("")]
    df = df[df[label_column].isin(LABEL_MAP)]
    df["label"] = df[label_column].map(LABEL_MAP).astype(int)

    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both labels: fake and original.")
    return df


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


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    reports_dir = Path(args.reports_dir)

    ensure_data(data_path, args.data_url, args.no_download)
    df = load_dataset(data_path, args.text_column, args.label_column)
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
        "label_counts": df["label"].value_counts().sort_index().to_dict(),
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
