import argparse
import inspect
import json
import random
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


DEFAULT_DATA_URL = "https://raw.githubusercontent.com/andrioktavianto/fake-review-shopee/master/train_review_only.csv"
LABEL_MAP = {"original": 0, "fake": 1}
ID_TO_LABEL = {0: "original", 1: "fake"}


class ReviewDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: AutoTokenizer, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune IndoBERT for Shopee fake review detection.")
    parser.add_argument("--data-path", default="data/train_review_only.csv", help="Path to labeled CSV.")
    parser.add_argument("--data-url", default=DEFAULT_DATA_URL, help="URL used when --data-path is missing.")
    parser.add_argument("--text-column", default="comment")
    parser.add_argument("--label-column", default="fakeornot")
    parser.add_argument("--model-name", default="indobenchmark/indobert-base-p1")
    parser.add_argument("--output-dir", default="models/indobert-fake-review")
    parser.add_argument("--reports-dir", default="reports/indobert")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true", help="Fail if CSV does not exist locally.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def build_training_args(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    training_kwargs = {
        "output_dir": str(output_dir),
        "save_strategy": "epoch",
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "logging_dir": "runs/indobert",
        "logging_steps": 25,
        "report_to": "none",
        "seed": args.seed,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**training_kwargs)


def build_trainer(
    model: AutoModelForSequenceClassification,
    training_args: TrainingArguments,
    train_dataset: ReviewDataset,
    val_dataset: ReviewDataset,
    tokenizer: AutoTokenizer,
) -> Trainer:
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": compute_metrics,
    }

    signature = inspect.signature(Trainer.__init__)
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    return Trainer(**trainer_kwargs)


def save_reports(trainer: Trainer, test_dataset: ReviewDataset, test_df: pd.DataFrame, reports_dir: Path, train_len: int, val_len: int) -> None:
    predictions_output = trainer.predict(test_dataset)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
    labels = np.array(test_df["label"].tolist())

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "n_train": int(train_len),
        "n_val": int(val_len),
        "n_test": int(len(test_df)),
        "label_counts": test_df["label"].value_counts().sort_index().to_dict(),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (reports_dir / "label_map.json").write_text(json.dumps(LABEL_MAP, indent=2), encoding="utf-8")
    (reports_dir / "classification_report.txt").write_text(
        classification_report(labels, predictions, target_names=["original", "fake"], zero_division=0),
        encoding="utf-8",
    )
    pd.DataFrame(confusion_matrix(labels, predictions), index=["true_original", "true_fake"], columns=["pred_original", "pred_fake"]).to_csv(
        reports_dir / "confusion_matrix.csv"
    )
    print(json.dumps(metrics, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)

    ensure_data(data_path, args.data_url, args.no_download)
    df = load_dataset(data_path, args.text_column, args.label_column)
    train, val, test = split_dataset(df, args.test_size, args.val_size, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = 2
    config.id2label = ID_TO_LABEL
    config.label2id = LABEL_MAP
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    train_dataset = ReviewDataset(train[args.text_column].tolist(), train["label"].tolist(), tokenizer, args.max_length)
    val_dataset = ReviewDataset(val[args.text_column].tolist(), val["label"].tolist(), tokenizer, args.max_length)
    test_dataset = ReviewDataset(test[args.text_column].tolist(), test["label"].tolist(), tokenizer, args.max_length)

    training_args = build_training_args(args, output_dir)

    trainer = build_trainer(model, training_args, train_dataset, val_dataset, tokenizer)

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_reports(trainer, test_dataset, test, reports_dir, len(train), len(val))


if __name__ == "__main__":
    main()
