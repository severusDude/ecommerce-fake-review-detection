import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*", category=UserWarning)

import numpy as np
import pandas as pd
from joblib.externals.loky.backend import context as loky_context
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


loky_context.physical_cores_cache = os.cpu_count() or 1

FEATURE_COLUMNS = [
    "review_count",
    "active_days",
    "span_days",
    "reviews_per_active_day",
    "max_reviews_per_window",
    "mean_interreview_hours",
    "unique_users",
    "user_repeat_ratio",
    "empty_comment_ratio",
    "duplicate_comment_ratio",
    "rating_mean",
    "rating_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect burst review patterns with clustering.")
    parser.add_argument("--data-path", default="data/review_shopee.csv", help="Path to raw Shopee review CSV.")
    parser.add_argument("--reports-dir", default="reports/burst_clustering", help="Directory for output reports.")
    parser.add_argument("--entity-column", default="item_id", help="Entity to cluster, usually item_id or shop_id.")
    parser.add_argument("--time-column", default="ctime", help="Unix timestamp column.")
    parser.add_argument("--user-column", default="userid", help="Reviewer/user id column.")
    parser.add_argument("--comment-column", default="comment", help="Review text column.")
    parser.add_argument("--rating-column", default="rating_star", help="Rating column.")
    parser.add_argument("--window", default="1D", help="Pandas time window for burst counts, e.g. 1D, 12H, 7D.")
    parser.add_argument("--dbscan-eps", type=float, default=1.25, help="DBSCAN eps on scaled feature space.")
    parser.add_argument("--dbscan-min-samples", type=int, default=5, help="DBSCAN min_samples.")
    parser.add_argument("--kmeans-clusters", type=int, default=3, help="Comparison KMeans cluster count.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=30, help="Rows to print from ranked burst candidates.")
    return parser.parse_args()


def clean_comment(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split()).strip().lower()


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def load_reviews(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.data_path)
    require_columns(
        df,
        [
            args.entity_column,
            args.time_column,
            args.user_column,
            args.comment_column,
            args.rating_column,
        ],
    )

    df = df.copy()
    df["review_datetime"] = pd.to_datetime(df[args.time_column], unit="s", errors="coerce")
    df = df[df["review_datetime"].notna()]
    df["review_date"] = df["review_datetime"].dt.date
    df["comment_clean"] = df[args.comment_column].map(clean_comment)
    df["comment_nonempty"] = df["comment_clean"].ne("")
    df[args.rating_column] = pd.to_numeric(df[args.rating_column], errors="coerce")

    if df.empty:
        raise ValueError("No rows with valid review timestamps.")
    return df


def mode_or_first(series: pd.Series) -> object:
    modes = series.dropna().mode()
    if not modes.empty:
        return modes.iloc[0]
    non_null = series.dropna()
    if not non_null.empty:
        return non_null.iloc[0]
    return np.nan


def duplicate_comment_ratio(series: pd.Series) -> float:
    nonempty = series[series.ne("")]
    if len(nonempty) == 0:
        return 0.0
    return float(1.0 - (nonempty.nunique() / len(nonempty)))


def build_entity_features(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    entity = args.entity_column
    timestamp = "review_datetime"

    sorted_df = df.sort_values([entity, timestamp]).copy()
    sorted_df["gap_hours"] = sorted_df.groupby(entity)[timestamp].diff().dt.total_seconds() / 3600

    grouped = sorted_df.groupby(entity, dropna=False)
    features = grouped.agg(
        review_count=(timestamp, "size"),
        first_review_at=(timestamp, "min"),
        last_review_at=(timestamp, "max"),
        active_days=("review_date", "nunique"),
        mean_interreview_hours=("gap_hours", "mean"),
        unique_users=(args.user_column, "nunique"),
        empty_comment_ratio=("comment_nonempty", lambda value: float(1.0 - value.mean())),
        duplicate_comment_ratio=("comment_clean", duplicate_comment_ratio),
        rating_mean=(args.rating_column, "mean"),
        rating_std=(args.rating_column, "std"),
    ).reset_index()

    if "shop_id" in sorted_df.columns and entity != "shop_id":
        shop_lookup = grouped["shop_id"].agg(mode_or_first).rename("shop_id").reset_index()
        features = features.merge(shop_lookup, on=entity, how="left")
    if "product_title" in sorted_df.columns:
        title_lookup = grouped["product_title"].agg(mode_or_first).rename("product_title").reset_index()
        features = features.merge(title_lookup, on=entity, how="left")

    span_seconds = (features["last_review_at"] - features["first_review_at"]).dt.total_seconds()
    features["span_days"] = (span_seconds / 86400).fillna(0).clip(lower=0)
    features["reviews_per_active_day"] = features["review_count"] / features["active_days"].clip(lower=1)
    features["mean_interreview_hours"] = features["mean_interreview_hours"].fillna(features["span_days"] * 24)
    features["mean_interreview_hours"] = features["mean_interreview_hours"].fillna(0)
    features["user_repeat_ratio"] = 1.0 - (features["unique_users"] / features["review_count"].clip(lower=1))
    features["rating_mean"] = features["rating_mean"].fillna(0)
    features["rating_std"] = features["rating_std"].fillna(0)

    window_counts = (
        sorted_df.set_index(timestamp)
        .groupby(entity)
        .resample(args.window)
        .size()
        .rename("window_review_count")
        .reset_index()
    )
    max_window = window_counts.groupby(entity)["window_review_count"].max().rename("max_reviews_per_window").reset_index()
    features = features.merge(max_window, on=entity, how="left")
    features["max_reviews_per_window"] = features["max_reviews_per_window"].fillna(features["review_count"])

    return features


def percentile(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=False) <= 1:
        return pd.Series(0.0, index=series.index)
    return series.rank(pct=True, method="average")


def add_risk_score(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    count_pressure = percentile(result["review_count"])
    window_pressure = percentile(result["max_reviews_per_window"])
    density_pressure = percentile(result["reviews_per_active_day"])
    compression_pressure = 1.0 - percentile(result["span_days"])
    gap_pressure = 1.0 - percentile(result["mean_interreview_hours"])
    repeat_pressure = percentile(result["user_repeat_ratio"])
    empty_pressure = percentile(result["empty_comment_ratio"])
    duplicate_pressure = percentile(result["duplicate_comment_ratio"])
    uniform_rating_pressure = 1.0 - percentile(result["rating_std"])

    result["burst_risk_score"] = (
        0.20 * count_pressure
        + 0.20 * window_pressure
        + 0.15 * density_pressure
        + 0.12 * compression_pressure
        + 0.10 * gap_pressure
        + 0.08 * repeat_pressure
        + 0.07 * empty_pressure
        + 0.05 * duplicate_pressure
        + 0.03 * uniform_rating_pressure
    ).round(4)
    return result


def add_clusters(features: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    result = features.copy()
    matrix = result[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled = StandardScaler().fit_transform(matrix)

    dbscan = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
    result["dbscan_cluster"] = dbscan.fit_predict(scaled)
    result["is_dbscan_anomaly"] = result["dbscan_cluster"].eq(-1)

    kmeans = KMeans(n_clusters=args.kmeans_clusters, random_state=args.seed, n_init=10)
    result["kmeans_cluster"] = kmeans.fit_predict(scaled)

    cluster_density = result.groupby("kmeans_cluster")["burst_risk_score"].mean().sort_values()
    cluster_names = {
        int(cluster): name
        for cluster, name in zip(cluster_density.index.tolist(), ["normal", "medium_density", "burst_like"], strict=False)
    }
    result["kmeans_cluster_name"] = result["kmeans_cluster"].map(cluster_names)

    summary = {
        "feature_columns": FEATURE_COLUMNS,
        "dbscan": {
            "eps": args.dbscan_eps,
            "min_samples": args.dbscan_min_samples,
            "cluster_counts": {str(k): int(v) for k, v in result["dbscan_cluster"].value_counts().sort_index().items()},
            "anomaly_count": int(result["is_dbscan_anomaly"].sum()),
        },
        "kmeans": {
            "clusters": args.kmeans_clusters,
            "cluster_names": {str(k): v for k, v in cluster_names.items()},
            "cluster_counts": {str(k): int(v) for k, v in result["kmeans_cluster"].value_counts().sort_index().items()},
        },
    }
    return result, summary


def save_outputs(results: pd.DataFrame, summary: dict[str, object], df: pd.DataFrame, args: argparse.Namespace) -> None:
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    ranked = results.sort_values(["burst_risk_score", "max_reviews_per_window", "review_count"], ascending=False)
    candidates = ranked[(ranked["is_dbscan_anomaly"]) | (ranked["kmeans_cluster_name"].eq("burst_like"))].copy()

    ranked.to_csv(reports_dir / "entity_burst_clusters.csv", index=False)
    candidates.to_csv(reports_dir / "burst_candidates.csv", index=False)

    metadata = {
        "data_path": args.data_path,
        "entity_column": args.entity_column,
        "window": args.window,
        "input_rows": int(len(df)),
        "entities": int(results[args.entity_column].nunique()),
        "ctime_min": df["review_datetime"].min().isoformat(),
        "ctime_max": df["review_datetime"].max().isoformat(),
        "outputs": {
            "all_entities": str(reports_dir / "entity_burst_clusters.csv"),
            "candidates": str(reports_dir / "burst_candidates.csv"),
        },
    }
    metadata.update(summary)
    (reports_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    columns = [
        args.entity_column,
        "shop_id",
        "review_count",
        "max_reviews_per_window",
        "span_days",
        "unique_users",
        "empty_comment_ratio",
        "duplicate_comment_ratio",
        "burst_risk_score",
        "dbscan_cluster",
        "kmeans_cluster_name",
    ]
    visible_columns = [column for column in columns if column in ranked.columns]
    print("\nTop burst candidates:")
    print(ranked[visible_columns].head(args.top_n).to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.kmeans_clusters < 2:
        raise ValueError("--kmeans-clusters must be at least 2.")

    df = load_reviews(args)
    features = build_entity_features(df, args)
    features = add_risk_score(features)
    results, summary = add_clusters(features, args)
    save_outputs(results, summary, df, args)


if __name__ == "__main__":
    main()
