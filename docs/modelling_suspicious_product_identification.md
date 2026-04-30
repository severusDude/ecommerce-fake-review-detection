# Modelling Suspicious Product Identification

## Research Objective

This modelling stage identifies products that should be prioritized for manual audit because their reviews show suspicious fake-review behavior. The output is a ranked product list, not a final fraud verdict.

The model first learns review-level fake probability from labelled review behavior. It then aggregates those probabilities and behavioral signals to the product level through `item_id`.

## Data and Label Provenance

The modelling input is `data/labels/review_shopee_labelled.csv`, built from the strong-label pipeline. It contains 500 balanced review labels: 250 `fake` and 250 `original`.

Label sources are treated differently:

- `human_seed` and `human_active`: highest trust, full sample weight.
- `label_spreading`: medium trust if present.
- `pseudo_xgb`: lower trust because pseudo labels may amplify previous model bias.

The selected predictors come from `reports/strong_labels/selected_features.csv`. They are behavioral features, not raw text embeddings. The trained review model is applied to all rows in `data/processed/review_shopee_processed.csv`, then grouped by `item_id`.

## Feature Rationale

The selected feature set combines five behavior groups:

- User behavior: repeated comments, number of shops reviewed, and user timing gaps indicate coordinated or unnatural reviewer behavior.
- Product behavior: duplicate comment rate, active days, review velocity, mean gaps, empty comments, and image-review proxy describe suspicious concentration around one product.
- Shop behavior: shop-level unique users, empty comments, and burst ratio help capture broader seller-level review patterns.
- Review behavior: duplicate flag and unique-word ratio capture low-effort or template-like comments.
- Temporal behavior: same-day activity and previous-gap features capture bursts and account reuse cadence.

This matches the research objective because suspicious products are more likely to be visible through aggregate review behavior than through one isolated review.

## Model and Product Score

Two review-level models are trained:

- Logistic Regression as a transparent baseline.
- XGBoost as the main model because it performs well on small tabular datasets, handles nonlinear interactions, and provides feature importance.

Evaluation uses stratified 5-fold cross-validation and stratified group cross-validation by `item_id` when enough product groups exist. Metrics include accuracy, precision, recall, F1, ROC-AUC, PR-AUC, classification report, and confusion matrix.

The final review model predicts `fake_probability` for every processed review. Product suspiciousness is computed as:

```text
suspicious_score =
  0.45 * mean_fake_probability
+ 0.25 * high_risk_review_share
+ 0.10 * review_count_percentile
+ 0.08 * burst_percentile
+ 0.07 * duplicate_comment_percentile
+ 0.05 * empty_comment_percentile
```

`high_risk_review_share` is the share of reviews with `fake_probability >= 0.70` by default.

Each product also receives reason codes such as high mean fake probability, many high-risk reviews, high review volume, burst-like timing, duplicate comment pattern, or empty comment pattern.

## Explainability

The modelling script writes `reports/modelling/feature_importance.csv` with:

- XGBoost impurity-based feature importance.
- Permutation importance on average precision.
- Original feature-selection context, including feature group and consensus rank.

Product-level explainability is delivered through ranked score components and reason codes in `product_suspicious_scores.csv` and `top_suspicious_products.csv`.

SHAP can be added later for local per-review explanations, but it is not required for the first reproducible modelling pass.

## Outputs

Running `uv run python src/model_suspicious_products.py` creates:

- `reports/modelling/review_model_metrics.json`
- `reports/modelling/review_classification_report.txt`
- `reports/modelling/review_confusion_matrix.csv`
- `reports/modelling/feature_importance.csv`
- `reports/modelling/product_suspicious_scores.csv`
- `reports/modelling/top_suspicious_products.csv`
- `reports/modelling/modelling_summary.json`

## Limitations

The available labels are review-level labels, not product-level ground truth. Therefore, the product score is an audit-priority signal. It should not be interpreted as proof that a product, seller, or user committed fraud.

Pseudo labels improve coverage but can reinforce earlier model assumptions. This is why pseudo-labelled rows receive lower training weight and why group-based evaluation is reported alongside regular stratified cross-validation.
