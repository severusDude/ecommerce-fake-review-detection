# Fake Review Detection with IndoBERT

Project ini fine-tune IndoBERT untuk klasifikasi review Shopee menjadi `fake` atau `original`.

Dataset default berasal dari repo [`andrioktavianto/fake-review-shopee`](https://github.com/andrioktavianto/fake-review-shopee):

- `train_review_only.csv`: dataset berlabel, dipakai untuk training supervised.
- `review_shopee.csv`: data mentah tanpa kolom label `fakeornot`, cocok untuk eksplorasi atau semi-supervised, bukan training utama.

## Setup with uv

```powershell
uv sync
```

## Baseline TF-IDF Logistic Regression

```powershell
uv run python src\baseline_tfidf.py
```

Output:

- `reports\tfidf_baseline\metrics.json`
- `reports\tfidf_baseline\classification_report.txt`
- `reports\tfidf_baseline\confusion_matrix.csv`

## Fine-tune IndoBERT

```powershell
uv run python src\train_indobert.py
```

Default model: `indobenchmark/indobert-base-p1`.

Output:

- `models\indobert-fake-review\`
- `reports\indobert\metrics.json`
- `reports\indobert\classification_report.txt`
- `reports\indobert\confusion_matrix.csv`
- `reports\indobert\label_map.json`

## Burst Review Clustering

```powershell
uv run python src\burst_review_clustering.py
```

Script ini memakai `data\review_shopee.csv` untuk identifikasi pola review yang padat dalam waktu pendek. Default entity adalah `item_id`, window waktu adalah `1D`, clustering utama memakai DBSCAN, dan KMeans 3 cluster dipakai sebagai pembanding.

Output:

- `reports\burst_clustering\entity_burst_clusters.csv`: semua produk dengan fitur burst, skor risiko, label DBSCAN, dan label KMeans.
- `reports\burst_clustering\burst_candidates.csv`: kandidat burst/anomali dari DBSCAN atau cluster KMeans `burst_like`.
- `reports\burst_clustering\summary.json`: metadata run, rentang waktu data, fitur yang dipakai, dan ringkasan cluster.

Opsi contoh:

```powershell
uv run python src\burst_review_clustering.py --entity-column shop_id
uv run python src\burst_review_clustering.py --window 12H --dbscan-eps 1.1 --dbscan-min-samples 4
```

## Adding Balanced Strong Labels Pipeline

```powershell
uv run python src\strong_label_pipeline.py baseline
uv run python src\strong_label_pipeline.py make-round --round 1
uv run python src\strong_label_pipeline.py ingest-labels --round 1 --export-path data\labels\label_studio\round_1\export.json
uv run python src\strong_label_pipeline.py make-round --round 2
uv run python src\strong_label_pipeline.py ingest-labels --round 2 --export-path data\labels\label_studio\round_2\export.json
uv run python src\strong_label_pipeline.py make-round --round 3
uv run python src\strong_label_pipeline.py ingest-labels --round 3 --export-path data\labels\label_studio\round_3\export.json
uv run python src\strong_label_pipeline.py make-round --round 4
uv run python src\strong_label_pipeline.py ingest-labels --round 4 --export-path data\labels\label_studio\round_4\export.json
uv run python src\strong_label_pipeline.py pseudo-label
uv run python src\strong_label_pipeline.py label-spread
uv run python src\strong_label_pipeline.py build-final
uv run python src\strong_label_pipeline.py validate
```

Output utama:

- `reports\strong_labels\xgb_baseline_cv_metrics.csv`
- `reports\strong_labels\xgb_baseline_cv_summary.csv`
- `reports\strong_labels\xgb_feature_importance.csv`
- `reports\strong_labels\selected_features.csv`
- `data\labels\label_studio\round_*\tasks.jsonl`
- `data\labels\review_shopee_500_strong_labels.csv`

Pipeline memakai 100 seed label dari `data\processed\review_shopee_processed.csv`, menambah active-learning human labels sampai subset human final 250 label seimbang, lalu mengisi sisa label dengan pseudo-label XGBoost dan Label Spreading. Jika 3 ronde belum cukup seimbang, lanjut saja ke ronde 4, 5, dan seterusnya sampai target human `125 fake / 125 original` tercapai. File Label Studio memakai import/export lokal; pakai `configs\label_studio_fake_review.xml` agar model probability tetap hanya menjadi metadata task dan tidak tampil ke annotator.

## Visualisasi Hasil

```powershell
uv sync
uv run jupyter notebook notebooks\visualize_results.ipynb
```

Notebook visualisasi memakai seaborn untuk membandingkan metrik TF-IDF Logistic Regression dan IndoBERT, confusion matrix kedua model, distribusi cluster DBSCAN/KMeans, serta kandidat burst review berdasarkan `burst_risk_score`.

## Opsi Penting

```powershell
uv run python src\train_indobert.py --epochs 3 --batch-size 8 --max-length 256
uv run python src\train_indobert.py --data-path data\train_review_only.csv
uv run python src\train_indobert.py --model-name indobenchmark/indobert-base-p2
```

## Catatan Metodologi

Pipeline utama memakai hanya teks `comment` agar hasil merefleksikan kemampuan IndoBERT pada review text. Metadata seperti `rating_star`, `userid`, `shop_id`, dan `ctime` bisa jadi eksperimen lanjutan, tetapi tidak dicampur ke model utama.

Burst clustering bersifat eksploratif karena `review_shopee.csv` tidak punya label `fakeornot`. Skor tinggi berarti perlu inspeksi manual, bukan bukti final review palsu.
