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

## Opsi Penting

```powershell
uv run python src\train_indobert.py --epochs 3 --batch-size 8 --max-length 256
uv run python src\train_indobert.py --data-path data\train_review_only.csv
uv run python src\train_indobert.py --model-name indobenchmark/indobert-base-p2
```

## Catatan Metodologi

Pipeline utama memakai hanya teks `comment` agar hasil merefleksikan kemampuan IndoBERT pada review text. Metadata seperti `rating_star`, `userid`, `shop_id`, dan `ctime` bisa jadi eksperimen lanjutan, tetapi tidak dicampur ke model utama.
