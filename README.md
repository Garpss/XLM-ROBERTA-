# Harnessing Sentiment Analysis for Smarter Recommendations in Filipino Online Reviews

A sentiment-aware product recommender system for **Filipino, English, and Taglish (Filipino-English code-switched)** e-commerce reviews. A fine-tuned **XLM-RoBERTa** model classifies each review's sentiment, and a **Bayesian Weighted Score (BWS)** aggregates and ranks products while protecting against low-sample bias.

The project ships as an interactive **Streamlit** application with four workspaces: an overview of the methodology, single-review inference, batch analysis with ranked recommendations, and a full evaluation suite (classification + ranking metrics).

> Undergraduate thesis, B.S. Computer Science — School of Information Technology, **Mapúa University**.
> Authors: **Cyprian Andrew D. Garpida**, **Seth Gabriel C. Lagman**, **Arsenio B. Nicolas Jr.**

---

## Table of Contents

- [Features](#features)
- [How It Works (Pipeline)](#how-it-works-pipeline)
- [Methodology & Formulas](#methodology--formulas)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [The Fine-Tuned Model (required)](#the-fine-tuned-model-required)
- [Running the App](#running-the-app)
- [Data Format](#data-format)
- [Using the App](#using-the-app)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)
- [Authors](#authors)

---

## Features

- **Multilingual sentiment classification** (negative / neutral / positive) using fine-tuned XLM-RoBERTa.
- **Single-review inference** with confidence, positive-probability score, class-probability chart, and language detection — reviews can be **pulled directly from your dataset** or typed manually.
- **Batch pipeline**: joins a reviews table to a product catalog, scores every review, aggregates per product, and produces **Top-N recommendations**.
- **Bayesian Weighted Score** ranking that trusts products with more reviews, compared against a **Raw Sentiment Average (RSA)** baseline.
- **Evaluation suite**:
  - Classification — Accuracy, Precision, Recall, F1, and a Confusion Matrix.
  - Ranking — Precision@K and NDCG@K (BWS vs. RSA).
- **Strict label-schema validation** — the app refuses to make predictions unless the loaded checkpoint maps exactly `0:negative, 1:neutral, 2:positive`, preventing silent fallback to an untrained base model.
- **Thesis-ready UI** with a dark theme, methodology cards, and LaTeX formulas.

---

## How It Works (Pipeline)

```
Reviews CSV ─┐
             ├─(join on Item ID)─▶ Reviews with product name/category
Catalog CSV ─┘
        │
        ▼
1. Sentiment inference (XLM-RoBERTa)  →  sentiment_label, sentiment_score (0–1), confidence_level
        │
        ▼
2. Aggregate per product             →  total_reviews, avg_sentiment_score, class counts
        │
        ▼
3. Bayesian Weighted Score (BWS)     →  volume-aware adjusted score
        │
        ▼
4. Rank (Top-N)  +  RSA baseline     →  recommendation table + chart
        │
        ▼
5. Evaluation                        →  classification metrics + Precision@K / NDCG@K
```

---

## Methodology & Formulas

**Sentiment score.** For each review, `sentiment_score` is the model's probability for the *positive* class (0–1). Lower = more negative, higher = more positive. `confidence_level` is the max class probability.

**Bayesian Weighted Score (BWS).** Adjusts a product's average sentiment by its review volume, using dataset-derived priors:

- Global average sentiment (prior value): `C = Σ(s̄ᵢ · nᵢ) / Σ nᵢ`
- Minimum review threshold (prior weight): `m = percentile(n, 25)`
- Score: `BWS = (s̄ · n + m · C) / (n + m)`

where `s̄` = product average sentiment, `n` = product review count. Products are sorted by BWS in descending order for the Top-N list.

**Ranking evaluation.** Relevance follows the study's criteria (high adjusted sentiment score **and** sufficient review count):

- `Precision@K = (relevant items in Top-K) / K`
- `NDCG@K = DCG@K / IDCG@K`, with `DCG@K = Σ (2^relᵢ − 1) / log₂(i + 1)`

The proposed BWS ranking is compared against a **Raw Sentiment Average (RSA)** baseline (arithmetic mean sentiment, no volume weighting) to demonstrate small-sample bias mitigation.

---

## Project Structure

```
thesis/
├── app.py              # Streamlit UI: overview, single review, batch & recommendations, evaluation
├── model.py            # Model service + all data logic (inference, aggregation, BWS, ranking, metrics)
├── requirements.txt    # Pinned dependencies
├── README.md           # This file
├── .gitignore          # Excludes large models, archives, datasets, caches
└── models/             # (NOT committed) place the fine-tuned checkpoint here
    └── xlmr_sentiment_model/
        ├── config.json
        ├── model.safetensors        # or pytorch_model.bin
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── special_tokens_map.json
        └── sentencepiece.bpe.model
```

> `models/`, `*.zip`, and `*.csv` are intentionally **git-ignored** — model weights are large (multiple GB) and datasets/archives are kept out of version control. Host the trained model on Google Drive / Hugging Face and download it locally.

### Key modules in `model.py`

| Component | Purpose |
| --- | --- |
| `SentimentModelService` | Loads tokenizer + model, validates label schema, runs inference. |
| `.predict()` / `.predict_labels()` | Single / batched sentiment prediction. |
| `.analyze_reviews()` | Adds `sentiment_label`, `sentiment_score`, `confidence_level` to a DataFrame. |
| `.evaluate_classification()` | Accuracy, Precision, Recall, F1, confusion matrix. |
| `aggregate_by_product()` | Per-product totals, average sentiment, class counts. |
| `apply_bayesian_score()` | Computes `C`, `m`, and the BWS column. |
| `rank_products()` / `raw_sentiment_average_rank()` | BWS ranking and RSA baseline. |
| `evaluate_ranking()` | Precision@K and NDCG@K for a chosen score column. |
| `parse_project_metadata()` / `resolve_model_source()` | Metadata + model-path resolution. |

---

## Requirements

- **Python 3.11+** (developed on 3.13)
- The packages in `requirements.txt`:

```
streamlit==1.62.0
plotly==6.9.0
pandas==2.2.3
numpy==2.2.6
torch==2.13.0
transformers==4.57.6
scikit-learn==1.7.0
sentencepiece==0.2.2
langdetect==1.0.9
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/Garpss/XLM-ROBERTA-.git
cd XLM-ROBERTA-

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## The Fine-Tuned Model (required)

The app needs the **fine-tuned** XLM-RoBERTa checkpoint. Without it, the app falls back to the generic `xlm-roberta-base` (which only has 2 generic labels), fails schema validation, and **disables all prediction buttons** on purpose.

Place your trained model here:

```
models/xlmr_sentiment_model/
```

The folder must contain the standard Hugging Face files: `config.json`, `model.safetensors` (or `pytorch_model.bin`), and the tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `sentencepiece.bpe.model`).

The checkpoint's label map must be exactly:

```json
{ "0": "negative", "1": "neutral", "2": "positive" }
```

You can also point to a custom directory at runtime via the sidebar field **"Fine-tuned model directory"**.

---

## Running the App

```bash
python -m streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

> Use `python -m streamlit ...` if the bare `streamlit` command is not on your PATH.

When the checkpoint loads correctly, the sidebar shows **"Fine-tuned sentiment checkpoint loaded"** in green and all buttons become active.

---

## Data Format

The batch pipeline expects a **relational** setup (matching the study's Shopee export):

### Reviews CSV (required)

Review text plus a product key. Auto-detected columns include:

| Field | Example column | Notes |
| --- | --- | --- |
| Review text | `Comment` / `product_review_text` | The text scored for sentiment. |
| Product key | `Item ID` | Foreign key used to join the catalog. |
| Rating (optional) | `Rating Star` | 1–5 stars, if present. |

### Product Catalog CSV (optional)

One row per product, joined to reviews on the product key:

| Field | Example column | Notes |
| --- | --- | --- |
| Product key | `Item ID` | Join key. |
| Product name | `Product Name` / `product_title` | Grouping label for aggregation. |
| Category | `Category ID` / `product_category` | Optional. |

The app joins reviews → catalog on `Item ID`, so all listings/variants of the same product are grouped under one **Product Name**. Reviews that don't match the catalog can be dropped automatically (recommended).

---

## Using the App

### Overview
Project framing, methodology cards, and the exact formulas (BWS, `C`, `m`, Precision@K, NDCG@K).

### Batch & Recommendations
1. Upload the **Reviews CSV** (and optionally the **Product catalog CSV**).
2. Confirm the auto-detected join keys and columns (`Item ID`, `Comment`, `Product Name`, `Category ID`).
3. Choose **Top-N** (5 or 10) and click **Run Pipeline**.
4. View aggregated metrics, the ranked recommendation table + chart, and the **BWS vs. RSA** comparison. Download the recommendations as CSV.

> A guardrail warns you if the chosen "group by" column is a per-row identifier (which would make every product show a review count of ~1).

### Single Review
Toggle **Pick from dataset** to select a real review (filter by product, browse, or pick a random one), or **Type manually**. Runs the model and shows label, confidence, positive-probability, and the class-probability chart. Every run is logged to the session history.

### Evaluation
- **Classification** — upload a labeled CSV (text + true label as `0/1/2` or `negative/neutral/positive`) to get Accuracy, Precision, Recall, F1, a confusion matrix, and a per-class report.
- **Ranking** — uses the aggregated products from the Batch tab to compute Precision@K and NDCG@K for **BWS vs. RSA** across configurable K values and relevance thresholds.

---

## Evaluation

| Task | Metrics |
| --- | --- |
| Sentiment classification | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Ranking quality | Precision@K, NDCG@K (BWS vs. RSA baseline) |

---

## Troubleshooting

| Symptom | Cause / Fix |
| --- | --- |
| All "Run" buttons are disabled | The fine-tuned checkpoint is missing. Place it at `models/xlmr_sentiment_model/` and restart. |
| `AttributeError: 'ModelMetadata' object has no attribute ...` | Stale Streamlit cache after a code change. Fully **restart** the server (or use *Clear cache* → *Rerun*). |
| Every product shows review count = 1 | Wrong "group by" column (a per-row ID). Join the catalog and group by **Product Name**. |
| Predictions all one class / all "0" | The generic base model loaded instead of the fine-tuned one — the label schema check will flag this. |
| `streamlit: command not found` | Use `python -m streamlit run app.py`. |

---

## Tech Stack

- **Model:** XLM-RoBERTa (fine-tuned) via Hugging Face `transformers` + `torch`
- **App:** Streamlit
- **Data/Metrics:** pandas, NumPy, scikit-learn
- **Viz:** Plotly
- **Language detection:** langdetect

---

## Authors

- **Cyprian Andrew D. Garpida**
- **Seth Gabriel C. Lagman**
- **Arsenio B. Nicolas Jr.**

School of Information Technology, **Mapúa University** — B.S. Computer Science.

---

## Academic Use

This repository accompanies an undergraduate thesis. Please cite the authors and Mapúa University if you build upon this work.
