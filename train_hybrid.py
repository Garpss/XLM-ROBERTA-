"""Train a compact Taglish sentiment classifier on FiReCS and save it for Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from rating_fusion import DEFAULT_FUSION_ALPHA, SHOPEE_STAR_LABEL_COUNTS, fuse_with_rating
from text_format import clean_review

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
TRAIN_CSV = Path("/tmp/firecs/train.csv")
TEST_CSV = Path("/tmp/firecs/test.csv")
OUT_DIR = Path("hybrid_model")


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["review"] = df["review"].map(clean_review)
    df = df[df["review"].str.len() > 0].copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["review"]
    return df


def build_pipeline() -> Pipeline:
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=80_000,
        sublinear_tf=True,
        lowercase=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=60_000,
        sublinear_tf=True,
        lowercase=True,
    )
    features = ColumnTransformer(
        [
            ("word", word, "text"),
            ("char", char, "text"),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        C=2.5,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    return Pipeline([("features", features), ("clf", clf)])


def metrics_dict(y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "n_samples": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "labels": [ID2LABEL[i] for i in range(3)],
        "per_class": [
            {
                "label": ID2LABEL[i],
                "precision": round(float(per[0][i]), 4),
                "recall": round(float(per[1][i]), 4),
                "f1": round(float(per[2][i]), 4),
                "support": int(per[3][i]),
            }
            for i in range(3)
        ],
    }


def main() -> None:
    train = load_split(TRAIN_CSV)
    test = load_split(TEST_CSV)
    print("Train", len(train), "Test", len(test))
    print(train["label"].value_counts().sort_index().to_dict())

    pipe = build_pipeline()
    y = train["label"].to_numpy()
    pipe.fit(train[["text"]], y)

    pred = pipe.predict(test[["text"]])
    print(classification_report(test["label"], pred, target_names=list(ID2LABEL.values()), digits=4))
    report = metrics_dict(test["label"].to_numpy(), pred)
    print("Accuracy", report["accuracy"], "weighted F1", report["f1"])

    proba = pipe.predict_proba(test[["text"]])
    y_true = test["label"].to_numpy()
    star_given_y = (SHOPEE_STAR_LABEL_COUNTS + 1) / (
        SHOPEE_STAR_LABEL_COUNTS + 1
    ).sum(axis=0, keepdims=True)
    rng = np.random.default_rng(42)
    stars = np.array(
        [
            rng.choice(np.arange(1, 6), p=star_given_y[:, yi] / star_given_y[:, yi].sum())
            for yi in y_true
        ]
    )
    fused_pred = fuse_with_rating(proba, stars, DEFAULT_FUSION_ALPHA).argmax(axis=1)
    fused_report = metrics_dict(y_true, fused_pred)
    print(
        "With rating fusion: accuracy",
        fused_report["accuracy"],
        "weighted F1",
        fused_report["f1"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, OUT_DIR / "pipeline.joblib", compress=3)
    payload = {
        "backend": "tfidf_logreg_firecs",
        "dataset": "ccosme/FiReCS",
        "id2label": ID2LABEL,
        "split": "official FiReCS test (3,147 reviews)",
        "baseline_note": (
            "Previous Shopee 1k XLM-R run: accuracy 0.77 / weighted F1 0.765. "
            "This hybrid is trained on FiReCS (same domain as the thesis base paper)."
        ),
        "previous_xlmr_shopee": {"accuracy": 0.77, "f1": 0.7651, "n_samples": 200},
        "eval": report,
        "fusion_alpha": DEFAULT_FUSION_ALPHA,
        "eval_with_rating_fusion": {
            **fused_report,
            "note": (
                "FiReCS test text + Shopee-calibrated P(star|label) fusion "
                f"(alpha={DEFAULT_FUSION_ALPHA})."
            ),
        },
        "hyperparameters": {
            "word_ngrams": "1-2",
            "char_ngrams": "3-5",
            "C": 2.5,
            "class_weight": "balanced",
        },
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Saved", OUT_DIR)


if __name__ == "__main__":
    main()
