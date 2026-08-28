from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from langdetect import LangDetectException, detect
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ndcg_score,
    precision_recall_fscore_support,
)

from rating_fusion import DEFAULT_FUSION_ALPHA, fuse_with_rating
from text_format import build_model_text, clean_review, guess_rating_column


ID2LABEL: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID: Dict[str, int] = {label: idx for idx, label in ID2LABEL.items()}
LABEL_ORDER: List[str] = ["negative", "neutral", "positive"]
LABEL_COLORS: Dict[str, str] = {
    "negative": "#E74C3C",
    "neutral": "#95A5A6",
    "positive": "#2ECC71",
}
MAX_MODEL_LENGTH = 192  # Matches the Colab notebook MAX_LENGTH.
MAX_INPUT_CHARS = 10_000
DEFAULT_BATCH_SIZE = 16
HYBRID_DIRNAME = "hybrid_model"


@dataclass
class ModelMetadata:
    project_title: str = (
        "Harnessing Sentiment Analysis for Smarter Recommendations "
        "in Filipino Online Reviews"
    )
    project_description: str = (
        "A sentiment-aware recommender system for Filipino, English, and Taglish "
        "(Filipino-English) e-commerce reviews. A FiReCS hybrid classifier "
        "(optional fine-tuned XLM-RoBERTa) scores review sentiment, then a "
        "Bayesian Weighted Score ranks products while protecting against low-sample bias."
    )
    authors: List[str] = field(
        default_factory=lambda: [
            "Cyprian Andrew D. Garpida",
            "Seth Gabriel C. Lagman",
            "Arsenio B. Nicolas Jr.",
        ]
    )
    institution: str = "School of Information Technology, Mapúa University"
    notebook_path: Optional[str] = None
    model_name: str = "xlm-roberta-base"
    model_dir_hint: str = "models/xlmr_sentiment_model"
    num_labels: int = 3
    labels: Dict[int, str] = field(default_factory=lambda: ID2LABEL.copy())
    labeled_rows: Optional[int] = None
    train_rows: Optional[int] = None
    val_rows: Optional[int] = None
    class_distribution: Dict[str, int] = field(default_factory=dict)
    eval_metrics: Dict[str, float] = field(default_factory=dict)
    training_hyperparameters: Dict[str, str] = field(default_factory=dict)
    supported_languages: List[str] = field(default_factory=list)
    language_support_note: str = ""
    model_runtime_note: str = ""
    dataset_note: str = (
        "FiReCS (10,487 Filipino-English code-switched Shopee + Google Maps reviews). "
        "Official split 7,340 train / 3,147 test, plus Shopee star-rating fusion."
    )
    split_ratio: str = "FiReCS 7340/3147 train/test"
    metadata_fields: List[str] = field(
        default_factory=lambda: [
            "product_title",
            "product_category",
            "product_review_text",
        ]
    )


@dataclass
class SentimentPrediction:
    label_id: int
    label: str
    confidence: float
    probabilities: Dict[str, float]
    detected_language_code: str
    detected_language_name: str
    supported_language: bool
    processed_text: str
    truncated: bool
    warnings: List[str]


LANGUAGE_NAME_BY_CODE = {
    "en": "English",
    "tl": "Tagalog/Filipino",
    "fil": "Filipino",
}

# The notebook does not provide an explicit language list; these are grounded in
# observed sample comments and project context.
PROJECT_LANGUAGE_CODES = {"en", "tl", "fil"}


def _hybrid_dir_candidates(explicit: Optional[str] = None) -> List[Path]:
    paths: List[Path] = []
    if explicit:
        paths.append(Path(explicit))
    here = Path(__file__).resolve().parent
    paths.extend(
        [
            Path.cwd() / HYBRID_DIRNAME,
            here / HYBRID_DIRNAME,
        ]
    )
    return paths


def is_hybrid_source(source: str) -> bool:
    path = Path(source)
    return (path / "pipeline.joblib").exists()


def is_xlmr_checkpoint(source: str) -> bool:
    config = Path(source) / "config.json"
    if not config.exists():
        return False
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    labels = payload.get("id2label") or {}
    normalized = {_normalize_label(v) for v in labels.values()}
    return {"negative", "neutral", "positive"} <= normalized


def load_hybrid_metrics() -> dict:
    for folder in _hybrid_dir_candidates():
        metrics_file = folder / "metrics.json"
        if metrics_file.exists():
            try:
                return json.loads(metrics_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return {}


def _normalize_label(label: str) -> str:
    return str(label).strip().lower()


# --------------------------------------------------------------------------- #
# Notebook metadata parsing
# --------------------------------------------------------------------------- #
def _read_notebook_text(notebook_path: Path) -> str:
    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)

    chunks: List[str] = []
    for cell in notebook.get("cells", []):
        for line in cell.get("source", []):
            chunks.append(line)
    return "".join(chunks)


def _search_int(pattern: str, text: str) -> Optional[int]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def _search_float(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def _search_str(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _candidate_notebook_paths(explicit_path: Optional[str] = None) -> List[Path]:
    candidates: List[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path))

    candidates.append(Path.cwd() / "pipeline.ipynb")
    candidates.append(Path("C:/Users/cypri/Downloads/pipeline.ipynb"))
    return candidates


def parse_project_metadata(notebook_path: Optional[str] = None) -> ModelMetadata:
    metadata = ModelMetadata()
    metadata.supported_languages = ["English", "Filipino", "Taglish (Filipino-English)"]
    metadata.language_support_note = (
        "Hybrid Taglish classifier (word+char TF-IDF logistic regression on FiReCS) "
        "with Shopee star-rating fusion. Optional XLM-RoBERTa checkpoint is used when "
        "a fine-tuned `xlmr_sentiment_model` folder is present."
    )

    notebook_file: Optional[Path] = None
    notebook_text: Optional[str] = None

    for candidate in _candidate_notebook_paths(notebook_path):
        if candidate.exists():
            notebook_file = candidate
            notebook_text = _read_notebook_text(candidate)
            break

    if notebook_file is not None and notebook_text is not None:
        metadata.notebook_path = str(notebook_file)
        metadata.model_name = (
            _search_str(r'MODEL_NAME\s*=\s*"([^"]+)"', notebook_text) or metadata.model_name
        )
        parsed_num_labels = _search_int(r"NUM_LABELS\s*=\s*(\d+)", notebook_text)
        if parsed_num_labels:
            metadata.num_labels = parsed_num_labels

        metadata.labeled_rows = _search_int(r"Labeled rows\s*:\s*([\d,]+)", notebook_text)
        metadata.train_rows = _search_int(r"Train:\s*([\d,]+)\s*\|\s*Val:", notebook_text)
        metadata.val_rows = _search_int(
            r"Train:\s*[\d,]+\s*\|\s*Val:\s*([\d,]+)", notebook_text
        )

        class_neg = _search_int(r"NEGATIVE\s*\(label=0\)\s*—\s*([\d,]+)\s*rows", notebook_text)
        class_neu = _search_int(r"NEUTRAL\s*\(label=1\)\s*—\s*([\d,]+)\s*rows", notebook_text)
        class_pos = _search_int(r"POSITIVE\s*\(label=2\)\s*—\s*([\d,]+)\s*rows", notebook_text)
        if class_neg is not None:
            metadata.class_distribution["negative"] = class_neg
        if class_neu is not None:
            metadata.class_distribution["neutral"] = class_neu
        if class_pos is not None:
            metadata.class_distribution["positive"] = class_pos

        for metric in ["accuracy", "precision", "recall", "f1"]:
            value = _search_float(rf"eval_{metric}:\s*([0-9]+\.[0-9]+)", notebook_text)
            if value is not None:
                metadata.eval_metrics[metric] = value

        epochs = _search_int(r"EPOCHS\s*=\s*(\d+)", notebook_text)
        batch_size = _search_int(r"BATCH_SIZE\s*=\s*(\d+)", notebook_text)
        learning_rate = _search_str(r"LR\s*=\s*([0-9eE\-\.+]+)", notebook_text)
        if epochs is not None:
            metadata.training_hyperparameters["epochs"] = str(epochs)
        if batch_size is not None:
            metadata.training_hyperparameters["batch_size"] = str(batch_size)
        if learning_rate:
            metadata.training_hyperparameters["learning_rate"] = learning_rate
        metadata.training_hyperparameters["max_length"] = str(MAX_MODEL_LENGTH)

    hybrid = load_hybrid_metrics()
    if hybrid:
        metadata.model_name = hybrid.get("backend", "tfidf_logreg_firecs")
        eval_block = hybrid.get("eval_with_rating_fusion") or hybrid.get("eval") or {}
        for key in ("accuracy", "precision", "recall", "f1"):
            if key in eval_block:
                metadata.eval_metrics[key] = float(eval_block[key])
        if "n_samples" in eval_block:
            metadata.val_rows = int(eval_block["n_samples"])
        metadata.labeled_rows = metadata.labeled_rows or 10487
        metadata.train_rows = metadata.train_rows or 7340
        metadata.class_distribution = {
            "negative": 2381 + 1027,
            "neutral": 2549 + 1087,
            "positive": 2410 + 1033,
        }
        hp = hybrid.get("hyperparameters") or {}
        metadata.training_hyperparameters.update({str(k): str(v) for k, v in hp.items()})
        metadata.training_hyperparameters["rating_fusion_alpha"] = str(
            hybrid.get("fusion_alpha", DEFAULT_FUSION_ALPHA)
        )
    return metadata


def resolve_model_source(
    metadata: ModelMetadata,
    explicit_model_dir: Optional[str] = None,
) -> Tuple[str, str]:
    candidates: List[Path] = []

    if explicit_model_dir:
        candidates.append(Path(explicit_model_dir))

    candidates.append(Path.cwd() / "models" / "xlmr_sentiment_model")
    if metadata.notebook_path:
        candidates.append(
            Path(metadata.notebook_path).parent / "models" / "xlmr_sentiment_model"
        )

    for candidate in candidates:
        if is_xlmr_checkpoint(str(candidate)):
            return str(candidate), "Using local fine-tuned XLM-RoBERTa checkpoint."
        if is_hybrid_source(str(candidate)):
            return str(candidate), "Using local hybrid sentiment pipeline."

    for folder in _hybrid_dir_candidates():
        if is_hybrid_source(str(folder)):
            return (
                str(folder),
                "Using FiReCS hybrid TF-IDF model with Shopee star-rating fusion.",
            )

    return (
        metadata.model_name,
        "No local sentiment checkpoint found. Predictions stay disabled until "
        "`hybrid_model/` or `models/xlmr_sentiment_model/` is present.",
    )


# --------------------------------------------------------------------------- #
# Model service
# --------------------------------------------------------------------------- #
class SentimentModelService:
    def __init__(self, model_source: str):
        self.model_source = model_source
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cpu"
        self.use_rating_prefix = False
        self.fuse_ratings = True
        self.fusion_alpha = DEFAULT_FUSION_ALPHA

        if is_hybrid_source(model_source):
            import joblib

            self.backend = "hybrid"
            self.pipeline = joblib.load(Path(model_source) / "pipeline.joblib")
            self.runtime_id2label = {idx: label for idx, label in ID2LABEL.items()}
        elif is_xlmr_checkpoint(model_source):
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.backend = "xlmr"
            self.use_rating_prefix = True
            self.fuse_ratings = False
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_source)
            self.model.to(self.device)
            self.model.eval()
            self.runtime_id2label = self._extract_runtime_label_map()
        else:
            raise FileNotFoundError(
                f"No usable sentiment model at {model_source!r}. "
                "Train `python train_hybrid.py` or place a fine-tuned XLM-R checkpoint "
                "in models/xlmr_sentiment_model/."
            )

        self.runtime_label2id = {
            label: idx for idx, label in self.runtime_id2label.items()
        }

    def _extract_runtime_label_map(self) -> Dict[int, str]:
        config_labels = getattr(self.model.config, "id2label", None)
        runtime_map: Dict[int, str] = {}
        if isinstance(config_labels, dict):
            for key, value in config_labels.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                runtime_map[idx] = _normalize_label(value)

        if runtime_map:
            return runtime_map

        num_labels = int(getattr(self.model.config, "num_labels", len(ID2LABEL)))
        return {idx: ID2LABEL.get(idx, f"label_{idx}") for idx in range(num_labels)}

    def runtime_label_map(self) -> Dict[int, str]:
        return {idx: label for idx, label in sorted(self.runtime_id2label.items())}

    def validate_expected_label_schema(self) -> Tuple[bool, str]:
        expected = {idx: _normalize_label(label) for idx, label in ID2LABEL.items()}
        runtime = {
            idx: _normalize_label(label)
            for idx, label in self.runtime_id2label.items()
        }

        if runtime == expected:
            return True, "Runtime checkpoint label schema matches expected sentiment mapping."

        detail = (
            "Runtime checkpoint label schema mismatch. "
            f"Expected {expected}, got {runtime}. "
            "Use the fine-tuned `xlmr_sentiment_model` checkpoint that was trained "
            "with id2label={0:'negative',1:'neutral',2:'positive'}."
        )
        return False, detail

    def _positive_index(self) -> int:
        if "positive" in self.runtime_label2id:
            return self.runtime_label2id["positive"]
        # Fallback: highest index behaves like the "most positive" class.
        return max(self.runtime_id2label.keys())

    def _detect_language(self, text: str) -> Tuple[str, str]:
        try:
            code = detect(text)
        except LangDetectException:
            code = "unknown"
        return code, LANGUAGE_NAME_BY_CODE.get(
            code, code.upper() if code != "unknown" else "Unknown"
        )

    def _forward_probabilities(self, texts: Sequence[str]) -> np.ndarray:
        cleaned = [clean_review(t) or " " for t in texts]
        if self.backend == "hybrid":
            frame = pd.DataFrame({"text": cleaned})
            return np.asarray(self.pipeline.predict_proba(frame), dtype=float)

        encoded = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=MAX_MODEL_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        import torch

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, text: str, rating=None) -> SentimentPrediction:
        if not text or not text.strip():
            raise ValueError("Input text is empty.")

        original = text.strip()
        clean_text = build_model_text(original, rating=rating)
        warnings: List[str] = []
        truncated = False

        if len(clean_text) > MAX_INPUT_CHARS:
            clean_text = clean_text[:MAX_INPUT_CHARS]
            truncated = True
            warnings.append(
                f"Input exceeded {MAX_INPUT_CHARS:,} characters and was truncated."
            )

        language_code, language_name = self._detect_language(original)
        supported = language_code in PROJECT_LANGUAGE_CODES
        if not supported and language_code != "unknown":
            warnings.append(
                "Detected language is outside the explicitly observed project training "
                "samples (English/Tagalog). Prediction confidence may be less reliable."
            )

        if self.use_rating_prefix:
            model_text = build_model_text(clean_text, rating)
        else:
            model_text = clean_review(clean_text) or " "
        probs = self._forward_probabilities([model_text])[0]
        if self.fuse_ratings:
            probs = fuse_with_rating(np.asarray([probs]), [rating], self.fusion_alpha)[0]
        probs = np.asarray(probs, dtype=float).tolist()
        if not probs:
            raise RuntimeError("Model produced no output probabilities.")

        if len(probs) != len(self.runtime_id2label):
            self.runtime_id2label = {
                idx: self.runtime_id2label.get(idx, f"label_{idx}")
                for idx in range(len(probs))
            }

        if any(
            _normalize_label(label).startswith("label_")
            for label in self.runtime_id2label.values()
        ):
            warnings.append(
                "Loaded checkpoint appears to be generic classifier labels "
                "(for example, label_0/label_1). For sentiment-specific results, "
                "point the app to your fine-tuned `xlmr_sentiment_model` directory."
            )

        probabilities = {
            self.runtime_id2label[idx]: float(probs[idx]) for idx in range(len(probs))
        }
        label_id = int(np.argmax(probs))
        confidence = float(probs[label_id])

        return SentimentPrediction(
            label_id=label_id,
            label=self.runtime_id2label.get(label_id, str(label_id)),
            confidence=confidence,
            probabilities=probabilities,
            detected_language_code=language_code,
            detected_language_name=language_name,
            supported_language=supported,
            processed_text=clean_text,
            truncated=truncated,
            warnings=warnings,
        )

    def predict_labels(
        self,
        texts: Sequence[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        ratings: Optional[Sequence] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return predicted label ids and full probability matrix for a batch."""
        all_probs: List[np.ndarray] = []
        rating_list: Sequence = ratings if ratings is not None else [None] * len(texts)
        for start in range(0, len(texts), batch_size):
            raw_chunk = [
                str(t) if t is not None else "" for t in texts[start : start + batch_size]
            ]
            rate_chunk = list(rating_list[start : start + batch_size])
            while len(rate_chunk) < len(raw_chunk):
                rate_chunk.append(None)
            if self.use_rating_prefix:
                chunk = [
                    build_model_text(t if t.strip() else " ", r)
                    for t, r in zip(raw_chunk, rate_chunk)
                ]
            else:
                chunk = [clean_review(t) or " " for t in raw_chunk]
            batch_probs = self._forward_probabilities(chunk)
            if self.fuse_ratings:
                batch_probs = fuse_with_rating(
                    batch_probs, rate_chunk, self.fusion_alpha
                )
            all_probs.append(batch_probs)
        if not all_probs:
            return np.array([]), np.zeros((0, len(self.runtime_id2label)))
        prob_matrix = np.vstack(all_probs)
        pred_ids = prob_matrix.argmax(axis=1)
        return pred_ids, prob_matrix

    def analyze_reviews(
        self,
        df: pd.DataFrame,
        text_col: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        rating_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Append sentiment_label, sentiment_score, confidence_level to reviews."""
        working = df.copy()
        texts = working[text_col].fillna("").astype(str).tolist()
        resolved_rating_col = rating_col or guess_rating_column(working.columns)
        ratings = (
            working[resolved_rating_col].tolist() if resolved_rating_col else None
        )
        pred_ids, prob_matrix = self.predict_labels(
            texts, batch_size=batch_size, ratings=ratings
        )

        positive_idx = self._positive_index()
        working["sentiment_label"] = [
            self.runtime_id2label.get(int(idx), str(int(idx))) for idx in pred_ids
        ]
        working["sentiment_score"] = prob_matrix[:, positive_idx].round(6)
        working["confidence_level"] = prob_matrix.max(axis=1).round(6)
        return working

    def evaluate_classification(
        self,
        texts: Sequence[str],
        true_labels: Sequence,
        batch_size: int = DEFAULT_BATCH_SIZE,
        ratings: Optional[Sequence] = None,
    ) -> Dict[str, object]:
        pred_ids, _ = self.predict_labels(
            texts, batch_size=batch_size, ratings=ratings
        )
        y_true = np.array([_coerce_label_id(v) for v in true_labels])
        y_pred = np.array([int(v) for v in pred_ids])

        accuracy = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        labels_present = sorted(ID2LABEL.keys())
        cm = confusion_matrix(y_true, y_pred, labels=labels_present)

        per_class = precision_recall_fscore_support(
            y_true, y_pred, labels=labels_present, zero_division=0
        )
        per_class_rows = []
        for i, label_id in enumerate(labels_present):
            per_class_rows.append(
                {
                    "label": ID2LABEL[label_id],
                    "precision": round(float(per_class[0][i]), 4),
                    "recall": round(float(per_class[1][i]), 4),
                    "f1": round(float(per_class[2][i]), 4),
                    "support": int(per_class[3][i]),
                }
            )

        return {
            "n_samples": int(len(y_true)),
            "accuracy": round(accuracy, 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "confusion_matrix": cm,
            "labels": [ID2LABEL[i] for i in labels_present],
            "per_class": per_class_rows,
        }


def _coerce_label_id(value) -> int:
    """Accept 0/1/2 ints or negative/neutral/positive strings."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip().lower()
    if text.isdigit():
        return int(text)
    if text in LABEL2ID:
        return LABEL2ID[text]
    raise ValueError(f"Unrecognized label value: {value!r}")


# --------------------------------------------------------------------------- #
# Aggregation, Bayesian ranking, and ranking metrics
# --------------------------------------------------------------------------- #
def aggregate_by_product(
    df: pd.DataFrame,
    product_col: str,
    category_col: Optional[str] = None,
) -> pd.DataFrame:
    counts = (
        df.groupby([product_col, "sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LABEL_ORDER, fill_value=0)
        .rename(
            columns={
                "negative": "negative_count",
                "neutral": "neutral_count",
                "positive": "positive_count",
            }
        )
        .reset_index()
    )

    agg = (
        df.groupby(product_col)
        .agg(
            total_reviews=("sentiment_score", "count"),
            avg_sentiment_score=("sentiment_score", "mean"),
        )
        .reset_index()
    )
    agg = agg.merge(counts, on=product_col, how="left")

    if category_col and category_col in df.columns:
        category_map = df.groupby(product_col)[category_col].agg(
            lambda s: s.dropna().iloc[0] if not s.dropna().empty else ""
        )
        agg["product_category"] = agg[product_col].map(category_map)

    agg["positive_ratio"] = (agg["positive_count"] / agg["total_reviews"]).round(4)
    agg["avg_sentiment_score"] = agg["avg_sentiment_score"].round(6)
    return agg


def apply_bayesian_score(agg: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    """Bayesian Weighted Score per the thesis (Section 3.5.2).

        C = global average sentiment = sum(s_bar_i * n_i) / sum(n_i)
        m = 25th percentile of per-product review counts (prior weight)
        BWS = (s_bar * n + m * C) / (n + m)
    """
    result = agg.copy()
    n = result["total_reviews"]
    s_bar = result["avg_sentiment_score"]

    total_reviews = float(n.sum())
    C = float((s_bar * n).sum() / total_reviews) if total_reviews > 0 else 0.0
    m = float(np.percentile(n.to_numpy(dtype=float), 25))

    denom = n + m
    result["bayesian_score"] = ((s_bar * n + m * C) / denom).round(6)
    return result, C, m


def raw_sentiment_average_rank(
    agg: pd.DataFrame, top_n: Optional[int] = None
) -> pd.DataFrame:
    """RSA baseline (paper 3.6.3.A): rank purely by arithmetic mean sentiment."""
    return rank_products(agg, top_n=top_n, sort_col="avg_sentiment_score")


def rank_products(
    agg: pd.DataFrame,
    top_n: Optional[int] = None,
    sort_col: str = "bayesian_score",
) -> pd.DataFrame:
    ranked = agg.sort_values(sort_col, ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    if top_n is not None:
        ranked = ranked.head(top_n).reset_index(drop=True)
    return ranked


def evaluate_ranking(
    agg: pd.DataFrame,
    product_col: str,
    ks: Sequence[int] = (5, 10, 20),
    sentiment_threshold: float = 0.6,
    min_reviews: int = 5,
    score_col: str = "bayesian_score",
) -> Dict[str, object]:
    """Precision@K and NDCG@K (paper Section 3.6.3).

    Relevance follows the paper's criteria of "high adjusted sentiment score and
    sufficient review count": a product is relevant when its average sentiment and
    review volume clear the configured thresholds. `score_col` selects the ranking
    signal, enabling a BWS vs. Raw-Sentiment-Average (RSA) comparison.
    """
    data = agg.copy()
    data["relevant"] = (
        (data["avg_sentiment_score"] >= sentiment_threshold)
        & (data["total_reviews"] >= min_reviews)
    ).astype(int)

    ranked = data.sort_values(score_col, ascending=False).reset_index(drop=True)
    ranked_ids = ranked[product_col].tolist()
    relevant_set = set(data.loc[data["relevant"] == 1, product_col])

    rows: List[Dict[str, float]] = []
    y_true_arr = ranked["relevant"].values.reshape(1, -1)
    y_score_arr = ranked[score_col].values.reshape(1, -1)
    n_products = len(ranked)

    for k in ks:
        k_eff = min(k, n_products)
        if k_eff <= 0:
            continue
        precision = len(set(ranked_ids[:k_eff]) & relevant_set) / k_eff
        if relevant_set and n_products >= 2:
            ndcg = float(ndcg_score(y_true_arr, y_score_arr, k=k_eff))
        else:
            ndcg = 0.0
        rows.append(
            {
                "K": int(k),
                "Precision@K": round(float(precision), 4),
                "NDCG@K": round(ndcg, 4),
            }
        )

    return {
        "total_products": int(n_products),
        "relevant_products": int(len(relevant_set)),
        "sentiment_threshold": sentiment_threshold,
        "min_reviews": int(min_reviews),
        "results": rows,
    }


def prediction_history_row(raw_text: str, prediction: SentimentPrediction) -> Dict[str, str]:
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": raw_text.strip(),
        "label": prediction.label,
        "confidence": f"{prediction.confidence:.4f}",
        "language": prediction.detected_language_name,
    }
