"""Review text formatting shared by training (Colab) and Streamlit inference."""

from __future__ import annotations

import re
from typing import Optional

SHOPEE_FIELD_RE = re.compile(
    r"(?im)^\s*(Product Quality|Performance|Best Feature|Worst Feature|Others?)\s*:\s*"
)


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return value != value  # NaN
    except Exception:
        return False


def clean_review(text) -> str:
    """Strip Shopee template field labels and collapse whitespace."""
    if _is_missing(text):
        return ""
    body = SHOPEE_FIELD_RE.sub("", str(text))
    return re.sub(r"\s+", " ", body).strip()


def build_model_text(text, rating=None, use_rating_context: bool = True) -> str:
    """Format review text the same way the Colab trainer does.

    When a 1–5 star rating is available, the model sees:
    ``Star rating: {n} out of 5.\\nReview: {body}``
    """
    body = clean_review(text)
    if not use_rating_context:
        return body
    try:
        if _is_missing(rating):
            return body
        stars = int(float(rating))
        if stars < 1 or stars > 5:
            return body
    except (TypeError, ValueError):
        return body
    return f"Star rating: {stars} out of 5.\nReview: {body}"


def guess_rating_column(columns) -> Optional[str]:
    candidates = ("Rating Star", "rating star", "rating", "stars", "star")
    lower = {str(col).lower(): col for col in columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None
