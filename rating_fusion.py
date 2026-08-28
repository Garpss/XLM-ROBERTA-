"""Blend text-model probabilities with an empirical star-rating prior.

The 1,000-row Shopee annotation table is highly informative:

    1★ → mostly negative,  3★ → mostly neutral,  5★ → almost always positive.

A star-only heuristic is ~76% accurate. Combining it with the FiReCS text
model (81.6% text-only) reaches ~86.8% on a Shopee-like rating simulation.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Rows = stars 1–5, cols = negative, neutral, positive. From the labeled
# Shopee sample (1,000 rows) used in the thesis pipeline.
SHOPEE_STAR_LABEL_COUNTS = np.array(
    [
        [190, 9, 1],
        [152, 40, 8],
        [59, 111, 30],
        [13, 80, 107],
        [0, 3, 197],
    ],
    dtype=float,
)

# Laplace-smoothed P(label | star rating)
RATING_PRIOR = (SHOPEE_STAR_LABEL_COUNTS + 1.0) / (
    SHOPEE_STAR_LABEL_COUNTS + 1.0
).sum(axis=1, keepdims=True)

DEFAULT_FUSION_ALPHA = 0.7


def _coerce_star(rating) -> Optional[int]:
    if rating is None:
        return None
    try:
        if rating != rating:  # NaN
            return None
        stars = int(float(rating))
    except (TypeError, ValueError):
        return None
    if stars < 1 or stars > 5:
        return None
    return stars


def fuse_with_rating(
    text_probs: np.ndarray,
    ratings: Optional[Sequence] = None,
    alpha: float = DEFAULT_FUSION_ALPHA,
) -> np.ndarray:
    """Return fused probabilities. ``text_probs`` shape (n, 3)."""
    probs = np.asarray(text_probs, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    if ratings is None or alpha <= 0:
        return probs

    fused = probs.copy()
    for i, raw in enumerate(list(ratings)[: len(fused)]):
        stars = _coerce_star(raw)
        if stars is None:
            continue
        prior = RATING_PRIOR[stars - 1]
        logp = np.log(fused[i] + 1e-12) + float(alpha) * np.log(prior + 1e-12)
        logp -= logp.max()
        exp = np.exp(logp)
        fused[i] = exp / exp.sum()
    return fused
