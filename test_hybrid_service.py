"""Tests for star-rating fusion and the hybrid sentiment service."""

from __future__ import annotations

import unittest

import numpy as np

from model import SentimentModelService, parse_project_metadata, resolve_model_source
from rating_fusion import fuse_with_rating


class RatingFusionTests(unittest.TestCase):
    def test_five_star_pulls_positive(self):
        ambiguous = np.array([[0.3, 0.4, 0.3]])
        fused = fuse_with_rating(ambiguous, [5], alpha=0.7)[0]
        self.assertEqual(int(fused.argmax()), 2)

    def test_one_star_pulls_negative(self):
        ambiguous = np.array([[0.3, 0.4, 0.3]])
        fused = fuse_with_rating(ambiguous, [1], alpha=0.7)[0]
        self.assertEqual(int(fused.argmax()), 0)

    def test_missing_rating_leaves_text_probs(self):
        text = np.array([[0.1, 0.2, 0.7]])
        fused = fuse_with_rating(text, [None], alpha=0.7)
        np.testing.assert_allclose(fused, text)


class HybridServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        metadata = parse_project_metadata("pipeline.ipynb")
        source, _note = resolve_model_source(metadata)
        cls.service = SentimentModelService(source)

    def test_schema_is_three_sentiment_classes(self):
        ok, _msg = self.service.validate_expected_label_schema()
        self.assertTrue(ok)

    def test_positive_taglish(self):
        pred = self.service.predict("Ang ganda, sulit na sulit!", rating=5)
        self.assertEqual(pred.label, "positive")

    def test_negative_taglish(self):
        pred = self.service.predict("Scam to, fake item, di gumana.", rating=1)
        self.assertEqual(pred.label, "negative")

    def test_neutral_mixed_review(self):
        pred = self.service.predict("Okay lang, medyo mahina ang mic.", rating=3)
        self.assertEqual(pred.label, "neutral")


if __name__ == "__main__":
    unittest.main()
