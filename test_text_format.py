"""Tests for Shopee review text formatting used in training and inference."""

from __future__ import annotations

import math
import unittest

from text_format import build_model_text, clean_review, guess_rating_column


class TextFormatTests(unittest.TestCase):
    def test_clean_review_strips_shopee_template_fields(self):
        raw = "Product Quality: ganda\nPerformance: ok\nBest Feature: battery"
        cleaned = clean_review(raw)
        self.assertNotIn("Product Quality:", cleaned)
        self.assertIn("ganda", cleaned)
        self.assertIn("ok", cleaned)

    def test_build_model_text_with_rating(self):
        text = build_model_text("Ang ganda!", rating=5)
        self.assertEqual(text, "Star rating: 5 out of 5.\nReview: Ang ganda!")

    def test_build_model_text_without_rating(self):
        self.assertEqual(build_model_text("Ang ganda!"), "Ang ganda!")
        self.assertEqual(build_model_text("Ang ganda!", rating=float("nan")), "Ang ganda!")
        self.assertEqual(build_model_text("Ang ganda!", rating=0), "Ang ganda!")

    def test_guess_rating_column(self):
        self.assertEqual(
            guess_rating_column(["Comment", "Rating Star", "Item ID"]),
            "Rating Star",
        )
        self.assertIsNone(guess_rating_column(["Comment", "Item ID"]))

    def test_nan_rating_is_omitted(self):
        self.assertTrue(math.isnan(float("nan")))
        self.assertEqual(build_model_text("ok", rating=float("nan")), "ok")


if __name__ == "__main__":
    unittest.main()
