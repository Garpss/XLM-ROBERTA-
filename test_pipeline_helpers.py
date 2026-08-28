"""Unit tests for transformers v4/v5 TrainingArguments / Trainer kwargs filtering."""

from __future__ import annotations

import unittest

from pipeline_helpers import (
    TRAINING_ARG_ALIASES,
    TRAINER_ALIASES,
    build_training_arguments,
    filter_init_kwargs,
    instantiate_filtering,
)


class TrainingArgumentsV4:
    """Closed signature like transformers 4.x (no **kwargs)."""

    def __init__(
        self,
        output_dir,
        evaluation_strategy="no",
        warmup_ratio=0.0,
        warmup_steps=0,
        fp16=False,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        weight_decay=0.0,
        lr_scheduler_type="linear",
        logging_steps=500,
        save_strategy="epoch",
        save_total_limit=None,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=False,
        seed=42,
        report_to="none",
        overwrite_output_dir=False,
        dataloader_num_workers=0,
    ):
        self.output_dir = output_dir
        self.evaluation_strategy = evaluation_strategy
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = warmup_steps
        self.captured = {
            "evaluation_strategy": evaluation_strategy,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
        }


class TrainingArgumentsV5Colab:
    """Matches the Colab crash: no warmup_ratio, no evaluation_strategy."""

    def __init__(
        self,
        output_dir,
        eval_strategy="no",
        warmup_steps=0,
        fp16=False,
        num_train_epochs=3,
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=False,
        seed=42,
        report_to="none",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        weight_decay=0.0,
        lr_scheduler_type="linear",
        logging_steps=500,
        save_total_limit=None,
        dataloader_num_workers=0,
    ):
        self.output_dir = output_dir
        self.eval_strategy = eval_strategy
        self.warmup_steps = warmup_steps
        self.captured = {
            "eval_strategy": eval_strategy,
            "warmup_steps": warmup_steps,
        }


class TrainingArgumentsV5Strict:
    """No **kwargs: unknown names raise TypeError, like a dataclass."""

    def __init__(self, output_dir, eval_strategy="no", warmup_steps=0.0):
        self.output_dir = output_dir
        self.eval_strategy = eval_strategy
        self.warmup_steps = warmup_steps


class TrainerV4:
    def __init__(self, model, args, tokenizer=None, train_dataset=None, eval_dataset=None):
        self.tokenizer = tokenizer


class TrainerV5:
    def __init__(
        self, model, args, processing_class=None, train_dataset=None, eval_dataset=None
    ):
        self.processing_class = processing_class


class FilterInitKwargsTests(unittest.TestCase):
    def test_v5_colab_drops_warmup_ratio_and_evaluation_strategy(self):
        kwargs = {
            "output_dir": "/tmp/out",
            "eval_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "warmup_ratio": 0.1,
            "warmup_steps": 0.1,
            "fp16": True,
        }
        filtered = filter_init_kwargs(
            TrainingArgumentsV5Colab.__init__,
            kwargs,
            exclusive_groups=TRAINING_ARG_ALIASES,
        )
        self.assertIn("eval_strategy", filtered)
        self.assertNotIn("evaluation_strategy", filtered)
        self.assertNotIn("warmup_ratio", filtered)
        self.assertEqual(filtered["warmup_steps"], 0.1)

    def test_v4_keeps_warmup_ratio_not_eval_strategy(self):
        kwargs = {
            "output_dir": "/tmp/out",
            "eval_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "warmup_ratio": 0.1,
            "warmup_steps": 0.1,
        }
        filtered = filter_init_kwargs(
            TrainingArgumentsV4.__init__,
            kwargs,
            exclusive_groups=TRAINING_ARG_ALIASES,
        )
        self.assertIn("evaluation_strategy", filtered)
        self.assertNotIn("eval_strategy", filtered)
        self.assertEqual(filtered["warmup_ratio"], 0.1)
        # v4 warmup_steps is an int; do not pass the ratio float.
        self.assertNotIn("warmup_steps", filtered)

    def test_v5_trainer_uses_processing_class(self):
        filtered = filter_init_kwargs(
            TrainerV5.__init__,
            {"model": "m", "args": "a", "processing_class": "tok", "tokenizer": "tok"},
            exclusive_groups=TRAINER_ALIASES,
        )
        self.assertEqual(filtered["processing_class"], "tok")
        self.assertNotIn("tokenizer", filtered)

    def test_v4_trainer_uses_tokenizer(self):
        filtered = filter_init_kwargs(
            TrainerV4.__init__,
            {"model": "m", "args": "a", "processing_class": "tok", "tokenizer": "tok"},
            exclusive_groups=TRAINER_ALIASES,
        )
        self.assertEqual(filtered["tokenizer"], "tok")
        self.assertNotIn("processing_class", filtered)


class InstantiateFilteringTests(unittest.TestCase):
    def test_peels_unknown_kwarg_from_typeerror(self):
        args = instantiate_filtering(
            TrainingArgumentsV5Strict,
            {
                "output_dir": "/tmp/out",
                "eval_strategy": "epoch",
                "evaluation_strategy": "epoch",
                "warmup_ratio": 0.1,
                "warmup_steps": 0.1,
            },
            exclusive_groups=TRAINING_ARG_ALIASES,
        )
        self.assertEqual(args.eval_strategy, "epoch")
        self.assertEqual(args.warmup_steps, 0.1)

    def test_build_training_arguments_v5_colab(self):
        args = build_training_arguments(
            TrainingArgumentsV5Colab,
            output_dir="/tmp/out",
            epochs=6,
            batch_size=8,
            grad_accum=2,
            learning_rate=1.5e-5,
            warmup_ratio=0.10,
            seed=42,
            fp16=True,
        )
        self.assertEqual(args.captured["eval_strategy"], "epoch")
        self.assertEqual(args.captured["warmup_steps"], 0.10)
        self.assertFalse(hasattr(args, "warmup_ratio"))

    def test_build_training_arguments_rejects_warmup_ratio_like_colab(self):
        """Reproduce the original crash shape, then show the helper avoids it."""
        with self.assertRaises(TypeError) as ctx:
            TrainingArgumentsV5Strict(
                output_dir="/tmp/out",
                eval_strategy="epoch",
                warmup_ratio=0.1,
            )
        self.assertIn("warmup_ratio", str(ctx.exception))

        args = build_training_arguments(
            TrainingArgumentsV5Strict,
            output_dir="/tmp/out",
            epochs=1,
            batch_size=8,
            grad_accum=1,
            learning_rate=1e-5,
            warmup_ratio=0.1,
            seed=0,
            fp16=False,
        )
        self.assertEqual(args.warmup_steps, 0.1)

    def test_build_training_arguments_v4(self):
        args = build_training_arguments(
            TrainingArgumentsV4,
            output_dir="/tmp/out",
            epochs=6,
            batch_size=8,
            grad_accum=2,
            learning_rate=1.5e-5,
            warmup_ratio=0.10,
            seed=42,
            fp16=True,
        )
        self.assertEqual(args.captured["evaluation_strategy"], "epoch")
        self.assertEqual(args.captured["warmup_ratio"], 0.10)
        self.assertEqual(args.captured["warmup_steps"], 0)


if __name__ == "__main__":
    unittest.main()
