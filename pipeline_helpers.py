"""Transformers v4/v5 compatibility helpers for Colab XLM-RoBERTa fine-tuning.

Colab currently ships transformers 5.x, which removed ``warmup_ratio`` and
``evaluation_strategy``. Older 4.x runtimes still expect those names.
``filter_init_kwargs`` inspects the constructor and only passes supported
arguments, so the same notebook cell works on both.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

TYPEERROR_KWARG_RE = re.compile(r"unexpected keyword argument ['\"](\w+)['\"]")


def _param_names(init_fn: Callable[..., Any]) -> Tuple[set, bool]:
    params = inspect.signature(init_fn).parameters
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    names = {name for name in params if name != "self"}
    return names, has_var_keyword


def filter_init_kwargs(
    init_fn: Callable[..., Any],
    kwargs: Mapping[str, Any],
    exclusive_groups: Sequence[Sequence[str]] = (),
) -> Dict[str, Any]:
    """Keep kwargs that ``init_fn`` accepts; resolve alias groups.

    ``exclusive_groups`` is a sequence of preferred-first alias lists. The first
    name that the constructor actually accepts is kept; the rest are dropped.
    Example: ``("eval_strategy", "evaluation_strategy")``.
    """
    names, has_var_keyword = _param_names(init_fn)
    selected = dict(kwargs)

    for group in exclusive_groups:
        present = [key for key in group if key in selected]
        if len(present) <= 1:
            continue
        chosen = None
        for key in group:
            if has_var_keyword or key in names:
                chosen = key
                break
        if chosen is None:
            for key in present:
                selected.pop(key, None)
        else:
            for key in present:
                if key != chosen:
                    selected.pop(key, None)

    if has_var_keyword:
        return selected
    return {key: value for key, value in selected.items() if key in names}


def instantiate_filtering(
    cls: type,
    kwargs: Mapping[str, Any],
    exclusive_groups: Sequence[Sequence[str]] = (),
) -> Any:
    """Construct ``cls`` while dropping kwargs it rejects.

    Inspects the signature first, then peels unknown names off the TypeError
    message if a wrapper hid the real parameters from ``inspect``.
    """
    filtered = filter_init_kwargs(cls.__init__, kwargs, exclusive_groups)
    while True:
        try:
            return cls(**filtered)
        except TypeError as exc:
            match = TYPEERROR_KWARG_RE.search(str(exc))
            if not match:
                raise
            bad_name = match.group(1)
            if bad_name not in filtered:
                raise
            filtered.pop(bad_name)


TRAINING_ARG_ALIASES = (
    ("eval_strategy", "evaluation_strategy"),
    # Prefer warmup_ratio when it exists (transformers 4.x requires warmup_steps
    # to be an int). On 5.x, warmup_ratio is gone and warmup_steps accepts a
    # float in [0, 1) as a ratio of total steps.
    ("warmup_ratio", "warmup_steps"),
)

TRAINER_ALIASES = (("processing_class", "tokenizer"),)


def build_training_arguments(
    training_arguments_cls: type,
    *,
    output_dir: str,
    epochs: float,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    warmup_ratio: float,
    seed: int,
    fp16: bool,
    metric_for_best_model: str = "eval_f1",
    greater_is_better: bool = True,
    logging_steps: int = 20,
    save_total_limit: int = 2,
    weight_decay: float = 0.01,
    lr_scheduler_type: str = "cosine",
    extra_kwargs: Optional[Mapping[str, Any]] = None,
):
    """Build ``TrainingArguments`` for both transformers 4.x and 5.x."""
    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "lr_scheduler_type": lr_scheduler_type,
        "logging_steps": logging_steps,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "save_total_limit": save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
        "seed": seed,
        "report_to": "none",
        "fp16": fp16,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_ratio,
        "overwrite_output_dir": True,
        "dataloader_num_workers": 0,
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return instantiate_filtering(
        training_arguments_cls, kwargs, exclusive_groups=TRAINING_ARG_ALIASES
    )


def build_trainer_kwargs(
    trainer_cls: type,
    *,
    model: Any,
    args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: Any,
    compute_metrics: Optional[Callable] = None,
    callbacks: Optional[list] = None,
    extra_kwargs: Optional[Mapping[str, Any]] = None,
    inspect_cls: Optional[type] = None,
) -> Dict[str, Any]:
    """Trainer init kwargs that use ``processing_class`` on v5 and ``tokenizer`` on v4.

    Inspect ``inspect_cls`` (the Hugging Face ``Trainer``) rather than a subclass
    whose ``__init__`` takes ``**kwargs`` — otherwise v4 would receive the v5
    name ``processing_class`` and crash.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "compute_metrics": compute_metrics,
    }
    if callbacks is not None:
        kwargs["callbacks"] = callbacks
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return filter_init_kwargs(
        (inspect_cls or trainer_cls).__init__,
        kwargs,
        exclusive_groups=TRAINER_ALIASES,
    )


def make_focal_trainer_class(trainer_base: type) -> type:
    """Return a Trainer subclass with class-weighted focal loss + label smoothing."""
    import torch
    import torch.nn.functional as F

    class FocalTrainer(trainer_base):
        def __init__(
            self,
            *args,
            class_weights=None,
            focal_gamma: float = 1.5,
            label_smoothing: float = 0.08,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self._class_weights = class_weights
            self._focal_gamma = float(focal_gamma)
            self._label_smoothing = float(label_smoothing)
            # Custom loss does not consume num_items_in_batch via model kwargs.
            self.model_accepts_loss_kwargs = False

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch=None,
        ):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weights = self._class_weights
            if weights is not None:
                weights = weights.to(device=logits.device, dtype=logits.dtype)
            per_item = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                weight=weights,
                reduction="none",
                label_smoothing=self._label_smoothing,
            )
            pt = torch.exp(-per_item)
            loss = (((1.0 - pt) ** self._focal_gamma) * per_item).mean()
            return (loss, outputs) if return_outputs else loss

    FocalTrainer.__name__ = "FocalTrainer"
    FocalTrainer.__qualname__ = "FocalTrainer"
    return FocalTrainer
