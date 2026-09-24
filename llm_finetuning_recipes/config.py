"""Loading and validating recipes/conversational-agent-qlora/config.yaml.

Kept dependency-light on purpose (just PyYAML) so that config validation,
which is part of ``train.py --dry-run``, never needs torch or unsloth to be
importable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_LORA_KEYS = {"r", "alpha", "dropout", "target_modules"}
REQUIRED_TRAINING_KEYS = {
    "learning_rate",
    "weight_decay",
    "lr_scheduler_type",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "num_train_epochs",
    "max_seq_length",
    "eval_strategy",
    "per_device_eval_batch_size",
}
REQUIRED_DATA_KEYS = {"train_path", "val_path"}
REQUIRED_TOP_KEYS = {"base_model", "lora", "training", "data", "output_dir"}


class ConfigError(ValueError):
    """Raised when config.yaml is missing keys or has out-of-range values."""


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ConfigError(f"{path} did not parse to a mapping")
    return config


def validate_config(config: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems. Empty list means valid."""
    errors: list[str] = []

    missing_top = REQUIRED_TOP_KEYS - config.keys()
    if missing_top:
        errors.append(f"missing top-level keys: {sorted(missing_top)}")
        # Some checks below assume the keys exist; bail out early.
        return errors

    if not isinstance(config["base_model"], str) or not config["base_model"].strip():
        errors.append("base_model must be a non-empty string")

    lora = config["lora"]
    if not isinstance(lora, dict):
        errors.append("lora must be a mapping")
    else:
        missing = REQUIRED_LORA_KEYS - lora.keys()
        if missing:
            errors.append(f"lora missing keys: {sorted(missing)}")
        else:
            if not isinstance(lora["r"], int) or lora["r"] <= 0:
                errors.append("lora.r must be a positive integer")
            if not isinstance(lora["alpha"], int) or lora["alpha"] <= 0:
                errors.append("lora.alpha must be a positive integer")
            if not isinstance(lora["dropout"], (int, float)) or not (0 <= lora["dropout"] < 1):
                errors.append("lora.dropout must be a number in [0, 1)")
            if not isinstance(lora["target_modules"], list) or not lora["target_modules"]:
                errors.append("lora.target_modules must be a non-empty list")

    training = config["training"]
    if not isinstance(training, dict):
        errors.append("training must be a mapping")
    else:
        missing = REQUIRED_TRAINING_KEYS - training.keys()
        if missing:
            errors.append(f"training missing keys: {sorted(missing)}")
        else:
            lr = training["learning_rate"]
            if not (isinstance(lr, (int, float)) and lr > 0):
                errors.append("training.learning_rate must be positive")
            wd = training["weight_decay"]
            if not (isinstance(wd, (int, float)) and wd >= 0):
                errors.append("training.weight_decay must be >= 0")
            if training["lr_scheduler_type"] not in {"cosine", "linear", "constant"}:
                errors.append("training.lr_scheduler_type must be one of cosine, linear, constant")
            for key in (
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "num_train_epochs",
                "max_seq_length",
                "per_device_eval_batch_size",
            ):
                if not isinstance(training[key], int) or training[key] <= 0:
                    errors.append(f"training.{key} must be a positive integer")
            if training["eval_strategy"] not in {"epoch", "steps", "no"}:
                errors.append("training.eval_strategy must be one of epoch, steps, no")

    data = config["data"]
    if not isinstance(data, dict):
        errors.append("data must be a mapping")
    else:
        missing = REQUIRED_DATA_KEYS - data.keys()
        if missing:
            errors.append(f"data missing keys: {sorted(missing)}")

    if not isinstance(config["output_dir"], str) or not config["output_dir"].strip():
        errors.append("output_dir must be a non-empty string")

    return errors
