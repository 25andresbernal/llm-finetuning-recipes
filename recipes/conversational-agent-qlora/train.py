#!/usr/bin/env python3
"""QLoRA fine-tuning of unsloth/llama-3-8b-bnb-4bit on the scheduling-agent dataset.

Run from the repository root:

    uv run python recipes/conversational-agent-qlora/train.py --dry-run

``--dry-run`` validates config.yaml, loads and formats the configured
train/val JSONL files, checks them against the same rules
scripts/check_dataset.py uses, prints estimated token statistics, and
exits. It does not import torch, unsloth, transformers, or any other GPU
dependency, so it runs on any machine, including the one this repo was
built on (a Mac with no NVIDIA GPU).

Without ``--dry-run`` this script does the real training run: 4-bit QLoRA
on unsloth/llama-3-8b-bnb-4bit with Unsloth's FastLanguageModel. That path
requires a CUDA GPU and the ``[train]`` optional dependency group
(``uv pip install -e ".[train]"``), and was developed against a 24 GB
NVIDIA L4 on GCP. See docs/cost-and-gpu-notes.md before running it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llm_finetuning_recipes.config import load_config, validate_config  # noqa: E402
from llm_finetuning_recipes.formatting import dataset_token_stats, format_example  # noqa: E402
from llm_finetuning_recipes.validation import (  # noqa: E402
    format_report,
    load_jsonl,
    validate_dataset,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _resolve(path_str: str) -> Path:
    """Resolve a config-relative path against the repo root."""
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def load_and_check_data(config: dict) -> tuple[list[dict], list[dict]]:
    train_path = _resolve(config["data"]["train_path"])
    val_path = _resolve(config["data"]["val_path"])

    train_examples = load_jsonl(train_path)
    val_examples = load_jsonl(val_path)

    train_report = validate_dataset(train_examples, path=str(train_path))
    val_report = validate_dataset(val_examples, path=str(val_path))

    print(format_report(train_report))
    print()
    print(format_report(val_report))
    print()

    if not (train_report.ok and val_report.ok):
        raise SystemExit("dataset validation failed, see issues above")

    return train_examples, val_examples


def print_token_stats(examples: list[dict], max_seq_length: int, label: str) -> None:
    stats = dataset_token_stats(examples, max_seq_length=max_seq_length)
    print(f"{label} token stats (estimated, ~4 chars/token, not the real tokenizer):")
    print(
        f"  n={stats.count} min={stats.min} max={stats.max} "
        f"mean={stats.mean} median={stats.median} "
        f"over_max_seq_length={stats.over_max_seq_length}"
    )


def run_dry_run(config: dict) -> int:
    errors = validate_config(config)
    if errors:
        print("config.yaml is invalid:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("config.yaml is valid.")
    print(f"base_model: {config['base_model']}")
    print(
        f"lora: r={config['lora']['r']} alpha={config['lora']['alpha']} "
        f"dropout={config['lora']['dropout']} "
        f"target_modules={config['lora']['target_modules']}"
    )
    train_cfg = config["training"]
    effective_batch = (
        train_cfg["per_device_train_batch_size"] * train_cfg["gradient_accumulation_steps"]
    )
    print(
        f"training: lr={train_cfg['learning_rate']} "
        f"epochs={train_cfg['num_train_epochs']} "
        f"per_device_train_batch_size={train_cfg['per_device_train_batch_size']} "
        f"gradient_accumulation_steps={train_cfg['gradient_accumulation_steps']} "
        f"(effective batch {effective_batch})"
    )
    print()

    train_examples, val_examples = load_and_check_data(config)

    max_seq_length = config["training"]["max_seq_length"]
    print_token_stats(train_examples, max_seq_length, "train")
    print_token_stats(val_examples, max_seq_length, "val")
    print()
    print("Example formatted prompt (first train example):")
    print(format_example(train_examples[0]))
    print()
    print("Dry run complete. No model was loaded, no GPU was used.")
    return 0


def run_training(config: dict) -> int:  # pragma: no cover - requires a CUDA GPU
    """The real training path. Requires the `train` extra and an NVIDIA GPU.

    Not exercised by the test suite: there is no GPU in CI or on the
    machine this recipe was built on. Every line below matches the
    settings in config.yaml and the lessons in
    docs/lessons-from-five-iterations.md.
    """
    train_examples, val_examples = load_and_check_data(config)

    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    lora_cfg = config["lora"]
    training_cfg = config["training"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model"],
        max_seq_length=training_cfg["max_seq_length"],
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    # Unsloth's fused/fast LoRA forward path only applies when lora_dropout
    # is exactly 0. This recipe uses dropout 0.1 for regularization (v1's
    # failure mode was overfitting with no dropout at all, see
    # docs/lessons-from-five-iterations.md), which means Unsloth silently
    # falls back to the slower, unfused LoRA path. That is a deliberate
    # tradeoff here, not an oversight: see docs/cost-and-gpu-notes.md.
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=13,
    )

    def to_text(example: dict) -> dict:
        return {"text": format_example(example)}

    train_dataset = Dataset.from_list(train_examples).map(to_text)
    val_dataset = Dataset.from_list(val_examples).map(to_text)

    sft_config = SFTConfig(
        output_dir=config["output_dir"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        num_train_epochs=training_cfg["num_train_epochs"],
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        eval_strategy=training_cfg["eval_strategy"],
        max_seq_length=training_cfg["max_seq_length"],
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        args=sft_config,
    )

    trainer.train()
    trainer.save_model(config["output_dir"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config and data, print token stats, exit without touching a GPU",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dry_run:
        return run_dry_run(config)
    return run_training(config)


if __name__ == "__main__":
    raise SystemExit(main())
