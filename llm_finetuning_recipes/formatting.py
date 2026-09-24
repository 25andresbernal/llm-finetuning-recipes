"""Prompt formatting and token-count estimation for the training recipe.

The real training run tokenizes with the base model's actual tokenizer,
which requires transformers and a downloaded tokenizer file. ``--dry-run``
is meant to work on a laptop with no network access and no GPU packages
installed, so it uses a plain-text approximation instead: roughly one
token per four characters, which is the commonly cited rule of thumb for
English text with byte-pair-encoding tokenizers such as Llama-3's. It is
not exact. It exists so a config and a dataset can be sanity-checked for
"this is roughly the right shape and will roughly fit in max_seq_length"
before spending any GPU time, not to replace the real tokenizer count that
happens during actual training.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

CHARS_PER_TOKEN_ESTIMATE = 4.0

ROLE_MAP = {"agent": "assistant", "caller": "user"}


def format_example(example: dict) -> str:
    """Render one example as a Llama-3-style chat prompt ending in the target.

    This mirrors the chat template used at real training time closely
    enough for length estimation and human inspection. The actual training
    path in train.py hands the same (system, history, target) fields to
    the tokenizer's own chat template, which is the source of truth for
    what the model actually sees.
    """
    parts = ["<|begin_of_text|>"]
    parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{example['system']}<|eot_id|>")
    for turn in example.get("history", []):
        role = ROLE_MAP.get(turn["speaker"], turn["speaker"])
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{turn['text']}<|eot_id|>")
    parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{example['target']}<|eot_id|>")
    return "".join(parts)


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN_ESTIMATE))


@dataclass
class TokenStats:
    count: int
    min: int
    max: int
    mean: float
    median: float
    over_max_seq_length: int


def dataset_token_stats(examples: list[dict], max_seq_length: int) -> TokenStats:
    lengths = [estimate_tokens(format_example(ex)) for ex in examples]
    if not lengths:
        return TokenStats(0, 0, 0, 0.0, 0.0, 0)
    return TokenStats(
        count=len(lengths),
        min=min(lengths),
        max=max(lengths),
        mean=round(statistics.mean(lengths), 1),
        median=statistics.median(lengths),
        over_max_seq_length=sum(1 for length in lengths if length > max_seq_length),
    )
