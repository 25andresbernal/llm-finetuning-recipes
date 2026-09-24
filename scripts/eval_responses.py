#!/usr/bin/env python3
"""Score a JSONL file of model responses for length, turn shape, and premature closes.

Usage:
    uv run python scripts/eval_responses.py data/sample/responses_sample.jsonl

Input format: one JSON object per line with at least ``id`` and ``text``.
This runs entirely offline; it does not call a model or an LLM judge, it
only scores text that is already on disk. See
llm_finetuning_recipes/eval_scoring.py for what each column means.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_finetuning_recipes.eval_scoring import format_table, score_responses  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSONL file of {id, text} responses")
    args = parser.parse_args()

    items = load_jsonl(args.path)
    summary = score_responses(items)
    print(format_table(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
