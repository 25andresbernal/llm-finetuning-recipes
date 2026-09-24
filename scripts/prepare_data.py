#!/usr/bin/env python3
"""Generate the synthetic scheduling-agent dataset and write a train/val split.

Usage:
    uv run python scripts/prepare_data.py
    uv run python scripts/prepare_data.py --num-examples 320 --seed 7 --out-dir data/sample

Everything this script writes is synthetic: fictional business, fictional
callers, fictional repair jobs. See llm_finetuning_recipes/data_gen.py for
the scenario templates and the reasoning behind the multi-turn, short-target
shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_finetuning_recipes.data_gen import generate_dataset  # noqa: E402


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-examples", type=int, default=300, help="target total examples (200-400)"
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out-dir", type=Path, default=Path("data/sample"))
    args = parser.parse_args()

    if not 200 <= args.num_examples <= 400:
        print(
            f"warning: --num-examples {args.num_examples} is outside the intended 200-400 range",
            file=sys.stderr,
        )

    dataset = generate_dataset(
        num_examples=args.num_examples, val_fraction=args.val_fraction, seed=args.seed
    )

    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val.jsonl"
    write_jsonl(dataset.train, train_path)
    write_jsonl(dataset.val, val_path)

    total = len(dataset.train) + len(dataset.val)
    print(f"wrote {len(dataset.train)} train examples to {train_path}")
    print(f"wrote {len(dataset.val)} val examples to {val_path}")
    print(f"total {total} examples, seed {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
