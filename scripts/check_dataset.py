#!/usr/bin/env python3
"""Validate a JSONL dataset against the rules in llm_finetuning_recipes/validation.py.

Usage:
    uv run python scripts/check_dataset.py data/sample/train.jsonl data/sample/val.jsonl

Exits 0 if every file passes, 1 otherwise. This is the honest, testable
core of the repo: every rule it checks is also unit-tested directly in
tests/test_check_dataset.py, independent of whatever the generator
happens to produce.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_finetuning_recipes.validation import (  # noqa: E402
    format_report,
    load_jsonl,
    validate_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="JSONL file(s) to validate")
    args = parser.parse_args()

    all_ok = True
    for path in args.paths:
        examples = load_jsonl(path)
        report = validate_dataset(examples, path=str(path))
        print(format_report(report))
        print()
        all_ok = all_ok and report.ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
