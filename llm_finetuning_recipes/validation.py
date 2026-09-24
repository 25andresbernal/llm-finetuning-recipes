"""Validation rules for the multi-turn scheduling dataset.

This is the part of the repo meant to be read closely: it is a small,
testable definition of what makes a training example good, not a wrapper
around a training run that cannot be checked without a GPU.

Rules enforced:

1. **Format**: every line is a JSON object with an ``id`` (str), ``system``
   (non-empty str), ``history`` (list of ``{"speaker", "text"}`` objects),
   and ``target`` (non-empty str).
2. **Speaker sanity**: every ``speaker`` value is ``"agent"`` or
   ``"caller"``. The history alternates strictly starting with ``"agent"``,
   and if the history is non-empty its last turn is ``"caller"`` (since the
   target is the agent's reply to that caller turn).
3. **Target length**: the target is between ``MIN_TARGET_WORDS`` and
   ``MAX_TARGET_WORDS`` words, counted by whitespace split.
4. **One question or one statement, never both**: the target has exactly
   one terminal punctuation mark (``.``, ``!``, or ``?``) and it is the
   final character of the string. A target with an interior ``.`` or ``?``
   is two clauses glued together, which is exactly the "restate, then ask"
   pattern that produced monologuing on live calls (see
   docs/lessons-from-five-iterations.md).
5. **No duplicate prompts**: no two examples share the same
   ``(system, history)`` pair. A duplicate prompt with a different target
   teaches the model two different correct answers to the same input,
   which is a contradiction the model resolves by hedging or drifting.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

VALID_SPEAKERS = {"agent", "caller"}
MIN_TARGET_WORDS = 5
MAX_TARGET_WORDS = 25
TERMINAL_CHARS = ".!?"


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts.

    Raises ValueError with a line number if any line is not valid JSON.
    """
    examples = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
    return examples


def word_count(text: str) -> int:
    return len(text.split())


def is_single_clause(text: str) -> bool:
    """True if `text` has exactly one terminal mark, and it is the last char."""
    stripped = text.strip()
    if not stripped:
        return False
    terminal_positions = [i for i, c in enumerate(stripped) if c in TERMINAL_CHARS]
    if len(terminal_positions) != 1:
        return False
    return terminal_positions[0] == len(stripped) - 1


def is_question(text: str) -> bool:
    return text.strip().endswith("?")


@dataclass
class ExampleIssue:
    index: int
    example_id: str | None
    message: str


@dataclass
class DatasetReport:
    path: str
    total: int = 0
    format_issues: list[ExampleIssue] = field(default_factory=list)
    speaker_issues: list[ExampleIssue] = field(default_factory=list)
    length_issues: list[ExampleIssue] = field(default_factory=list)
    clause_issues: list[ExampleIssue] = field(default_factory=list)
    duplicate_issues: list[ExampleIssue] = field(default_factory=list)
    target_word_counts: list[int] = field(default_factory=list)
    num_questions: int = 0
    num_statements: int = 0

    @property
    def all_issues(self) -> list[ExampleIssue]:
        return (
            self.format_issues
            + self.speaker_issues
            + self.length_issues
            + self.clause_issues
            + self.duplicate_issues
        )

    @property
    def ok(self) -> bool:
        return len(self.all_issues) == 0 and self.total > 0

    def length_stats(self) -> dict:
        if not self.target_word_counts:
            return {}
        counts = self.target_word_counts
        buckets = {"<5": 0, "5-15": 0, "16-25": 0, ">25": 0}
        for c in counts:
            if c < 5:
                buckets["<5"] += 1
            elif c <= 15:
                buckets["5-15"] += 1
            elif c <= 25:
                buckets["16-25"] += 1
            else:
                buckets[">25"] += 1
        return {
            "min": min(counts),
            "max": max(counts),
            "mean": round(statistics.mean(counts), 2),
            "median": statistics.median(counts),
            "buckets": buckets,
        }


def _validate_format(example: dict, index: int) -> tuple[bool, list[str]]:
    errors = []
    for key, expected in (("id", str), ("system", str), ("target", str)):
        if key not in example:
            errors.append(f"missing key '{key}'")
        elif not isinstance(example[key], expected):
            errors.append(f"key '{key}' must be a string")
    if "history" not in example:
        errors.append("missing key 'history'")
    elif not isinstance(example["history"], list):
        errors.append("key 'history' must be a list")
    else:
        for i, turn in enumerate(example["history"]):
            if not isinstance(turn, dict) or "speaker" not in turn or "text" not in turn:
                errors.append(f"history[{i}] must be an object with 'speaker' and 'text'")
    if "system" in example and isinstance(example["system"], str) and not example["system"].strip():
        errors.append("'system' must not be empty")
    if "target" in example and isinstance(example["target"], str) and not example["target"].strip():
        errors.append("'target' must not be empty")
    return (len(errors) == 0, errors)


def _validate_speakers(example: dict) -> list[str]:
    errors = []
    history = example.get("history", [])
    for i, turn in enumerate(history):
        speaker = turn.get("speaker")
        if speaker not in VALID_SPEAKERS:
            errors.append(f"history[{i}] has invalid speaker '{speaker}'")
    if not history:
        return errors
    if history[0].get("speaker") != "agent":
        errors.append("history must start with the agent's opening turn")
    for i in range(1, len(history)):
        prev = history[i - 1].get("speaker")
        cur = history[i].get("speaker")
        if prev in VALID_SPEAKERS and cur in VALID_SPEAKERS and prev == cur:
            errors.append(f"history[{i}] repeats the same speaker as history[{i - 1}]")
    if history[-1].get("speaker") != "caller":
        errors.append("history must end on a caller turn, since the target is the agent's reply")
    return errors


def validate_dataset(examples: list[dict], path: str = "<in-memory>") -> DatasetReport:
    report = DatasetReport(path=path, total=len(examples))
    seen_prompts: dict[tuple, int] = {}

    for i, example in enumerate(examples):
        example_id = example.get("id") if isinstance(example, dict) else None
        ok, format_errors = _validate_format(example, i)
        if not ok:
            for msg in format_errors:
                report.format_issues.append(ExampleIssue(i, example_id, msg))
            # Can't safely check the rest of the rules without a valid shape.
            continue

        for msg in _validate_speakers(example):
            report.speaker_issues.append(ExampleIssue(i, example_id, msg))

        target = example["target"]
        wc = word_count(target)
        report.target_word_counts.append(wc)
        if wc < MIN_TARGET_WORDS or wc > MAX_TARGET_WORDS:
            report.length_issues.append(
                ExampleIssue(
                    i,
                    example_id,
                    f"target has {wc} words, expected {MIN_TARGET_WORDS}-{MAX_TARGET_WORDS}",
                )
            )

        if not is_single_clause(target):
            report.clause_issues.append(
                ExampleIssue(i, example_id, f"target is not a single clause: {target!r}")
            )
        elif is_question(target):
            report.num_questions += 1
        else:
            report.num_statements += 1

        prompt_key = (
            example["system"],
            tuple((t.get("speaker"), t.get("text")) for t in example.get("history", [])),
        )
        if prompt_key in seen_prompts:
            first_index = seen_prompts[prompt_key]
            report.duplicate_issues.append(
                ExampleIssue(i, example_id, f"duplicate prompt, first seen at index {first_index}")
            )
        else:
            seen_prompts[prompt_key] = i

    return report


def format_report(report: DatasetReport) -> str:
    lines = []
    lines.append(f"Dataset: {report.path}")
    lines.append(f"Total examples: {report.total}")
    lines.append("")

    stats = report.length_stats()
    if stats:
        lines.append(
            f"Target length (words): min={stats['min']} max={stats['max']} "
            f"mean={stats['mean']} median={stats['median']}"
        )
        buckets = stats["buckets"]
        lines.append("  distribution: " + ", ".join(f"{k}={v}" for k, v in buckets.items()))
    lines.append(
        f"Turn shape: {report.num_questions} questions, {report.num_statements} statements"
    )
    lines.append("")

    def _section(name: str, issues: list[ExampleIssue]) -> None:
        lines.append(f"{name}: {len(issues)} issue(s)")
        for issue in issues[:10]:
            ref = issue.example_id or f"index {issue.index}"
            lines.append(f"  - [{ref}] {issue.message}")
        if len(issues) > 10:
            lines.append(f"  ... and {len(issues) - 10} more")

    _section("Format", report.format_issues)
    _section("Speaker labels", report.speaker_issues)
    _section("Target length", report.length_issues)
    _section("One question or one statement", report.clause_issues)
    _section("Duplicate prompts", report.duplicate_issues)
    lines.append("")
    lines.append("PASS" if report.ok else "FAIL")
    return "\n".join(lines)
