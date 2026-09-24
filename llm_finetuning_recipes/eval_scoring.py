"""Offline scoring for a JSONL file of model responses.

This scores standalone agent responses, the kind you would collect from a
call log or a batch of generations, on three cheap, rule-based signals:

- **Length**: word count, and whether it falls in the same 5-25 word band
  the training data targets. A model that has drifted back toward long
  responses will show up here immediately, no judge model required.
- **One question or one statement**: reuses the same single-clause rule
  from ``llm_finetuning_recipes.validation``. A model producing
  "We can fit you in Tuesday. Does that work?" fails this even though
  each half looks fine in isolation.
- **Premature-close phrases**: a response is flagged if it contains a
  phrase that tries to end the call ("goodbye," "have a great day," and
  similar). This is a coarse heuristic, not a judgment on whether the call
  was actually finished: it does not have the rest of the conversation, so
  a genuine, correctly timed goodbye at the real end of a call will also be
  flagged. Treat a flag as "look at this one," not "this one is wrong."
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .validation import MAX_TARGET_WORDS, MIN_TARGET_WORDS, is_single_clause, word_count

PREMATURE_CLOSE_PHRASES = [
    "goodbye",
    "bye for now",
    "bye now",
    "have a great day",
    "have a nice day",
    "have a good one",
    "take care now",
    "that will be all for today",
    "thanks for calling, goodbye",
]


def contains_premature_close(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in PREMATURE_CLOSE_PHRASES)


@dataclass
class ResponseScore:
    response_id: str
    text: str
    word_count: int
    length_ok: bool
    single_clause_ok: bool
    is_question: bool
    premature_close: bool


@dataclass
class ScoringSummary:
    scores: list[ResponseScore] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.scores)

    @property
    def length_ok_count(self) -> int:
        return sum(1 for s in self.scores if s.length_ok)

    @property
    def single_clause_ok_count(self) -> int:
        return sum(1 for s in self.scores if s.single_clause_ok)

    @property
    def premature_close_count(self) -> int:
        return sum(1 for s in self.scores if s.premature_close)

    @property
    def mean_word_count(self) -> float:
        if not self.scores:
            return 0.0
        return round(sum(s.word_count for s in self.scores) / len(self.scores), 2)


def score_response(item: dict) -> ResponseScore:
    text = item.get("text", "")
    response_id = item.get("id", "?")
    wc = word_count(text)
    return ResponseScore(
        response_id=response_id,
        text=text,
        word_count=wc,
        length_ok=MIN_TARGET_WORDS <= wc <= MAX_TARGET_WORDS,
        single_clause_ok=is_single_clause(text),
        is_question=text.strip().endswith("?"),
        premature_close=contains_premature_close(text),
    )


def score_responses(items: list[dict]) -> ScoringSummary:
    return ScoringSummary(scores=[score_response(item) for item in items])


def format_table(summary: ScoringSummary) -> str:
    header = (
        f"{'id':<12} {'words':>6} {'len_ok':>7} {'1clause':>8} {'question':>9} {'prem_close':>11}"
    )
    rows = [header, "-" * len(header)]
    for s in summary.scores:
        rows.append(
            f"{s.response_id:<12} {s.word_count:>6} {str(s.length_ok):>7} "
            f"{str(s.single_clause_ok):>8} {str(s.is_question):>9} "
            f"{str(s.premature_close):>11}"
        )
    rows.append("-" * len(header))
    rows.append(
        f"{summary.total} responses | "
        f"{summary.length_ok_count}/{summary.total} in length band | "
        f"{summary.single_clause_ok_count}/{summary.total} single-clause | "
        f"{summary.premature_close_count}/{summary.total} flagged premature-close | "
        f"mean words {summary.mean_word_count}"
    )
    return "\n".join(rows)
