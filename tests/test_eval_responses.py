from llm_finetuning_recipes.eval_scoring import (
    contains_premature_close,
    score_response,
    score_responses,
)


def test_short_response_flagged_length_not_ok():
    score = score_response({"id": "r1", "text": "Sure."})
    assert score.length_ok is False
    assert score.word_count == 1


def test_in_band_response_flagged_length_ok():
    score = score_response({"id": "r2", "text": "What day this week works best for you?"})
    assert score.length_ok is True
    assert score.single_clause_ok is True
    assert score.is_question is True


def test_two_clause_response_fails_single_clause():
    score = score_response(
        {"id": "r3", "text": "We can fit you in Thursday. Does that work for you?"}
    )
    assert score.single_clause_ok is False


def test_premature_close_detection():
    assert contains_premature_close("Thanks for calling, have a great day, goodbye now.")
    assert not contains_premature_close("What day this week works best for you?")


def test_score_responses_summary_counts():
    items = [
        {"id": "a", "text": "What day this week works best for you?"},
        {"id": "b", "text": "Sure."},
        {"id": "c", "text": "Have a great day!"},
    ]
    summary = score_responses(items)
    assert summary.total == 3
    assert summary.length_ok_count == 1
    assert summary.premature_close_count == 1
