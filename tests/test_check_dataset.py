from llm_finetuning_recipes.validation import (
    is_single_clause,
    validate_dataset,
    word_count,
)

VALID_EXAMPLE = {
    "id": "ex-1",
    "system": "You are a scheduling assistant.",
    "history": [
        {"speaker": "agent", "text": "Thanks for calling, how can I help you today?"},
        {"speaker": "caller", "text": "I need to book a repair visit."},
    ],
    "target": "What day this week works best for you?",
}


def test_valid_example_passes():
    report = validate_dataset([VALID_EXAMPLE])
    assert report.ok
    assert report.total == 1


def test_missing_key_is_a_format_issue():
    bad = dict(VALID_EXAMPLE)
    del bad["target"]
    report = validate_dataset([bad])
    assert not report.ok
    assert len(report.format_issues) == 1


def test_empty_target_is_a_format_issue():
    bad = {**VALID_EXAMPLE, "target": ""}
    report = validate_dataset([bad])
    assert not report.ok
    assert any("empty" in issue.message for issue in report.format_issues)


def test_target_too_short_is_a_length_issue():
    bad = {**VALID_EXAMPLE, "target": "Sure, okay."}
    report = validate_dataset([bad])
    assert not report.ok
    assert len(report.length_issues) == 1


def test_target_too_long_is_a_length_issue():
    long_target = " ".join(["word"] * 30) + "."
    bad = {**VALID_EXAMPLE, "target": long_target}
    report = validate_dataset([bad])
    assert not report.ok
    assert len(report.length_issues) == 1


def test_statement_then_question_is_a_clause_issue():
    bad = {**VALID_EXAMPLE, "target": "We can fit you in Tuesday. Does that work?"}
    report = validate_dataset([bad])
    assert not report.ok
    assert len(report.clause_issues) == 1


def test_pure_question_is_fine():
    assert is_single_clause("What day works best for you?")


def test_pure_statement_is_fine():
    assert is_single_clause("You're all set for Tuesday at 2 PM.")


def test_invalid_speaker_label_is_a_speaker_issue():
    bad = {
        **VALID_EXAMPLE,
        "history": [{"speaker": "robot", "text": "Beep boop."}],
    }
    report = validate_dataset([bad])
    assert not report.ok
    assert len(report.speaker_issues) >= 1


def test_history_must_start_with_agent():
    bad = {
        **VALID_EXAMPLE,
        "history": [{"speaker": "caller", "text": "Hello?"}],
    }
    report = validate_dataset([bad])
    assert not report.ok
    assert any("start" in issue.message for issue in report.speaker_issues)


def test_history_must_end_on_caller():
    bad = {
        **VALID_EXAMPLE,
        "history": [
            {"speaker": "agent", "text": "Thanks for calling, how can I help?"},
            {"speaker": "caller", "text": "I need a repair."},
            {"speaker": "agent", "text": "What day works for you?"},
        ],
    }
    report = validate_dataset([bad])
    assert not report.ok
    assert any("end on a caller" in issue.message for issue in report.speaker_issues)


def test_repeated_speaker_is_flagged():
    bad = {
        **VALID_EXAMPLE,
        "history": [
            {"speaker": "agent", "text": "Thanks for calling, how can I help?"},
            {"speaker": "agent", "text": "Are you still there?"},
        ],
    }
    report = validate_dataset([bad])
    assert not report.ok
    assert any("repeats" in issue.message for issue in report.speaker_issues)


def test_duplicate_prompt_is_flagged():
    examples = [
        VALID_EXAMPLE,
        {**VALID_EXAMPLE, "id": "ex-2", "target": "A different answer entirely."},
    ]
    report = validate_dataset(examples)
    assert not report.ok
    assert len(report.duplicate_issues) == 1


def test_word_count_helper():
    assert word_count("one two three") == 3
    assert word_count("") == 0
