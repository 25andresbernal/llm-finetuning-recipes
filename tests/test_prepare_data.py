from llm_finetuning_recipes.data_gen import generate_dataset
from llm_finetuning_recipes.validation import validate_dataset


def test_determinism_same_seed_same_output():
    a = generate_dataset(num_examples=120, seed=13)
    b = generate_dataset(num_examples=120, seed=13)
    assert a.train == b.train
    assert a.val == b.val


def test_different_seed_different_output():
    a = generate_dataset(num_examples=120, seed=13)
    b = generate_dataset(num_examples=120, seed=99)
    assert a.train != b.train


def test_total_count_is_close_to_requested():
    dataset = generate_dataset(num_examples=200, seed=1)
    total = len(dataset.train) + len(dataset.val)
    # Conversations are kept whole, so the total can overshoot by up to one
    # scenario's worth of turns, but should never fall short.
    assert total >= 200
    assert total <= 200 + 12


def test_split_has_no_overlap_and_roughly_matches_val_fraction():
    dataset = generate_dataset(num_examples=300, val_fraction=0.15, seed=13)
    total = len(dataset.train) + len(dataset.val)
    val_share = len(dataset.val) / total
    assert 0.05 < val_share < 0.30

    train_ids = {ex["id"] for ex in dataset.train}
    val_ids = {ex["id"] for ex in dataset.val}
    assert train_ids.isdisjoint(val_ids)


def test_generated_dataset_passes_validation():
    dataset = generate_dataset(num_examples=250, seed=42)
    train_report = validate_dataset(dataset.train, path="train")
    val_report = validate_dataset(dataset.val, path="val")
    assert train_report.ok, train_report.all_issues
    assert val_report.ok, val_report.all_issues


def test_no_duplicate_prompts_across_train_and_val():
    # A conversation's turns should land entirely in one split, so the same
    # (system, history) prompt should never appear in both.
    dataset = generate_dataset(num_examples=250, seed=42)

    def prompt_key(ex):
        return (ex["system"], tuple((t["speaker"], t["text"]) for t in ex["history"]))

    train_keys = {prompt_key(ex) for ex in dataset.train}
    val_keys = {prompt_key(ex) for ex in dataset.val}
    assert train_keys.isdisjoint(val_keys)


def test_invalid_num_examples_raises():
    import pytest

    with pytest.raises(ValueError):
        generate_dataset(num_examples=0)
    with pytest.raises(ValueError):
        generate_dataset(num_examples=10, val_fraction=1.0)
