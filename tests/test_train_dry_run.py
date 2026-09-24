import subprocess
import sys
from pathlib import Path

import pytest

from llm_finetuning_recipes.config import load_config, validate_config
from llm_finetuning_recipes.formatting import dataset_token_stats, estimate_tokens, format_example

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "recipes" / "conversational-agent-qlora" / "config.yaml"
TRAIN_SCRIPT = REPO_ROOT / "recipes" / "conversational-agent-qlora" / "train.py"


def test_committed_config_is_valid():
    config = load_config(CONFIG_PATH)
    assert validate_config(config) == []


def test_validate_config_reports_missing_top_level_keys():
    errors = validate_config({"base_model": "x"})
    assert any("missing top-level keys" in e for e in errors)


def test_validate_config_reports_bad_lora_values():
    config = load_config(CONFIG_PATH)
    config["lora"]["r"] = -1
    config["lora"]["dropout"] = 1.5
    errors = validate_config(config)
    assert any("lora.r" in e for e in errors)
    assert any("lora.dropout" in e for e in errors)


def test_validate_config_reports_bad_scheduler():
    config = load_config(CONFIG_PATH)
    config["training"]["lr_scheduler_type"] = "not-a-real-schedule"
    errors = validate_config(config)
    assert any("lr_scheduler_type" in e for e in errors)


def test_format_example_ends_with_target():
    example = {
        "system": "sys",
        "history": [
            {"speaker": "agent", "text": "hello"},
            {"speaker": "caller", "text": "hi"},
        ],
        "target": "what day works?",
    }
    text = format_example(example)
    assert text.startswith("<|begin_of_text|>")
    assert text.endswith("what day works?<|eot_id|>")
    assert "<|start_header_id|>user<|end_header_id|>" in text  # caller mapped to user
    assert "<|start_header_id|>assistant<|end_header_id|>" in text  # agent mapped to assistant


def test_estimate_tokens_is_a_rough_length_proxy():
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens("") == 1  # never zero, avoids downstream divide-by-zero


def test_dataset_token_stats_flags_over_budget_examples():
    examples = [
        {"system": "s", "history": [], "target": "short target here for the test"},
        {"system": "s", "history": [], "target": "x " * 5000},
    ]
    stats = dataset_token_stats(examples, max_seq_length=50)
    assert stats.count == 2
    assert stats.over_max_seq_length == 1


@pytest.mark.parametrize("flag", ["--dry-run"])
def test_train_script_dry_run_exits_zero_and_never_needs_a_gpu(flag):
    # torch/unsloth are intentionally not installed in this environment
    # (they're a separate [train] extra for Linux+CUDA machines). If
    # --dry-run imported either, this subprocess would fail with
    # ModuleNotFoundError instead of exiting cleanly.
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), flag],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dry run complete" in result.stdout
    assert "PASS" in result.stdout
    assert "torch" not in result.stderr.lower()
