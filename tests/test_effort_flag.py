"""`--effort` -> `output_config.effort` on the Anthropic adapter.

Effort controls how deep adaptive thinking goes. The API default is
"high", so an unset run and an `--effort high` run are the same run —
these tests pin that, and pin the two ways the field can be silently
lost: overwritten by a task budget sharing `output_config`, or sent to
a model that 400s on it (which would make the learned-rejection net
drop `output_config` for the whole process).

No network: `_build_kwargs` is pure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policy.vllm_policy import (  # noqa: E402
    AnthropicPolicy, check_effort_support,
)


def _kwargs(policy):
    return policy._build_kwargs(
        system="s", prompt="p", temperature=0.4, max_tokens=16000)


@pytest.fixture
def policy(monkeypatch):
    """An AnthropicPolicy without touching keyring or the network."""
    monkeypatch.setattr("policy.vllm_policy._get_key", lambda _p: "k")
    monkeypatch.setattr(
        "anthropic.Anthropic", lambda **kw: object())
    return AnthropicPolicy(model="claude-opus-4-8")


def test_unset_effort_sends_no_output_config(policy):
    # Unset must stay off the wire: the API reads a missing field as
    # "high", so sending nothing is what reproduces every prior run.
    assert "output_config" not in _kwargs(policy)


def test_effort_max_is_sent(policy):
    policy.effort = "max"
    assert _kwargs(policy)["output_config"] == {"effort": "max"}


def test_effort_merges_with_task_budget(policy):
    # Both features share output_config; assigning would drop one.
    policy.effort = "max"
    policy.task_budget = 24000
    oc = _kwargs(policy)["output_config"]
    assert oc["effort"] == "max"
    assert oc["task_budget"] == {"type": "tokens", "total": 24000}
    assert _kwargs(policy)["betas"] == ["task-budgets-2026-03-13"]


def test_adaptive_thinking_still_requested_alongside_effort(policy):
    # Effort tunes thinking; it must not replace the opt-in. Omitting
    # `thinking` on Opus 4.8 means NO thinking at all.
    policy.effort = "max"
    assert _kwargs(policy)["thinking"] == {"type": "adaptive"}


def test_learned_rejection_drops_effort_and_its_betas(policy):
    policy.effort = "max"
    policy.task_budget = 24000
    policy._rejected_params.add("output_config")
    k = _kwargs(policy)
    assert "output_config" not in k
    assert "betas" not in k


@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
def test_all_levels_accepted_on_opus_4_8(level):
    assert check_effort_support("claude-opus-4-8", level) == level


def test_xhigh_downgrades_on_4_6(capsys):
    # xhigh arrived with Opus 4.7; 4.6 400s on it.
    assert check_effort_support("claude-opus-4-6", "xhigh") == "high"
    assert "xhigh" in capsys.readouterr().out


def test_effort_dropped_for_model_without_support(capsys):
    assert check_effort_support("claude-haiku-4-5", "max") is None
    assert "does not accept" in capsys.readouterr().out


def test_unknown_level_is_a_launch_error():
    with pytest.raises(SystemExit):
        check_effort_support("claude-opus-4-8", "highest")


def test_none_effort_is_passthrough():
    assert check_effort_support("claude-opus-4-8", None) is None
