"""The empty-response retry must BOUND thinking, not just raise the cap.

Regression test for the miniF2F-18 void cells (2026-08-26). Two hard
problems -- `aime_1995_p7` and `imo_2019_p1` -- each produced three
EMPTY sketches. Every billed call showed stop_reason=max_tokens with
out=16000 then out=32000, text=0 and thinking_chars=0: with
`thinking={"type": "adaptive"}` the model spent the whole output budget
inside a thinking block and never opened a text block, and because the
platform withholds reasoning text the response logged as nothing at all.

The retry that existed at the time only DOUBLED max_tokens, which hands
adaptive thinking more room to expand into. Both cells voided and one
cost $7.32 for zero result rows.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policy.vllm_policy import AnthropicPolicy  # noqa: E402


def _block(kind: str, **fields):
    b = types.SimpleNamespace(type=kind)
    for k, v in fields.items():
        setattr(b, k, v)
    return b


def _resp(blocks, stop_reason):
    return types.SimpleNamespace(content=blocks, stop_reason=stop_reason,
                                 usage=None)


class _Recorder:
    """Stands in for `_call_with_param_net`, capturing each call's kwargs.

    First call mimics the observed failure: budget exhausted inside
    thinking, no text block. Later calls return text, so the test can
    assert on what the RETRY asked for.
    """

    def __init__(self, replies):
        self.calls: list[dict] = []
        self._replies = list(replies)

    def __call__(self, kwargs):
        self.calls.append(dict(kwargs))
        return self._replies.pop(0)


@pytest.fixture
def policy(monkeypatch):
    p = AnthropicPolicy.__new__(AnthropicPolicy)
    p.model = "claude-opus-4-8"
    p._rejected_params = set()
    p.continuation_rounds = 0
    p.task_budget = 0
    p.last_thinking = None
    monkeypatch.setattr(p, "_check_spend_cap", lambda: None)
    return p


def _adaptive_kwargs(max_tokens=16000):
    return {
        "model": "claude-opus-4-8",
        "max_tokens": max_tokens,
        "system": "",
        "messages": [{"role": "user", "content": "sketch this"}],
        "thinking": {"type": "adaptive"},
    }


def test_retry_shortens_thinking_via_effort(policy, monkeypatch):
    """The retry must ask for LOW EFFORT, keeping adaptive thinking.

    Opus 4.8 rejects `{"type": "enabled", "budget_tokens": N}` with
    '"thinking.type.enabled" is not supported for this model. Use
    "thinking.type.adaptive" and "output_config.effort"'. Sending the
    enabled form "works" only by accident: the 400 makes the param net
    drop `thinking` for the whole process, so every later call in the
    run silently loses thinking.
    """
    rec = _Recorder([
        # The observed failure: thinking-only, budget exhausted.
        _resp([_block("thinking", thinking="", signature="x" * 8)],
              "max_tokens"),
        _resp([_block("text", text="{\"haves\": []}")], "end_turn"),
    ])
    monkeypatch.setattr(policy, "_call_with_param_net", rec)

    text = policy._one_call(_adaptive_kwargs())

    assert text == "{\"haves\": []}"
    assert len(rec.calls) == 2, "an empty response must trigger one retry"

    retry = rec.calls[1]
    assert retry["thinking"] == {"type": "adaptive"}, (
        "thinking must stay adaptive — the enabled form is a 400 on this "
        "model and costs the whole process its thinking")
    assert retry["output_config"]["effort"] == "low", (
        "retry did not shorten thinking; adaptive will expand to fill the "
        "larger budget and return empty again")


def test_first_call_still_uses_adaptive(policy, monkeypatch):
    """Bounding is a RETRY behaviour; the first call is left alone."""
    rec = _Recorder([_resp([_block("text", text="ok")], "end_turn")])
    monkeypatch.setattr(policy, "_call_with_param_net", rec)

    assert policy._one_call(_adaptive_kwargs()) == "ok"
    assert len(rec.calls) == 1
    assert rec.calls[0]["thinking"] == {"type": "adaptive"}


def test_retry_raises_the_cap_as_well(policy, monkeypatch):
    rec = _Recorder([
        _resp([], "max_tokens"),
        _resp([_block("text", text="ok")], "end_turn"),
    ])
    monkeypatch.setattr(policy, "_call_with_param_net", rec)

    policy._one_call(_adaptive_kwargs(max_tokens=16000))

    assert rec.calls[1]["max_tokens"] == 32000


def test_refusal_is_final_and_not_retried(policy, monkeypatch):
    """A refusal is a decision, not a truncation — never pay twice."""
    rec = _Recorder([_resp([], "refusal")])
    monkeypatch.setattr(policy, "_call_with_param_net", rec)

    assert policy._one_call(_adaptive_kwargs()) == ""
    assert len(rec.calls) == 1


def test_non_adaptive_thinking_is_left_untouched(policy, monkeypatch):
    """An explicit budget the caller chose must not be overwritten."""
    kwargs = _adaptive_kwargs()
    kwargs["thinking"] = {"type": "enabled", "budget_tokens": 4096}
    # (no output_config is added for a non-adaptive request)
    rec = _Recorder([
        _resp([], "max_tokens"),
        _resp([_block("text", text="ok")], "end_turn"),
    ])
    monkeypatch.setattr(policy, "_call_with_param_net", rec)

    policy._one_call(kwargs)

    assert rec.calls[1]["thinking"] == {"type": "enabled",
                                        "budget_tokens": 4096}
    assert "output_config" not in rec.calls[1]


def test_effort_not_sent_once_output_config_rejected(policy, monkeypatch):
    """If the API already 400'd output_config, do not send it again."""
    policy._rejected_params.add("output_config")
    rec = _Recorder([
        _resp([], "max_tokens"),
        _resp([_block("text", text="ok")], "end_turn"),
    ])
    monkeypatch.setattr(policy, "_call_with_param_net", rec)

    policy._one_call(_adaptive_kwargs())

    assert "output_config" not in rec.calls[1]
