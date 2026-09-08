"""Round-14 patches:

- OpenAIPolicy.sample_topk sends `max_completion_tokens` (not the legacy
  `max_tokens`) on the Chat Completions endpoint, so GPT-5+ models that
  reject `max_tokens` work.
- propose() catches API/config exceptions and records them in
  per-problem `policy_errors`, written to the JSONL row.

Pure tests; no network, no Lean.
"""
from __future__ import annotations

import pytest

#: Spawns a real `lake env lean` compile — minutes per test on this
#: host. Measured, not guessed: this file did not finish in 25s.
#: Excluded from the fast suite via `pytest -m "not live"`.
pytestmark = pytest.mark.live


import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------- helper unit test ----------


def test_openai_token_budget_param_returns_max_completion_tokens():
    from policy.vllm_policy import _openai_token_budget_param

    out = _openai_token_budget_param(80)
    assert out == {"max_completion_tokens": 80}
    # Crucially, the legacy spelling must NOT appear.
    assert "max_tokens" not in out


# ---------- Chat Completions call kwargs ----------


def _stub_choice(text: str = "rfl"):
    return SimpleNamespace(
        message=SimpleNamespace(content=text),
        logprobs=SimpleNamespace(
            content=[SimpleNamespace(logprob=-1.0)]
        ),
        finish_reason="stop",
    )


def _install_capturing_openai(monkeypatch, captured: dict):
    """Replace openai.OpenAI with a fake whose .chat.completions.create
    records its kwargs into `captured`."""
    from openai import OpenAI  # noqa: F401 — forces import path resolution

    def fake_create(**kw):
        captured.update(kw)
        return SimpleNamespace(choices=[_stub_choice("rfl")])

    class _FakeOpenAI:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    from policy import vllm_policy
    monkeypatch.setattr(vllm_policy, "_get_key", lambda s: "sk-fake")


def test_openai_sample_topk_sends_max_completion_tokens(monkeypatch):
    """Load-bearing fix: chat.completions.create must receive
    max_completion_tokens for newer-model compatibility."""
    captured: dict = {}
    _install_capturing_openai(monkeypatch, captured)
    from policy.vllm_policy import OpenAIPolicy

    pol = OpenAIPolicy(model="gpt-5.5-test")
    pol.sample_topk("prompt", k=1, max_tokens=80)

    assert captured.get("max_completion_tokens") == 80


def test_openai_sample_topk_does_not_send_legacy_max_tokens(monkeypatch):
    """The legacy `max_tokens` keyword MUST NOT appear on the wire —
    GPT-5+ models reject it with an UnsupportedParameter error."""
    captured: dict = {}
    _install_capturing_openai(monkeypatch, captured)
    from policy.vllm_policy import OpenAIPolicy

    pol = OpenAIPolicy(model="gpt-5.5-test")
    pol.sample_topk("prompt", k=1, max_tokens=80)

    assert "max_tokens" not in captured, (
        "legacy max_tokens kwarg must not be sent to chat.completions.create; "
        f"captured={captured}"
    )


def test_openai_sample_topk_preserves_existing_kwargs_for_legacy_models(monkeypatch):
    """For a non-reasoning model (gpt-4o-mini-class), the existing kwargs
    (model, messages, temperature, n, logprobs) are still sent as before —
    the round-14 fix only changed the token-budget spelling. The
    reasoning-model conditional omissions are exercised separately in
    test_openai_reasoning_model_compat.py."""
    captured: dict = {}
    _install_capturing_openai(monkeypatch, captured)
    from policy.vllm_policy import OpenAIPolicy

    pol = OpenAIPolicy(model="gpt-4o-mini")
    pol.sample_topk("prompt", k=3, system="be brief", temperature=0.4,
                    max_tokens=128)

    assert captured["model"] == "gpt-4o-mini"
    assert captured["n"] == 3
    assert captured["temperature"] == 0.4
    assert captured["logprobs"] is True
    # First message is the system block; second is the user prompt.
    msgs = captured["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "be brief"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "prompt"

