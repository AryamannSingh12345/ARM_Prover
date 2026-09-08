"""Round-16 patches:

- `--policy-max-tokens` is configurable and flows through to
  `policy.sample_topk(max_tokens=…)` and into the JSONL row.
- OpenAIPolicy populates `Sample.finish_reason`, `Sample.completion_tokens`,
  and `Sample.reasoning_tokens` from the response so the runner can
  diagnose empty visible outputs on reasoning models.
- The runner writes a bounded `empty_sample_diagnostics` list into every
  JSONL row.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------- shared OpenAI-capturing fake ----------


def _install_capturing_openai(monkeypatch, captured: dict, *, choices: list,
                              usage: object | None = None):
    def fake_create(**kw):
        captured.update(kw)
        return SimpleNamespace(choices=choices, usage=usage)

    class _FakeOpenAI:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    from policy import vllm_policy
    monkeypatch.setattr(vllm_policy, "_get_key", lambda s: "sk-fake")


# ---------- OpenAIPolicy populates per-Sample diagnostics ----------


def test_sample_carries_finish_reason_and_token_diagnostics(monkeypatch):
    """Reasoning-model failure mode: empty `text` + finish_reason='length'
    + nonzero `reasoning_tokens` from `usage.completion_tokens_details`."""
    captured: dict = {}
    choices = [
        SimpleNamespace(
            message=SimpleNamespace(content=""),    # empty visible output
            logprobs=None,
            finish_reason="length",
        )
    ]
    usage = SimpleNamespace(
        prompt_tokens=120,
        completion_tokens=1024,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=1024),
    )
    _install_capturing_openai(monkeypatch, captured,
                              choices=choices, usage=usage)

    from policy.vllm_policy import OpenAIPolicy
    samples = OpenAIPolicy(model="gpt-5.5").sample_topk(
        "prompt", k=1, max_tokens=1024,
    )

    assert len(samples) == 1
    s = samples[0]
    assert s.text == ""
    assert s.finish_reason == "length"
    assert s.completion_tokens == 1024
    assert s.reasoning_tokens == 1024


def test_sample_diagnostics_default_to_none_without_usage(monkeypatch):
    """A response object lacking `usage` entirely must not crash; the
    diagnostics fields just stay None."""
    captured: dict = {}
    choices = [
        SimpleNamespace(
            message=SimpleNamespace(content="rfl"),
            logprobs=None,
            finish_reason="stop",
        )
    ]
    _install_capturing_openai(monkeypatch, captured,
                              choices=choices, usage=None)
    from policy.vllm_policy import OpenAIPolicy

    samples = OpenAIPolicy(model="gpt-4o-mini").sample_topk(
        "prompt", k=1, max_tokens=80,
    )
    s = samples[0]
    assert s.finish_reason == "stop"
    assert s.completion_tokens is None
    assert s.reasoning_tokens is None

