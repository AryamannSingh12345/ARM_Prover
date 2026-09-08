"""Tasks 4 & 5: score-semantics contract and prompt premise splice.

These are pure tests — no API calls, no Lean. They verify:

- Anthropic rank fallback returns monotonically increasing costs.
- OpenAI policy assigns -mean(logprob) when content tokens carry logprobs.
- OpenAI policy falls back to rank when logprobs missing (no NameError on `i`).
- step_tactic_prompt splices retrieved premises when given, and omits the
  block otherwise.
- The premise block is capped (token budget guard).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------- Task 4: policy score semantics ----------


def test_anthropic_rank_fallback_is_monotonically_increasing(monkeypatch):
    """AnthropicPolicy makes k separate calls and assigns rank-based cost.
    The cost MUST be monotonically increasing in i (so sample 0 is cheapest)
    AND must be tagged source='rank-fallback'."""
    import anthropic as anthropic_mod  # noqa: F401 — used by isinstance later

    from policy import vllm_policy

    # Build a fake Anthropic client. The adapter streams (messages.stream
    # + get_final_message) — long max_tokens / adaptive-thinking calls
    # would hit HTTP timeouts on the non-streaming endpoint.
    class _FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="rfl")],
                stop_reason="end_turn",
            )

    class _FakeMessages:
        def stream(self, **kw):
            return _FakeStream()

    class _FakeAnthropic:
        def __init__(self, api_key, **_kw):
            # **_kw: the adapter passes `timeout` and
            # `max_retries` (a silent SDK retry of a reasoning
            # call doubles the bill invisibly). A fake that
            # rejects them fails on the constructor, not the
            # behaviour under test.
            self.messages = _FakeMessages()

    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropic)
    monkeypatch.setattr(vllm_policy, "_get_key", lambda s: "fake-key")
    # Isolate the spend-cap side channel: _check_spend_cap reads the live
    # results/spend_cap_usd.txt + usage_log.jsonl, so a real campaign that
    # crosses the cap would spuriously fail this ranking test. This test is
    # about rank-fallback score semantics, not the cap.
    monkeypatch.setattr(
        vllm_policy.AnthropicPolicy, "_check_spend_cap", lambda self: None)

    pol = vllm_policy.AnthropicPolicy(model="claude-haiku-test")
    samples = pol.sample_topk("prompt", k=4)

    assert len(samples) == 4
    # All sources must be rank-fallback.
    assert {s.source for s in samples} == {"rank-fallback"}
    # Scores must be strictly increasing in index — the contract is
    # "lower=better, sample 0 cheapest".
    scores = [s.score for s in samples]
    assert scores == sorted(scores)
    assert all(scores[i] < scores[i + 1] for i in range(len(scores) - 1))
    # And bounded in (0, 1].
    assert scores[0] > 0 and scores[-1] <= 1.0


def test_openai_logprob_path_returns_neg_mean_logprob(monkeypatch):
    """When choice.logprobs.content is populated, score must be
    -mean(token_logprob), tagged 'logprob'."""
    from openai import OpenAI  # noqa: F401

    from policy import vllm_policy

    # Fake completion with two tokens, logprobs -1.0 and -3.0 → mean -2.0
    # → score = 2.0.
    fake_choice = SimpleNamespace(
        message=SimpleNamespace(content="omega"),
        logprobs=SimpleNamespace(
            content=[
                SimpleNamespace(logprob=-1.0),
                SimpleNamespace(logprob=-3.0),
            ]
        ),
        finish_reason="stop",
    )
    fake_resp = SimpleNamespace(choices=[fake_choice])

    class _FakeOpenAI:
        def __init__(self, api_key, **_kw):
            # **_kw: the adapter passes `timeout` and
            # `max_retries` (a silent SDK retry of a reasoning
            # call doubles the bill invisibly). A fake that
            # rejects them fails on the constructor, not the
            # behaviour under test.
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: fake_resp)
            )

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    monkeypatch.setattr(vllm_policy, "_get_key", lambda s: "sk-fake")

    pol = vllm_policy.OpenAIPolicy(model="gpt-4o-mini")
    samples = pol.sample_topk("prompt", k=1)

    assert len(samples) == 1
    assert samples[0].source == "logprob"
    assert samples[0].score == 2.0  # -mean(-1.0, -3.0) = 2.0
    assert samples[0].text == "omega"


def test_openai_rank_fallback_when_no_logprobs(monkeypatch):
    """Critical regression: when logprobs is None (or content empty), the old
    code referenced an undefined variable `i` and would have raised NameError.
    After round 14 the fallback is deterministic `0.1 * sample_index` tagged
    `rank_fallback` (underscore) — k-independent and visually distinct from
    the historical Anthropic `rank-fallback` (dash) tag."""
    from openai import OpenAI  # noqa: F401

    from policy import vllm_policy

    fake_choices = [
        SimpleNamespace(
            message=SimpleNamespace(content=f"tac{i}"),
            logprobs=None,
            finish_reason="stop",
        )
        for i in range(3)
    ]
    fake_resp = SimpleNamespace(choices=fake_choices)

    class _FakeOpenAI:
        def __init__(self, api_key, **_kw):
            # **_kw: the adapter passes `timeout` and
            # `max_retries` (a silent SDK retry of a reasoning
            # call doubles the bill invisibly). A fake that
            # rejects them fails on the constructor, not the
            # behaviour under test.
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: fake_resp)
            )

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    monkeypatch.setattr(vllm_policy, "_get_key", lambda s: "sk-fake")

    pol = vllm_policy.OpenAIPolicy(model="gpt-4o-mini")
    samples = pol.sample_topk("prompt", k=3)  # k matches len(fake_choices)

    assert {s.source for s in samples} == {"rank_fallback"}
    scores = [s.score for s in samples]
    # Deterministic 0.1 * i values.
    assert scores == [0.0, 0.1, 0.2]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
