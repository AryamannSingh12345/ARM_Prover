"""Round-15 patches:

- `_openai_supports_custom_temperature(model)` and
  `_openai_supports_logprobs(model)` return False for gpt-5 / o-series
  reasoning models.
- `OpenAIPolicy.sample_topk` omits `temperature` and `logprobs` for those
  models; sends them for older / non-reasoning models.
- When the response has no logprobs, score = 0.1 * sample_index and
  source = "rank_fallback".

Pure tests; no network, no Lean.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------- capability-helper unit tests ----------


def test_temperature_unsupported_on_reasoning_prefixes():
    from policy.vllm_policy import _openai_supports_custom_temperature
    for m in ("gpt-5", "gpt-5.5", "gpt-5-thinking", "o1", "o1-mini",
              "o3", "o3-mini", "o4", "o4-preview"):
        assert _openai_supports_custom_temperature(m) is False, m


def test_temperature_supported_on_legacy_and_unknown_models():
    from policy.vllm_policy import _openai_supports_custom_temperature
    for m in ("gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
              "custom-model-name", ""):
        assert _openai_supports_custom_temperature(m) is True, m


def test_logprobs_unsupported_on_reasoning_prefixes():
    from policy.vllm_policy import _openai_supports_logprobs
    for m in ("gpt-5", "gpt-5.5", "gpt-5-thinking", "o1", "o1-mini",
              "o3", "o3-mini", "o4", "o4-preview"):
        assert _openai_supports_logprobs(m) is False, m


def test_logprobs_supported_on_legacy_models():
    from policy.vllm_policy import _openai_supports_logprobs
    for m in ("gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"):
        assert _openai_supports_logprobs(m) is True, m


def test_capability_lookup_is_case_insensitive():
    from policy.vllm_policy import (
        _openai_supports_custom_temperature,
        _openai_supports_logprobs,
    )
    assert _openai_supports_custom_temperature("GPT-5.5") is False
    assert _openai_supports_logprobs("O1-Mini") is False
    assert _openai_supports_custom_temperature("GPT-4o-Mini") is True


# ---------- sample_topk request-kwargs (capturing fake) ----------


def _install_capturing_openai(monkeypatch, captured: dict,
                              *, logprobs_in_response: bool = False):
    """Install a fake OpenAI client whose chat.completions.create records
    its kwargs into `captured` and returns a single-choice response.
    `logprobs_in_response` controls whether the stubbed response carries
    per-token logprobs (sets `choice.logprobs` accordingly)."""
    def fake_create(**kw):
        captured.update(kw)
        if logprobs_in_response:
            lp = SimpleNamespace(content=[SimpleNamespace(logprob=-1.0)])
        else:
            lp = None
        choice = SimpleNamespace(
            message=SimpleNamespace(content="rfl"),
            logprobs=lp,
            finish_reason="stop",
        )
        return SimpleNamespace(choices=[choice])

    class _FakeOpenAI:
        def __init__(self, api_key, **_kw):
            # **_kw: the adapter passes `timeout` and
            # `max_retries` (a silent SDK retry of a reasoning
            # call doubles the bill invisibly). A fake that
            # rejects them fails on the constructor, not the
            # behaviour under test.
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    from policy import vllm_policy
    monkeypatch.setattr(vllm_policy, "_get_key", lambda s: "sk-fake")


def test_gpt5_request_uses_max_completion_tokens(monkeypatch):
    captured: dict = {}
    _install_capturing_openai(monkeypatch, captured)
    from policy.vllm_policy import OpenAIPolicy

    OpenAIPolicy(model="gpt-5.5").sample_topk("prompt", k=1, max_tokens=80)

    assert captured.get("max_completion_tokens") == 80
    assert "max_tokens" not in captured


def test_gpt5_request_omits_temperature(monkeypatch):
    captured: dict = {}
    _install_capturing_openai(monkeypatch, captured)
    from policy.vllm_policy import OpenAIPolicy

    OpenAIPolicy(model="gpt-5.5").sample_topk(
        "prompt", k=1, temperature=0.4, max_tokens=80,
    )

    assert "temperature" not in captured, (
        f"gpt-5.x rejects custom temperature; got captured={captured}"
    )


def test_gpt5_request_omits_logprobs(monkeypatch):
    captured: dict = {}
    _install_capturing_openai(monkeypatch, captured)
    from policy.vllm_policy import OpenAIPolicy

    OpenAIPolicy(model="gpt-5.5").sample_topk("prompt", k=1, max_tokens=80)

    assert "logprobs" not in captured, (
        f"gpt-5.x does not expose logprobs; got captured={captured}"
    )


def test_gpt4omini_request_still_includes_temperature_and_logprobs(monkeypatch):
    """Sanity: the conditional omissions must NOT regress the legacy path."""
    captured: dict = {}
    _install_capturing_openai(monkeypatch, captured,
                              logprobs_in_response=True)
    from policy.vllm_policy import OpenAIPolicy

    OpenAIPolicy(model="gpt-4o-mini").sample_topk(
        "prompt", k=1, temperature=0.3, max_tokens=64,
    )

    assert captured.get("temperature") == 0.3
    assert captured.get("logprobs") is True
    assert captured.get("max_completion_tokens") == 64


# ---------- score handling when logprobs are absent ----------


def test_gpt5_response_returns_rank_fallback_with_deterministic_costs(monkeypatch):
    """A reasoning-model response carries no per-token logprobs (`logprobs
    is None`); the OpenAIPolicy must produce Samples with source
    `"rank_fallback"` and deterministic costs `0.1 * i`."""
    def fake_create(**kw):
        return SimpleNamespace(choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=f"tac{i}"),
                logprobs=None,
                finish_reason="stop",
            )
            for i in range(4)
        ])

    class _FakeOpenAI:
        def __init__(self, api_key, **_kw):
            # **_kw: the adapter passes `timeout` and
            # `max_retries` (a silent SDK retry of a reasoning
            # call doubles the bill invisibly). A fake that
            # rejects them fails on the constructor, not the
            # behaviour under test.
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    from policy import vllm_policy
    monkeypatch.setattr(vllm_policy, "_get_key", lambda s: "sk-fake")

    samples = vllm_policy.OpenAIPolicy(model="gpt-5.5").sample_topk(
        "prompt", k=4, max_tokens=80,
    )

    assert len(samples) == 4
    # Source tag must be the underscore form (distinct from Anthropic's
    # historical dash form).
    assert {s.source for s in samples} == {"rank_fallback"}
    # Costs are deterministic 0.1 * i — k-independent. Use pytest.approx
    # because IEEE 754 makes 0.1 * 3 == 0.30000000000000004, not 0.3.
    import pytest
    scores = [s.score for s in samples]
    assert scores == pytest.approx([0.0, 0.1, 0.2, 0.3])
    # And lower is still better (search contract): index 0 cheapest.
    assert scores[0] == min(scores)
    # Monotonic in sample index — preserves policy order.
    assert all(scores[i] < scores[i + 1] for i in range(len(scores) - 1))


def test_logprobs_present_in_response_still_yields_logprob_source(monkeypatch):
    """Belt-and-braces: even when the model name says reasoning, if the
    response happens to carry .logprobs.content (e.g. OpenAI flips a model
    later), the runtime path uses the real signal."""
    def fake_create(**kw):
        return SimpleNamespace(choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="omega"),
                logprobs=SimpleNamespace(
                    content=[
                        SimpleNamespace(logprob=-1.0),
                        SimpleNamespace(logprob=-3.0),
                    ]
                ),
                finish_reason="stop",
            )
        ])

    class _FakeOpenAI:
        def __init__(self, api_key, **_kw):
            # **_kw: the adapter passes `timeout` and
            # `max_retries` (a silent SDK retry of a reasoning
            # call doubles the bill invisibly). A fake that
            # rejects them fails on the constructor, not the
            # behaviour under test.
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    from policy import vllm_policy
    monkeypatch.setattr(vllm_policy, "_get_key", lambda s: "sk-fake")

    samples = vllm_policy.OpenAIPolicy(model="gpt-5.5").sample_topk(
        "prompt", k=1, max_tokens=80,
    )
    assert samples[0].source == "logprob"
    assert samples[0].score == 2.0  # -mean(-1, -3)


def test_gpt5_response_missing_logprobs_attribute_does_not_crash(monkeypatch):
    """Defensive: a stripped-down response object whose choice lacks the
    `logprobs` attribute entirely must still fall back to rank_fallback
    rather than raise."""
    def fake_create(**kw):
        # Note: NO `logprobs` attribute at all on the choice — not even
        # set to None. Tests the `getattr(... , None)` guard.
        choice = SimpleNamespace(
            message=SimpleNamespace(content="rfl"),
            finish_reason="stop",
        )
        return SimpleNamespace(choices=[choice])

    class _FakeOpenAI:
        def __init__(self, api_key, **_kw):
            # **_kw: the adapter passes `timeout` and
            # `max_retries` (a silent SDK retry of a reasoning
            # call doubles the bill invisibly). A fake that
            # rejects them fails on the constructor, not the
            # behaviour under test.
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    from policy import vllm_policy
    monkeypatch.setattr(vllm_policy, "_get_key", lambda s: "sk-fake")

    samples = vllm_policy.OpenAIPolicy(model="gpt-5.5").sample_topk(
        "prompt", k=1, max_tokens=80,
    )
    assert samples[0].source == "rank_fallback"
    assert samples[0].score == 0.0
