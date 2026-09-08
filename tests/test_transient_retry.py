"""Transient-failure retry in the policy adapter (2026-08-21).

MEASURED motivation: three long DAG runs on putnam_1963_a2 died to a
single failed provider call, after hours of compile time and with
kernel-proved lemmas in hand, writing NO result row:

    p1963a2_dag_v2       t=12766s   InternalServerError: 500
    p1963a2_dag_v3gate   t=18662s   APITimeoutError: Request timed out

`run_dag` treats a policy exception as a terminal `stage=llm` failure, so
one 500 decides a five-hour experiment.

The properties pinned here are the ones that make the retry safe rather
than merely present:
  - deterministic failures (400/401/404) are NEVER retried, because they
    fail identically every time and retrying turns a clear error into a
    slow one;
  - the budget is bounded and the ORIGINAL exception is re-raised when it
    is spent, so an unreachable provider still fails with the provider's
    own message;
  - every retry prints, because the SDK's own silent retry was disabled
    deliberately (a hidden retry of a multi-minute reasoning call doubles
    the bill and appears nowhere).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policy import vllm_policy as vp  # noqa: E402


# ---------------------------------------------------------------- fakes

class APITimeoutError(Exception):
    """Name-matched to the real SDK class."""


class InternalServerError(Exception):
    status_code = 500


class RateLimitError(Exception):
    status_code = 429


class OverloadedError(Exception):
    """Anthropic 529."""


class BadRequestError(Exception):
    status_code = 400


class AuthenticationError(Exception):
    status_code = 401


class NotFoundError(Exception):
    status_code = 404


class WeirdServerError(Exception):
    status_code = 503


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Retry backoff must not slow the suite."""
    monkeypatch.setattr(vp.time, "sleep", lambda _s: None)


# ------------------------------------------------- transient classification

@pytest.mark.parametrize("exc", [
    APITimeoutError("Request timed out."),
    InternalServerError("Error code: 500 - server had an error"),
    RateLimitError("429"),
    OverloadedError("529"),
    WeirdServerError("503 unknown class, known status"),
])
def test_transient_failures_are_recognised(exc):
    assert vp._is_transient(exc)


@pytest.mark.parametrize("exc", [
    BadRequestError("Unsupported parameter: 'max_tokens'"),
    AuthenticationError("invalid api key"),
    NotFoundError("model not found"),
    ValueError("a plain bug in our own code"),
    KeyError("messages"),
])
def test_deterministic_failures_are_not_retried(exc):
    assert not vp._is_transient(exc)


def test_the_two_real_failures_are_transient():
    """The exact exceptions that killed p1963a2_dag_v2 and _v3gate."""
    assert vp._is_transient(APITimeoutError("Request timed out."))
    assert vp._is_transient(
        InternalServerError("Error code: 500 - {'error': {'message': "
                            "'The server had an error while processing "
                            "your request. Sorry about that!'}}"))


# ------------------------------------------------------------ retry behaviour

def test_succeeds_without_retry_when_the_call_works():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert vp._with_transient_retry(fn, what="t") == "ok"
    assert len(calls) == 1, "a working call must not be repeated"


def test_recovers_after_transient_failures():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise InternalServerError("500")
        return "recovered"

    assert vp._with_transient_retry(fn, what="t") == "recovered"
    assert len(calls) == 3


def test_deterministic_failure_raises_immediately():
    calls = []

    def fn():
        calls.append(1)
        raise BadRequestError("Unsupported parameter")

    with pytest.raises(BadRequestError):
        vp._with_transient_retry(fn, what="t")
    assert len(calls) == 1, "a 400 must not be retried even once"


def test_budget_is_bounded_and_original_exception_survives():
    calls = []

    def fn():
        calls.append(1)
        raise APITimeoutError("Request timed out.")

    with pytest.raises(APITimeoutError) as ei:
        vp._with_transient_retry(fn, what="t")
    assert len(calls) == vp._RETRY_ATTEMPTS
    # The caller must see the PROVIDER's message, not a wrapper's.
    assert "timed out" in str(ei.value)


def test_every_retry_is_announced(capsys):
    """Silent retries were disabled on purpose; these must be visible."""
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise InternalServerError("500")
        return "ok"

    vp._with_transient_retry(fn, what="openai gpt-5.6-sol (k=3)")
    out = capsys.readouterr().out
    assert out.count("[policy] TRANSIENT") == 2
    assert "InternalServerError" in out
    assert "openai gpt-5.6-sol (k=3)" in out, "must name the failing call"
    assert "retrying in" in out


def test_backoff_grows_and_is_jittered(monkeypatch):
    delays = []
    monkeypatch.setattr(vp.time, "sleep", lambda s: delays.append(s))
    monkeypatch.setattr(vp.random, "random", lambda: 0.5)  # jitter factor 1.0

    def fn():
        raise InternalServerError("500")

    with pytest.raises(InternalServerError):
        vp._with_transient_retry(fn, what="t")
    assert len(delays) == vp._RETRY_ATTEMPTS - 1
    assert delays == sorted(delays), "backoff must be non-decreasing"
    assert all(d <= vp._RETRY_MAX_S for d in delays), "cap must hold"


def test_delay_never_exceeds_the_cap_at_high_attempt_counts(monkeypatch):
    monkeypatch.setattr(vp, "_RETRY_ATTEMPTS", 12)
    monkeypatch.setattr(vp, "_RETRY_BASE_S", 5.0)
    delays = []
    monkeypatch.setattr(vp.time, "sleep", lambda s: delays.append(s))
    monkeypatch.setattr(vp.random, "random", lambda: 1.0)  # worst-case jitter

    def fn():
        raise InternalServerError("500")

    with pytest.raises(InternalServerError):
        vp._with_transient_retry(fn, what="t")
    assert max(delays) <= vp._RETRY_MAX_S * 1.5 + 1e-9


# ------------------------------------------------------------ wiring

def test_all_provider_call_sites_are_wrapped():
    """Every billed API call must go through the retry helper.

    Asserted on source because the call sites live inside provider
    classes whose SDKs are not importable in the fast suite.
    """
    src = (Path(__file__).resolve().parents[1]
           / "src" / "policy" / "vllm_policy.py").read_text(encoding="utf-8")
    # No bare create() call may remain outside the helper.
    for line in src.split("\n"):
        s = line.strip()
        if s.startswith("resp = self.client.chat.completions.create"):
            raise AssertionError(f"unwrapped provider call: {s}")
        if s.startswith("resp = self._stream_final("):
            raise AssertionError(f"unwrapped provider call: {s}")
    assert src.count("_with_transient_retry(") >= 5  # 1 def + 4 call sites


def test_sdk_level_silent_retry_stays_disabled():
    """The SDK's own retry was disabled deliberately; keep it that way."""
    src = (Path(__file__).resolve().parents[1]
           / "src" / "policy" / "vllm_policy.py").read_text(encoding="utf-8")
    assert "max_retries=0" in src


def test_motivation_is_recorded_in_source():
    src = (Path(__file__).resolve().parents[1]
           / "src" / "policy" / "vllm_policy.py").read_text(encoding="utf-8")
    assert "p1963a2_dag_v3gate" in src
    assert "InternalServerError" in src
