"""Persistent REPL step backend — unit + integration tests.

The compile-file backend is unusable on this Windows host: ~14 s for
`lake env lean --version` alone and 20+ minutes for a single tactic on
mathd_numbertheory_100. This file pins
the REPL backend's behavioural contract — DONE / PROGRESS / FAIL /
timeout classification, lazy startup, and soundness fall-through to
verify_proof — all without touching real lake.

We mock at two layers depending on which contract we're pinning:

  - For the session itself (test_session_*): patch
    `LeanReplStepSession._send` to return a scripted JSON response,
    and patch `_spawn` to a no-op so no real subprocess starts.
  - For caller wiring: patch the search entry point or the
    session's `step_for` to a stub that captures the inputs.

Pure tests; no lake, no LLM, no network.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------- session-level helpers ----------


def _make_session(monkeypatch, *, scripted_responses):
    """Return a LeanReplStepSession whose subprocess plumbing is mocked.

    `scripted_responses` is a list of dict payloads (or callables that
    raise) that _send will hand out in order. The session thinks it has
    a live REPL because _dead is flipped to False and the proc handle
    is a sentinel; the actual stdin/stdout never get touched.
    """
    from backend.repl_step import LeanReplStepSession

    monkeypatch.setattr(
        LeanReplStepSession, "_spawn",
        lambda self: setattr(self, "_proc", SimpleNamespace(poll=lambda: None))
                     or setattr(self, "_dead", False),
    )

    responses = list(scripted_responses)
    sent_commands: list[dict] = []

    def fake_send(self, cmd, *, timeout_s=None):
        sent_commands.append(dict(cmd))
        if not responses:
            raise AssertionError(
                f"no scripted response for command {cmd!r}"
            )
        nxt = responses.pop(0)
        if callable(nxt):
            return nxt()
        return nxt

    monkeypatch.setattr(LeanReplStepSession, "_send", fake_send)

    session = LeanReplStepSession()
    return session, sent_commands


# ---------- startup contract ----------


def test_startup_imports_mathlib_and_caches_env(monkeypatch):
    """startup() must send the import command exactly once and remember
    the returned env id for subsequent start_problem calls."""
    session, sent = _make_session(monkeypatch, scripted_responses=[
        {"env": 0, "messages": []},  # response to `import Mathlib`
    ])
    elapsed = session.startup()
    assert elapsed >= 0.0
    assert session.startup_s == elapsed
    assert session._env_id == 0
    assert sent == [{"cmd": "import Mathlib"}]


def test_startup_raises_on_import_error(monkeypatch):
    """An error message in the import response is a hard failure — the
    session is unusable, and we surface the error rather than silently
    leaving an env-less session."""
    from backend.repl_step import ReplStartupError

    session, _ = _make_session(monkeypatch, scripted_responses=[
        {"env": None, "messages": [
            {"severity": "error", "data": "unknown package 'Mathlib'"},
        ]},
    ])
    try:
        session.startup()
    except ReplStartupError as e:
        assert "unknown package" in str(e)
    else:
        raise AssertionError("ReplStartupError must be raised on import error")


# ---------- start_problem contract ----------


def test_start_problem_seeds_initial_state_cache(monkeypatch):
    """start_problem must send `<header> := by sorry` and seed the
    prefix-cache at the empty tuple with the proofState that came back."""
    session, sent = _make_session(monkeypatch, scripted_responses=[
        {"env": 0, "messages": []},  # import
        {  # start_problem response
            "env": 1,
            "sorries": [{
                "proofState": 7,
                "goal": "n : Nat\n⊢ n = 70",
            }],
            "messages": [{"severity": "warning",
                          "data": "declaration uses 'sorry'"}],
        },
    ])
    session.startup()
    session.start_problem("theorem t (n : Nat) : n = 70")
    assert session._state_cache[()] == 7
    assert session._current_theorem_header == "theorem t (n : Nat) : n = 70"
    # The second command must have been the `<header> := by sorry`
    # against the cached env id.
    assert sent[1] == {
        "cmd": "theorem t (n : Nat) : n = 70 := by sorry",
        "env": 0,
    }


# ---------- step classifier contract ----------


def test_step_classifies_progress(monkeypatch):
    """The bug-report case: a tactic that yields unsolved goals must
    classify as PROGRESS and surface the new goal text. This is the
    contract the persistent backend has to preserve when replacing the
    minutes-per-call compile-file backend."""
    from backend.compile_step import StepOutcome

    session, _ = _make_session(monkeypatch, scripted_responses=[
        {"env": 0, "messages": []},
        {"env": 1, "sorries": [{"proofState": 7}],
         "messages": [{"severity": "warning",
                       "data": "declaration uses 'sorry'"}]},
        # Tactic response: new proofState + non-empty goal list.
        {"proofState": 8, "goals": [
            "n : Nat\nhprod : n.gcd 40 * n.lcm 40 = n * 40\n⊢ n = 70",
        ]},
    ])
    session.startup()
    session.start_problem("theorem t (n : Nat) : n = 70")
    res = session.step(
        theorem_header="theorem t (n : Nat) : n = 70",
        prefix_tactics=(),
        new_tactic="have hprod := Nat.gcd_mul_lcm n 40",
    )
    assert res.outcome is StepOutcome.PROGRESS
    assert "hprod : n.gcd 40 * n.lcm 40 = n * 40" in res.new_goal_text
    # Cache must remember the successor state.
    assert session._state_cache[("have hprod := Nat.gcd_mul_lcm n 40",)] == 8


def test_step_classifies_done(monkeypatch):
    """An empty `goals` list means the proof closed → DONE. The session
    must NOT mark this as verified — the caller has to re-verify via
    compile_lean (the caller does that already)."""
    from backend.compile_step import StepOutcome

    session, _ = _make_session(monkeypatch, scripted_responses=[
        {"env": 0, "messages": []},
        {"env": 1, "sorries": [{"proofState": 7}],
         "messages": [{"severity": "warning", "data": "uses sorry"}]},
        {"proofState": 8, "goals": []},
    ])
    session.startup()
    session.start_problem("theorem t : True")
    res = session.step("theorem t : True", (), "trivial")
    assert res.outcome is StepOutcome.DONE
    assert res.new_goal_text == ""


def test_step_classifies_fail_on_error_message(monkeypatch):
    """Any error-severity message in the response makes the tactic FAIL,
    even if a proofState is also present (defensive against future REPL
    versions returning both)."""
    from backend.compile_step import StepOutcome

    session, _ = _make_session(monkeypatch, scripted_responses=[
        {"env": 0, "messages": []},
        {"env": 1, "sorries": [{"proofState": 7}],
         "messages": [{"severity": "warning", "data": "uses sorry"}]},
        {"messages": [
            {"severity": "error", "data": "unknown identifier 'foobar'"},
        ]},
    ])
    session.startup()
    session.start_problem("theorem t : True")
    res = session.step("theorem t : True", (), "exact foobar")
    assert res.outcome is StepOutcome.FAIL
    assert "unknown identifier" in res.error


def test_step_classifies_timeout(monkeypatch):
    """When _send raises TimeoutExpired, step() must FAIL with a
    `timeout after Ns` error and a populated elapsed_s for diagnostics."""
    from backend.compile_step import StepOutcome
    from backend.repl_step import LeanReplStepSession

    monkeypatch.setattr(
        LeanReplStepSession, "_spawn",
        lambda self: setattr(self, "_proc", SimpleNamespace(poll=lambda: None))
                     or setattr(self, "_dead", False),
    )

    responses_q = [
        {"env": 0, "messages": []},
        {"env": 1, "sorries": [{"proofState": 7}],
         "messages": [{"severity": "warning", "data": "uses sorry"}]},
    ]

    def fake_send(self, cmd, *, timeout_s=None):
        if responses_q:
            return responses_q.pop(0)
        raise subprocess.TimeoutExpired(cmd="repl-tactic", timeout=timeout_s)

    monkeypatch.setattr(LeanReplStepSession, "_send", fake_send)
    session = LeanReplStepSession()
    session.startup()
    session.start_problem("theorem t : True")
    res = session.step("theorem t : True", (), "omega", timeout_s=2)
    assert res.outcome is StepOutcome.FAIL
    assert "timeout after" in res.error
    assert res.elapsed_s >= 0.0


def test_step_uses_cached_state_for_backtracking(monkeypatch):
    """Best-first pops nodes out of order. Each pop must dispatch the
    tactic against the cached proofState for that node's prefix — no
    re-application required."""
    session, sent = _make_session(monkeypatch, scripted_responses=[
        {"env": 0, "messages": []},
        {"env": 1, "sorries": [{"proofState": 7}],
         "messages": [{"severity": "warning", "data": "uses sorry"}]},
        # First tactic.
        {"proofState": 8, "goals": ["⊢ g1"]},
        # Second tactic from the SAME root state (backtrack scenario).
        {"proofState": 9, "goals": ["⊢ g2"]},
    ])
    session.startup()
    session.start_problem("theorem t : True")
    session.step("theorem t : True", (), "intro h")
    session.step("theorem t : True", (), "intros")
    # Both tactic commands used proofState=7 (the cached root state).
    assert sent[2] == {"tactic": "intro h", "proofState": 7}
    assert sent[3] == {"tactic": "intros", "proofState": 7}


def test_step_records_elapsed_times_summary(monkeypatch):
    """step_times_summary feeds the JSONL diagnostic — must populate
    after any real step (success or fail)."""
    session, _ = _make_session(monkeypatch, scripted_responses=[
        {"env": 0, "messages": []},
        {"env": 1, "sorries": [{"proofState": 7}],
         "messages": [{"severity": "warning", "data": "uses sorry"}]},
        {"proofState": 8, "goals": ["⊢ g"]},
        {"proofState": 9, "goals": []},
    ])
    session.startup()
    session.start_problem("theorem t : True")
    session.step("theorem t : True", (), "intro")
    session.step("theorem t : True", ("intro",), "trivial")
    summary = session.step_times_summary()
    assert summary["count"] == 2
    assert "min" in summary and "mean" in summary and "max" in summary


# ---------- step_for lazy lifecycle ----------


def test_step_for_lazy_starts_problem(monkeypatch):
    """A caller wires step_for in as the step_fn. It must
    auto-start the session (spawn + import + open theorem) on first
    invocation."""
    from backend.compile_step import StepOutcome

    session, sent = _make_session(monkeypatch, scripted_responses=[
        {"env": 0, "messages": []},
        {"env": 1, "sorries": [{"proofState": 7}],
         "messages": [{"severity": "warning", "data": "uses sorry"}]},
        {"proofState": 8, "goals": ["⊢ g"]},
    ])
    # No explicit startup / start_problem — step_for should do both.
    res = session.step_for(
        "theorem t : True", (), "intro h",
    )
    assert res.outcome is StepOutcome.PROGRESS
    # Confirm the round-trip pattern: import → open theorem → tactic.
    assert sent[0] == {"cmd": "import Mathlib"}
    assert sent[1]["cmd"] == "theorem t : True := by sorry"
    assert sent[2] == {"tactic": "intro h", "proofState": 7}


def test_step_for_does_not_restart_within_same_theorem(monkeypatch):
    """Within a single theorem, repeat step_for calls must dispatch
    directly — no extra start_problem round-trip per tactic."""
    session, sent = _make_session(monkeypatch, scripted_responses=[
        {"env": 0, "messages": []},
        {"env": 1, "sorries": [{"proofState": 7}],
         "messages": [{"severity": "warning", "data": "uses sorry"}]},
        {"proofState": 8, "goals": ["⊢ g"]},
        {"proofState": 9, "goals": ["⊢ g'"]},
    ])
    session.step_for("theorem t : True", (), "intro h")
    session.step_for("theorem t : True", ("intro h",), "intro k")
    # Only ONE start_problem command was issued (`:= by sorry`).
    n_start = sum(1 for c in sent if "cmd" in c and c["cmd"].endswith("by sorry"))
    assert n_start == 1

