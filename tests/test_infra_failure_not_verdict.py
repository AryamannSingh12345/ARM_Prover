"""lake failing to start is not a proof verdict.

`has_errors` is `error:` in the output OR a non-zero exit. So a lake
subprocess that dies without printing anything produced
`ok=False, errors=""` in ~0s — and every caller reads that as "the proof
is wrong". The model is told its lemma failed, ARM burns a revision
round, and the trace records a mathematical failure that never happened.

MEASURED on mf20_amc12b_2021_p13_scaffold (2026-08-28): 27 of 41 verify
calls returned that way while the preceding cell had none. ARM made 23
prove_lemma calls and banked zero lemmas, including for
`Real.sin (2 * π) = 0` — which Mathlib proves by `Real.sin_two_pi`.

No Lean: the subprocess is faked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backend import compile_verify  # noqa: E402

SRC = "import Mathlib\n\ntheorem t : True := by trivial\n"


def _fake_run(monkeypatch, results):
    """Feed subprocess.run a queue of (returncode, stdout, stderr)."""
    calls = {"n": 0}
    seq = list(results)

    def fake(*a, **kw):
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        rc, out, err = seq[i]
        return SimpleNamespace(returncode=rc, stdout=out, stderr=err)

    monkeypatch.setattr(compile_verify.subprocess, "run", fake)
    monkeypatch.setattr(compile_verify.time, "sleep", lambda _s: None)
    return calls


def test_silent_nonzero_exit_is_marked_not_silent(monkeypatch, capsys):
    compile_verify._INFRA_STATE["failures"] = 0
    calls = _fake_run(monkeypatch, [(1, "", "")])
    res = compile_verify.compile_lean(SRC, timeout_s=5)
    assert not res.ok
    assert "INFRASTRUCTURE" in res.errors
    assert "NOT a proof verdict" in res.errors
    # One initial spawn plus every backoff step.
    assert calls["n"] == 1 + len(compile_verify._INFRA_RETRY_DELAYS)
    assert "lake exited" in capsys.readouterr().out
    assert compile_verify.infra_failure_count() == 1


def test_backoff_outlasts_a_longer_pressure_window(monkeypatch):
    """The 2s single retry was too short; a later attempt must still win.

    MEASURED: 82% of one cell's verify calls never ran because the
    pressure window outlived the retry.
    """
    compile_verify._INFRA_STATE["failures"] = 0
    calls = _fake_run(monkeypatch, [(1, "", ""), (1, "", ""), (0, "", "")])
    res = compile_verify.compile_lean(SRC, timeout_s=5, check_axioms=False)
    assert res.ok, "a spawn that recovers on the third try must count"
    assert compile_verify.infra_failure_count() == 0


def test_retry_recovers_a_transient_failure(monkeypatch):
    """First spawn dies silently, second works — that is a clean compile.

    check_axioms=False throughout the clean cases: the axiom gate runs a
    SECOND `#print axioms` compile and rightly rejects a faked empty
    report. That gate is exercised elsewhere; what is under test here is
    the spawn-failure path, which runs before it.
    """
    compile_verify._INFRA_STATE["failures"] = 0
    _fake_run(monkeypatch, [(1, "", ""), (0, "", "")])
    res = compile_verify.compile_lean(SRC, timeout_s=5, check_axioms=False)
    assert res.ok, "a recovered compile must not stay failed"
    assert res.errors.strip() == ""


def test_real_lean_errors_are_untouched(monkeypatch):
    """A genuine failure still reports its own diagnostics, no retry."""
    calls = _fake_run(monkeypatch, [(1, "foo.lean:3:0: error: unsolved goals", "")])
    res = compile_verify.compile_lean(SRC, timeout_s=5)
    assert not res.ok
    assert "unsolved goals" in res.errors
    assert "INFRASTRUCTURE" not in res.errors
    assert calls["n"] == 1, "output present — nothing transient to retry"


def test_clean_compile_untouched(monkeypatch):
    calls = _fake_run(monkeypatch, [(0, "", "")])
    res = compile_verify.compile_lean(SRC, timeout_s=5, check_axioms=False)
    assert res.ok and calls["n"] == 1


def test_zero_exit_with_no_output_is_not_infrastructure(monkeypatch):
    """Success is silent on this toolchain; only non-zero + silence is the bug."""
    calls = _fake_run(monkeypatch, [(0, "", "")])
    res = compile_verify.compile_lean(SRC, timeout_s=5, check_axioms=False)
    assert res.ok
    assert "INFRASTRUCTURE" not in (res.errors or "")
    assert calls["n"] == 1


def test_sigterm_is_not_counted_as_a_spawn_failure(monkeypatch, capsys):
    """A compile WE killed is not a compile that failed to start.

    Segment recycling SIGTERMs the in-flight lake (143 / -15). That
    arrives non-zero with no output — identical in shape to a spawn
    failure — and without this it is retried three times in a dying
    process and charged to the spawn-failure budget, which is the one
    number distinguishing a poisoned run from a real one.
    """
    compile_verify._INFRA_STATE["failures"] = 0
    calls = _fake_run(monkeypatch, [(143, "", "")])
    res = compile_verify.compile_lean(SRC, timeout_s=5, check_axioms=False)
    assert not res.ok
    assert "KILLED" in res.errors
    assert "INFRASTRUCTURE" not in res.errors
    assert calls["n"] == 1, "a killed compile must not be retried"
    assert compile_verify.infra_failure_count() == 0
    assert "terminated by signal" in capsys.readouterr().out
