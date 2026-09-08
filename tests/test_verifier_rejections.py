"""Reliability gates — Task 1/2/3 from the patch round.

These tests exercise the hard-reject paths *without* invoking lake or the
Lean REPL. The verifier module is exercised via a subprocess.run mock; the
step module via a compile_lean mock; the REPL judge via its pure helper.

Why this matters: the Lean verifier is the only oracle, AND it must reject
sorry / native_decide / unsafe shortcuts. A bug in the wrapper that flips
even one such proof to OK is a "silent failure" (proposal §II) and would
overstate solve rates in any ablation log.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------- helpers ----------


def _fake_proc(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# ---------- Task 1: compile_verify rejects native_decide ----------


def test_compile_verify_rejects_native_decide_even_when_lean_accepts(tmp_path, monkeypatch):
    """Lean may compile a `native_decide` proof cleanly. Our wrapper must
    still mark it ok=False — native_decide is a verifier escape hatch and
    is explicitly excluded from "proof counts"."""
    from backend import compile_verify

    src = "example : 2 + 2 = 4 := by native_decide\n"

    def fake_run(*args, **kw):
        # Pretend lake env lean returned cleanly with no error output.
        return _fake_proc(returncode=0, stdout="", stderr="")

    # Avoid touching the real Generated dir.
    monkeypatch.setattr(compile_verify, "_write_temp", lambda s: tmp_path / "x.lean")
    # Also pre-create the file so the finally-unlink doesn't error.
    (tmp_path / "x.lean").write_text(src, encoding="utf-8")
    monkeypatch.setattr(compile_verify.subprocess, "run", fake_run)

    res = compile_verify.compile_lean(src, timeout_s=10)
    assert res.ok is False, "native_decide must not be accepted as ok"
    assert res.used_native_decide is True


def test_compile_verify_rejects_explicit_sorry(tmp_path, monkeypatch):
    from backend import compile_verify

    src = "example : True := by sorry\n"
    monkeypatch.setattr(compile_verify, "_write_temp", lambda s: tmp_path / "y.lean")
    (tmp_path / "y.lean").write_text(src, encoding="utf-8")
    monkeypatch.setattr(
        compile_verify.subprocess, "run",
        lambda *a, **kw: _fake_proc(0, "", "declaration uses 'sorry'\n"),
    )
    res = compile_verify.compile_lean(src, timeout_s=10)
    assert res.ok is False
    assert res.used_sorry is True


def test_compile_verify_clean_proof_is_ok(tmp_path, monkeypatch):
    """Sanity check: when source has no rejection markers and lake returns
    cleanly, we DO accept."""
    from backend import compile_verify

    src = "example : 2 + 2 = 4 := rfl\n"
    monkeypatch.setattr(compile_verify, "_write_temp", lambda s: tmp_path / "z.lean")
    (tmp_path / "z.lean").write_text(src, encoding="utf-8")
    monkeypatch.setattr(
        compile_verify.subprocess, "run",
        lambda *a, **kw: _fake_proc(0, "", ""),
    )
    res = compile_verify.compile_lean(src, timeout_s=10)
    assert res.ok is True
    assert res.used_native_decide is False
    assert res.used_sorry is False


# ---------- Task 3: compile_step does not silently mark a native_decide reject as DONE ----------


def _fake_compile_lean(*, ok: bool, errors: str = "", used_sorry: bool = False,
                       used_native_decide: bool = False):
    from backend.compile_verify import CompileResult

    def _f(*a, **kw):
        return CompileResult(ok=ok, errors=errors, used_sorry=used_sorry,
                             used_native_decide=used_native_decide)
    return _f


def test_compile_step_does_not_mark_native_decide_as_done(monkeypatch):
    """The previous fallback in compile_step accepted DONE when there were
    no errors and no sorry — masking native_decide rejection. Regression
    test: a not-ok CompileResult with no error line AND no sorry but
    used_native_decide=True must classify as FAIL, never DONE."""
    from backend import compile_step

    fake = _fake_compile_lean(ok=False, errors="", used_sorry=False,
                               used_native_decide=True)
    monkeypatch.setattr(compile_step, "compile_lean", fake)

    res = compile_step.step("theorem t : True", (), "native_decide")
    assert res.outcome is compile_step.StepOutcome.FAIL
    assert "native_decide" in (res.error or "")


def test_compile_step_clean_done_path(monkeypatch):
    from backend import compile_step

    fake = _fake_compile_lean(ok=True)
    monkeypatch.setattr(compile_step, "compile_lean", fake)

    res = compile_step.step("theorem t : True", (), "trivial")
    assert res.outcome is compile_step.StepOutcome.DONE


def test_compile_step_progress_when_only_unsolved_goals(monkeypatch):
    from backend import compile_step

    fake = _fake_compile_lean(
        ok=False,
        errors="error: unsolved goals\n  P\n",
        used_sorry=False,
        used_native_decide=False,
    )
    monkeypatch.setattr(compile_step, "compile_lean", fake)

    res = compile_step.step("theorem t : True", (), "intro")
    assert res.outcome is compile_step.StepOutcome.PROGRESS
    assert "P" in res.new_goal_text


def test_compile_step_fail_on_unclassifiable_rejection(monkeypatch):
    """No error line, no sorry, no unsolved goals, but ok=False — the old
    code would have returned DONE. New code must return FAIL."""
    from backend import compile_step

    fake = _fake_compile_lean(ok=False, errors="", used_sorry=False,
                               used_native_decide=False)
    monkeypatch.setattr(compile_step, "compile_lean", fake)

    res = compile_step.step("theorem t : True", (), "weird_tactic")
    assert res.outcome is compile_step.StepOutcome.FAIL


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
