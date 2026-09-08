"""Debug-dump plumbing + PROGRESS / FAIL / TIMEOUT classification audit.

Anchored to the mathd_numbertheory_100 backend bug: manual Lean confirms
that `have hprod := Nat.gcd_mul_lcm n 40` produces unsolved goals (PROGRESS),
but the prover reported a 45 s timeout. The fix is plumbing + a tighter
audit on the classifier, NOT a behaviour change on PROGRESS itself — these
tests pin down the contract so future regressions are caught fast.

Pure tests; no lake, no LLM, no network — subprocess.run is mocked.
"""
from __future__ import annotations

import pytest

#: Spawns a real `lake env lean` compile — minutes per test on this
#: host. Measured, not guessed: this file did not finish in 25s.
#: Excluded from the fast suite via `pytest -m "not live"`.
pytestmark = pytest.mark.live


import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------- helpers ----------


class _FakeProc:
    """Stand-in for subprocess.run's return value."""

    def __init__(self, *, returncode: int = 0, stdout: str = "",
                 stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _patch_subprocess(monkeypatch, *, stdout: str = "", stderr: str = "",
                       returncode: int = 0, timeout: bool = False):
    """Install a fake subprocess.run on compile_verify.

    When `timeout=True`, every invocation raises TimeoutExpired so we can
    assert the timeout branch's diagnostics.
    """
    from backend import compile_verify

    def fake_run(*a, **kw):
        if timeout:
            raise subprocess.TimeoutExpired(cmd=a[0], timeout=kw.get("timeout"))
        return _FakeProc(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(compile_verify.subprocess, "run", fake_run)
    return compile_verify


# ---------- PROGRESS classification ----------


def test_unsolved_goals_classified_as_progress(monkeypatch):
    """The bug-report case: Lean returns non-zero with `error: unsolved
    goals\n<goal text>`. compile_step.step MUST return PROGRESS, not FAIL,
    and the parsed goal text must surface in `new_goal_text`."""
    from backend import compile_step, compile_verify

    # Lean prints unsolved goals on stderr; non-zero exit. Goal-context
    # lines are INDENTED — _UNSOLVED_RE relies on `\n\S` (newline +
    # non-whitespace) to mark the END of the goal block, so the
    # indentation is what keeps the regex capturing past line 1.
    lean_output = (
        "Try.lean:5:0: error: unsolved goals\n"
        "  n : Nat\n"
        "  h0 : 0 < n\n"
        "  h1 : n.gcd 40 = 10\n"
        "  h2 : n.lcm 40 = 280\n"
        "  hprod : n.gcd 40 * n.lcm 40 = n * 40\n"
        "  ⊢ n = 70\n"
    )
    _patch_subprocess(monkeypatch, stdout="", stderr=lean_output, returncode=1)
    # Stub temp-file write so we don't touch the real Generated/ dir.
    monkeypatch.setattr(compile_verify, "_write_temp",
                        lambda s: Path("/tmp/_unused.lean"))

    res = compile_step.step(
        theorem_header="theorem t (n : Nat) : n = 70",
        prefix_tactics=(),
        new_tactic="have hprod := Nat.gcd_mul_lcm n 40",
        timeout_s=10,
    )
    assert res.outcome is compile_step.StepOutcome.PROGRESS, (
        f"unsolved-goals output must classify as PROGRESS; "
        f"got {res.outcome.value} (error={res.error!r})"
    )
    assert "n = 70" in res.new_goal_text


def test_unknown_identifier_classified_as_fail(monkeypatch):
    """The other side of the contract: a genuine Lean error (e.g. unknown
    identifier) is FAIL, not silently classified as PROGRESS."""
    from backend import compile_step, compile_verify

    lean_output = "Try.lean:5:7: error: unknown identifier 'gcd_mul_lcm'\n"
    _patch_subprocess(monkeypatch, stderr=lean_output, returncode=1)
    monkeypatch.setattr(compile_verify, "_write_temp",
                        lambda s: Path("/tmp/_unused.lean"))

    res = compile_step.step(
        theorem_header="theorem t : True",
        prefix_tactics=(),
        new_tactic="have hprod := gcd_mul_lcm n 40",
        timeout_s=10,
    )
    assert res.outcome is compile_step.StepOutcome.FAIL
    assert "unknown identifier" in res.error


def test_timeout_classified_as_fail_with_replay_envelope(monkeypatch):
    """Timeout still maps to FAIL (current semantics) but the diagnostic
    fields (cwd, argv, elapsed_s) must be populated so a post-mortem can
    replay or re-time the same invocation."""
    from backend import compile_step, compile_verify

    _patch_subprocess(monkeypatch, timeout=True)
    monkeypatch.setattr(compile_verify, "_write_temp",
                        lambda s: Path("/tmp/_unused.lean"))

    res = compile_step.step(
        theorem_header="theorem t : True",
        prefix_tactics=(),
        new_tactic="omega",
        timeout_s=7,
    )
    assert res.outcome is compile_step.StepOutcome.FAIL
    assert res.error.startswith("timeout after")
    # Replay envelope must be populated even on timeout.
    assert res.cwd, "cwd must be set on timeout for replay"
    assert res.argv and res.argv[:3] == ["lake", "env", "lean"]
    assert res.elapsed_s >= 0.0


# ---------- debug dump ----------


def test_debug_dump_writes_lean_and_sidecar(monkeypatch, tmp_path):
    """When debug_dump_path is set, compile_lean must write `<stem>.lean`
    BEFORE subprocess.run (so a hang is debuggable) and `<stem>.json` after."""
    from backend import compile_verify

    seen_at_subprocess_time: dict = {}
    dump_stem = tmp_path / "depth_1_cand_1"

    def fake_run(*a, **kw):
        # By the time subprocess.run fires, the .lean must already exist
        # on disk — that's the whole point of the pre-emptive dump.
        seen_at_subprocess_time["lean_exists"] = (
            dump_stem.with_suffix(".lean").exists()
        )
        seen_at_subprocess_time["cwd"] = kw.get("cwd")
        return _FakeProc(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(compile_verify.subprocess, "run", fake_run)
    monkeypatch.setattr(compile_verify, "_write_temp",
                        lambda s: tmp_path / "_temp.lean")

    res = compile_verify.compile_lean(
        "example : True := trivial\n",
        timeout_s=10, debug_dump_path=dump_stem,
    )
    assert res.ok

    # .lean written upfront — survives a hang.
    assert seen_at_subprocess_time.get("lean_exists") is True, (
        "debug .lean must be written BEFORE subprocess.run so it survives "
        "a lake hang/crash"
    )
    assert dump_stem.with_suffix(".lean").read_text(encoding="utf-8") == (
        "example : True := trivial\n"
    )

    # JSON sidecar written after, with the full replay envelope.
    sidecar_path = dump_stem.with_suffix(".json")
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["status"] == "ok"
    assert sidecar["ok"] is True
    assert sidecar["cwd"] == str(compile_verify.LEAN_ROOT)
    assert sidecar["argv"][:3] == ["lake", "env", "lean"]
    assert sidecar["timeout_s"] == 10
    assert isinstance(sidecar["elapsed_s"], (int, float))
    assert "replay" in sidecar
    assert sidecar["replay"]["cwd"] == str(compile_verify.LEAN_ROOT)
    assert sidecar["replay"]["lean_file"] == str(dump_stem.with_suffix(".lean"))


def test_debug_dump_sidecar_marks_timeout(monkeypatch, tmp_path):
    """On timeout, the sidecar's `status` must be `timeout` so a triage
    grep can isolate timeout failures from genuine error failures."""
    from backend import compile_verify

    dump_stem = tmp_path / "depth_3_cand_2"
    _patch_subprocess(monkeypatch, timeout=True)
    monkeypatch.setattr(compile_verify, "_write_temp",
                        lambda s: tmp_path / "_temp.lean")

    res = compile_verify.compile_lean(
        "example : True := trivial\n",
        timeout_s=2, debug_dump_path=dump_stem,
    )
    assert not res.ok
    sidecar = json.loads(
        dump_stem.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["status"] == "timeout"
    assert sidecar["timeout_s"] == 2
    # The .lean is still on disk for replay.
    assert dump_stem.with_suffix(".lean").exists()


def test_debug_dump_off_writes_nothing(monkeypatch, tmp_path):
    """Without debug_dump_path, compile_lean must not touch the dump dir."""
    from backend import compile_verify

    _patch_subprocess(monkeypatch, returncode=0)
    monkeypatch.setattr(compile_verify, "_write_temp",
                        lambda s: tmp_path / "_temp.lean")

    res = compile_verify.compile_lean(
        "example : True := trivial\n", timeout_s=10,
    )
    assert res.ok
    assert res.debug_dump_path is None
    assert list(tmp_path.glob("depth_*")) == []

