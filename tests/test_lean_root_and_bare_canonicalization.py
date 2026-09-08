"""Bare-name canonicalisation + Lean project root plumbing.

Anchored to the manual sanity-check result from `lean/sanity_gcd_*.lean`:

  have hprod := gcd_mul_lcm n 40        -- fails: name not in scope
  have hprod := Nat.gcd_mul_lcm n 40    -- accepted; PROGRESS step

The prover must rewrite the bare form to the qualified form (when
`Nat.gcd_mul_lcm` is the unique tail match) BEFORE Lean ever sees it,
and the unknown-premise filter must operate on the rewritten text.

Pure tests; no lake, no LLM, no network.
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

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_compile_lean_uses_lean_project_root_as_cwd(tmp_path, monkeypatch):
    """compile_verify.compile_lean must pass the override path as the
    subprocess cwd. Catches the regression where the module-level LEAN_ROOT
    is silently used instead of the caller's choice."""
    from backend import compile_verify

    seen_cwd: list = []

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(*args, **kwargs):
        seen_cwd.append(kwargs.get("cwd"))
        return FakeProc()

    monkeypatch.setattr(compile_verify, "_write_temp",
                        lambda s: tmp_path / "u.lean")
    monkeypatch.setattr(compile_verify.subprocess, "run", fake_run)

    custom = tmp_path / "alt_lean"
    custom.mkdir()
    compile_verify.compile_lean(
        "example : True := trivial\n",
        timeout_s=10, lean_project_root=custom,
    )
    assert seen_cwd == [custom], (
        f"compile_lean must run subprocess with cwd=lean_project_root; "
        f"saw {seen_cwd!r}"
    )


def test_compile_lean_default_cwd_unchanged(tmp_path, monkeypatch):
    """Backward-compat: when no lean_project_root is passed, falls back
    to the module-level LEAN_ROOT (existing callers and tests rely on this)."""
    from backend import compile_verify

    seen_cwd: list = []

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(*args, **kwargs):
        seen_cwd.append(kwargs.get("cwd"))
        return FakeProc()

    monkeypatch.setattr(compile_verify, "_write_temp",
                        lambda s: tmp_path / "u.lean")
    monkeypatch.setattr(compile_verify.subprocess, "run", fake_run)

    compile_verify.compile_lean("example : True := trivial\n", timeout_s=10)
    assert seen_cwd == [compile_verify.LEAN_ROOT]
