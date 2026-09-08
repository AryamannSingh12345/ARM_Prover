"""Compile-verifier smoke test without Mathlib — runs while lake update finishes."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from backend.compile_verify import compile_lean

#: Spawns a real `lake env lean` compile. It is fast on an IDLE machine
#: (which is why a 25s classification scan let it through) but its 60s
#: budget is blown as soon as anything else is compiling — it failed on
#: `timeout after 60s` in a suite run that was otherwise green. Spawning
#: lake is the criterion for `live`, not the stopwatch.
pytestmark = pytest.mark.live


def test_trivial_no_mathlib():
    src = "example : 2 + 2 = 4 := rfl\n"
    res = compile_lean(src, timeout_s=60)
    assert res.ok, res.errors


def test_detects_sorry():
    src = "example : True := by sorry\n"
    res = compile_lean(src, timeout_s=60)
    assert not res.ok
    assert res.used_sorry


def test_detects_error():
    src = "example : 1 = 2 := rfl\n"
    res = compile_lean(src, timeout_s=60)
    assert not res.ok


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
