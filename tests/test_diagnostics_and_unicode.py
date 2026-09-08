"""Round-3 patches:

- compile_verify forces utf-8 + errors='replace' on subprocess.run.
- search-level soundness: step DONE + verify_proof rejected → solved=False,
  failure recorded.
- failed_tactic_examples populates on plain FAIL paths.

All tests are pure/mocked: no lake, no LLM, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------- Task 1 regression: subprocess.run kwargs ----------


def test_compile_lean_decodes_utf8_with_replace(tmp_path, monkeypatch):
    """The Windows-default cp1252 decoder crashes on Lean's Unicode goal
    symbols (⊢, ∀, ⟨ …). compile_lean must request utf-8 + errors='replace'
    so a single bad byte is replaced, not crash-the-search."""
    from backend import compile_verify

    captured: dict = {}

    def fake_run(cmd, **kw):
        captured.update(kw)
        # Simulate lake output containing a real Unicode goal symbol —
        # this is what would have crashed under cp1252 before the fix.
        return SimpleNamespace(returncode=0, stdout="⊢ True\n", stderr="")

    monkeypatch.setattr(compile_verify, "_write_temp", lambda s: tmp_path / "u.lean")
    (tmp_path / "u.lean").write_text("example : True := trivial\n", encoding="utf-8")
    monkeypatch.setattr(compile_verify.subprocess, "run", fake_run)

    # Must not raise UnicodeDecodeError or similar.
    res = compile_verify.compile_lean("example : True := trivial\n", timeout_s=10)

    assert captured.get("encoding") == "utf-8", \
        "compile_lean must request utf-8 decoding"
    assert captured.get("errors") == "replace", \
        "compile_lean must use errors='replace' so a bad byte cannot crash decoding"
    # And the function still functions (returns a CompileResult).
    assert res is not None


def test_compile_lean_does_not_crash_on_replacement_chars(tmp_path, monkeypatch):
    """Even if Lake's output contained un-decodable bytes that got coerced
    to �, compile_lean must classify the result (ok or not-ok) without
    raising."""
    from backend import compile_verify

    # Simulate: errors='replace' already kicked in upstream, leaving �.
    bad_output = "error: something �� happened\n"

    def fake_run(cmd, **kw):
        return SimpleNamespace(returncode=1, stdout="", stderr=bad_output)

    # NB the source must NOT contain `sorry`/`native_decide`: compile_lean
    # short-circuits those before spawning lake (they can never be `ok`),
    # so the fake subprocess — and this test's decode assertion — would
    # never be reached. The property under test is decode safety, not
    # sorry handling, so any compiling-shaped source does.
    src = "example : True := trivial\n"
    monkeypatch.setattr(compile_verify, "_write_temp", lambda s: tmp_path / "u2.lean")
    (tmp_path / "u2.lean").write_text(src, encoding="utf-8")
    monkeypatch.setattr(compile_verify.subprocess, "run", fake_run)

    res = compile_verify.compile_lean(src, timeout_s=10)
    assert res.ok is False
    assert "�" in res.errors  # replacement char preserved for diagnostics


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
