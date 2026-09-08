"""Round-4 patches:

- clean_tactic pre-Lake rejection of LLM placeholder/escape-hatch junk.
- Timeout failures land in failed_tactic_examples as `timeout after Ns`.

Pure tests; no lake, no LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_compile_step_timeout_path_preserves_message(monkeypatch):
    """Belt-and-braces: confirm the compile_step → compile_lean → step()
    boundary actually carries the exact `timeout after Ns` string from
    `compile_verify.compile_lean`'s TimeoutExpired branch."""
    from backend import compile_step
    from backend.compile_verify import CompileResult

    def fake_compile_lean(*a, **kw):
        # This is exactly what compile_verify returns on a TimeoutExpired.
        return CompileResult(
            ok=False,
            errors="timeout after 90s",
            used_sorry=False,
            used_native_decide=False,
        )

    monkeypatch.setattr(compile_step, "compile_lean", fake_compile_lean)
    res = compile_step.step("theorem t : True", (), "decide", timeout_s=90)
    assert res.outcome is compile_step.StepOutcome.FAIL
    assert res.error.startswith("timeout after"), (
        f"compile_step.step must preserve timeout-prefix in error; got {res.error!r}"
    )


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
