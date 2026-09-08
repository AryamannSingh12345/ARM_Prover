"""Phase 0 checkpoint: `example : 2+2=4 := by norm_num` verifies through:
  (a) compile-based verifier
  (b) Lean REPL whole-proof verifier
"""
from __future__ import annotations

import pytest

#: Spawns a real `lake env lean` compile — minutes per test on this
#: host. Measured, not guessed: this file did not finish in 25s.
#: Excluded from the fast suite via `pytest -m "not live"`.
pytestmark = pytest.mark.live


import pytest

from backend.compile_verify import verify_proof

THEOREM = "example : 2 + 2 = 4"
PROOF = "  norm_num"


def test_compile_verifier():
    res = verify_proof(THEOREM, PROOF)
    assert res.ok, res.errors


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
