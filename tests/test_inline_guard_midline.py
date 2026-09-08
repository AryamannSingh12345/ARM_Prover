"""A tactic already on ONE line can still be structure-sensitive.

`_NON_INLINABLE_LINE_RE` decides whether a leaf tactic may be collapsed
into a single `; `-joined `first | (…) | fallback | …` alternative. Its
first two branches test a line's START and its END — which catches the
multi-line spellings and misses the identical constructs mid-line. When
the model emits its tactic already on one line, that is exactly where
they land.

Root cause of `putnam_1982_b4` (run `p1982b4_v1`, 7921 s, FAILED). The
assembled body carried

  * one 451-char ladder line with THREE `:= by` (char cols 80/154/226) —
    each `by` opens a tactic block that greedily swallows every following
    `; `-joined part, so those parts silently become part of proving the
    `have` rather than the goal;
  * one 542-char ladder line with TWO `; ·` — a focus bullet after a
    semicolon, which is not valid Lean in any position.

Lean reported `unexpected token 'by'; expected '{' or tactic` fourteen
times, across 3 sketches and 9 abduction rounds, at the same column,
while the mathematics underneath changed completely each round. The
theory loop had already found the correct argument (all |i| > 1, the two
neighbour products too small to both be divisible by ∏i) three separate
times; not one lemma ever reached the kernel.

The repair loop cannot escape this class on its own: an ASSEMBLER syntax
error is indistinguishable to it from a MODEL syntax error, so it asks
for a new tactic and re-flattens the answer the same way.

Failing the guard is cheap: `_wrap_with_fallback` returns the model's
tactic verbatim, which its own docstring already calls the better trade.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.dag.assembly import (                               # noqa: E402
    _inlinable, _inline_tactic_body, _wrap_with_fallback,
)

LADDER = ("norm_num", "omega", "aesop", "decide")

#: Verbatim from the p1982b4 assembly (`results/p1982b4_v1.jsonl`),
#: trimmed to the shape that matters.
LIVE_NESTED_BY = (
    "intro h0; have hd := hdiv 1; "
    "have hprod0 : (∏ i ∈ n, i) = 0 := by simp [Finset.prod_eq_zero h0]; "
    "have hshift : (∏ i ∈ n, (i + 1)) ≠ 0 := by apply Finset.prod_ne_zero_iff.mpr; "
    "rw [hprod0] at hd; exact hshift (zero_dvd_iff.mp hd)"
)
LIVE_MIDLINE_BULLET = (
    "intro i hi; by_cases hpos : 0 ≤ i; "
    "· have hi2 : 2 ≤ i := by omega; omega; "
    "· have hi2 : i ≤ -2 := by omega; omega"
)


# ---- the gap, stated as the four shapes -------------------------------------

def test_multiline_nested_by_is_rejected():
    """Already worked: the line ENDS with `by`."""
    assert not _inlinable("have h : T := by\n  simp")


def test_singleline_nested_by_is_rejected():
    """The regression. Nothing about the hazard depends on the newline."""
    assert not _inlinable("intro h; have a : T := by simp; exact a")


def test_multiline_bullets_are_rejected():
    """Already worked: the line STARTS with `·`."""
    assert not _inlinable("constructor\n· simp\n· simp")


def test_singleline_bullets_are_rejected():
    """The regression, second shape."""
    assert not _inlinable("by_cases h : p; · simp; · simp")


# ---- the two bodies that actually cost the run ------------------------------

def test_the_live_nested_by_body_is_rejected():
    assert not _inlinable(LIVE_NESTED_BY)


def test_the_live_bullet_body_is_rejected():
    assert not _inlinable(LIVE_MIDLINE_BULLET)


def test_rejected_tactics_reach_lean_verbatim():
    """The whole point of rejecting: no ladder, no flattening, no
    corruption. The model's tactic is passed through untouched."""
    for t in (LIVE_NESTED_BY, LIVE_MIDLINE_BULLET):
        assert _wrap_with_fallback(t, LADDER) == t


def test_a_corrupting_ladder_is_never_built():
    """States the invariant over the output rather than the input: no
    assembled alternative may contain a swallowing `by` or a `; ·`."""
    for t in (LIVE_NESTED_BY, LIVE_MIDLINE_BULLET):
        out = _wrap_with_fallback(t, LADDER)
        if out.startswith("first | "):
            assert ":= by" not in out and "; ·" not in out


# ---- honest tactics must still be laddered ----------------------------------

def test_ordinary_tactics_are_still_inlinable():
    """Over-rejecting costs every leaf its fallback ladder, so the guard
    must stay narrow. `by` inside a TERM (`fun`, `⟨by …⟩`) is not a
    `have := by` block and does not swallow a following `;`."""
    for t in ("simp [foo]",
              "intro h\nexact h",
              "have a : T := foo bar; exact a",
              "nlinarith [sq_nonneg (a-b), sq_nonneg (b-c)]",
              "refine ⟨1, ?_⟩; norm_num"):
        assert _inlinable(t), t


def test_an_inlinable_multiline_tactic_still_gets_the_ladder():
    out = _wrap_with_fallback("intro h\nexact h", LADDER)
    assert out.startswith("first | (intro h; exact h)")


def test_inlining_still_joins_with_semicolons():
    assert _inline_tactic_body("intro h\nexact h") == "intro h; exact h"
