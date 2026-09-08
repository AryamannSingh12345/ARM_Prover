"""A line break inside a bracket is a continuation, not a new tactic.

Regression test for the `unknown tactic` failure that killed
`lrs_a0_unit_open_v1` (2026-08-26). `_inline_tactic_body` joined EVERY
line with `; `, so a tactic whose argument list spanned lines came out
with a statement separator inside the list:

    nlinarith [sq_nonneg (a - b),
      sq_nonneg (b - c)]
    -> nlinarith [sq_nonneg (a - b),; sq_nonneg (b - c)]

That does not parse. Lean reports it as `unknown tactic`, and on the LRS
run it killed three satisfaction-gate probes AND the final verify — the
run's terminal failure was `Try_505296e7d621.lean:99:43: unknown tactic`
with an `unsolved goals` on the line above.

`_NON_INLINABLE_LINE_RE` cannot catch this shape: no line starts with a
bullet and none ends with `by` / `:=`, so the block looks inlinable.
Multi-line `simp only [...]` and `refine ⟨_, _, _⟩` are the common
cases, and both are frequent on Polynomial goals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.dag.assembly import (  # noqa: E402
    _inline_tactic_body, _wrap_with_fallback,
)


@pytest.mark.parametrize("tactic,expected", [
    # The exact shape that killed the LRS run.
    ("nlinarith [sq_nonneg (a - b),\n  sq_nonneg (b - c)]",
     "nlinarith [sq_nonneg (a - b), sq_nonneg (b - c)]"),
    # Anonymous constructor — refine/exact lean on these.
    ("refine ⟨_, _,\n  _⟩", "refine ⟨_, _, _⟩"),
    # simp only with a wrapped lemma list.
    ("simp only [Polynomial.natDegree_X_pow,\n  Polynomial.natDegree_C]",
     "simp only [Polynomial.natDegree_X_pow, Polynomial.natDegree_C]"),
    # Parens.
    ("apply foo (bar\n  baz)", "apply foo (bar baz)"),
    # Three-line continuation.
    ("simp [a,\n  b,\n  c]", "simp [a, b, c]"),
])
def test_continuation_lines_rejoin_with_space(tactic, expected):
    assert _inline_tactic_body(tactic) == expected


@pytest.mark.parametrize("tactic,expected", [
    ("have h := foo\nrw [h]\nomega", "have h := foo; rw [h]; omega"),
    ("intro x\nsimp", "intro x; simp"),
    # Balanced brackets on each line: still separate tactics.
    ("simp [a]\nomega", "simp [a]; omega"),
])
def test_genuine_multi_tactic_blocks_still_use_semicolons(tactic, expected):
    assert _inline_tactic_body(tactic) == expected


def test_no_semicolon_inside_any_bracket():
    """The invariant the bug violated, stated directly."""
    out = _inline_tactic_body(
        "refine ⟨fun h => ?_,\n  fun h => ?_⟩\nsimp only [foo,\n  bar]")
    depth = 0
    for ch in out:
        if ch in "([{⟨":
            depth += 1
        elif ch in ")]}⟩":
            depth -= 1
        elif ch == ";":
            assert depth == 0, f"separator inside a delimiter: {out}"
    assert depth == 0


def test_bracket_inside_string_literal_is_not_counted():
    """A `]` in a string must not close a real bracket."""
    assert _inline_tactic_body('simp [foo]\nexact bar "]("') == (
        'simp [foo]; exact bar "]("')


def test_wrapped_ladder_stays_parseable_shape():
    """The full ladder wrap must not reintroduce the bad separator."""
    wrapped = _wrap_with_fallback(
        "simp only [Polynomial.natDegree_X_pow,\n  Polynomial.natDegree_C]",
        ("omega", "norm_num"),
    )
    assert ",;" not in wrapped
    assert wrapped.startswith("first | (")
    assert "(omega)" in wrapped and "(norm_num)" in wrapped


def test_single_line_tactic_is_unchanged():
    t = "nlinarith [sq_nonneg (a-b), sq_nonneg (b-c)]"
    assert _inline_tactic_body(t) == t
