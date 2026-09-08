"""The closer block must be indented like every other body.

`assemble_proof_with_map` builds the tactic body spliced under the
theorem's `:= by`. Each `have`'s tactic is indented uniformly (base 4);
the closer used to be special-cased — a line that already began with a
space was appended VERBATIM, without the base indent.

That silently broke any closer containing a nested `have … := by`:

    have hclosed : ∀ m : ℕ, S m = 1 := by     <- unindented, so got base 2
    intro m                                   <- model indented it 2, kept at 2

The `have` and its body land in the SAME column, so Lean reads the `by`
block as empty and reports

    error: expected '{' or indented tactic sequence

Root cause of `putnam_1967_a2` (run `p1967a2_v3`, 6100 s, FAILED): six
occurrences across four satisfaction-gate probes, zero lemmas to the
kernel. The theory had the problem right all along — symmetric matrices
with unit column sums are involutions, `a(n+1) = a(n) + n·a(n-1)`, EGF
`exp(x + x²/2)` — and never got to test any of it.

The compounding harm is worth recording: `retry_with_error_feedback` fed
those syntax errors back as if they were the sketch's fault, and sketch 3
duly abandoned the (correct) generating-function architecture for trivia
about 1x1 matrices. An assembler defect can degrade the sketch policy,
not merely waste a round.

A leading bare `by` line is stripped for the same reason — the body is
already under `:= by`, so a second one is always spurious. Seen twice in
the same run (`007_verify.lean`, `014_verify.lean`).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.dag.assembly import assemble_proof_with_map          # noqa: E402
from search.proof_dag import CLOSER_ID, HaveNode, Sketch         # noqa: E402

#: The shape the theory probe emitted on p1967a2_v3.
NESTED_HAVE_CLOSER = (
    "have hclosed : ∀ m : ℕ, S m = 1 := by\n"
    "  intro m\n"
    "  cases m with\n"
    "  | zero => simp\n"
    "  | succ k => simp\n"
    "exact hclosed"
)


def _body(closer: str, haves=None) -> list[str]:
    sk = Sketch(haves=haves if haves is not None
                else [HaveNode("h1", "True", "trivial", [])],
                closer=closer)
    body, _ = assemble_proof_with_map(sk)
    return body.split("\n")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


# ---- the defect --------------------------------------------------------------

def test_nested_have_body_is_deeper_than_its_have():
    """The exact failure: `have … := by` and its body at equal indent."""
    lines = _body(NESTED_HAVE_CLOSER)
    hv = next(i for i, l in enumerate(lines) if "have hclosed" in l)
    assert _indent(lines[hv + 1]) > _indent(lines[hv]), (
        "closer's nested `have` body is not indented past the `have` — "
        "Lean reads the `by` block as empty")


def test_every_closer_line_carries_the_base_indent():
    for l in _body(NESTED_HAVE_CLOSER):
        if l.strip():
            assert _indent(l) >= 2, repr(l)


def test_relative_indentation_is_preserved():
    """Base indent is added, model structure is not flattened."""
    lines = _body(NESTED_HAVE_CLOSER)
    hv = next(i for i, l in enumerate(lines) if "have hclosed" in l)
    body = [l for l in lines[hv + 1:] if l.strip() and "exact hclosed" not in l]
    assert all(_indent(l) == _indent(lines[hv]) + 2 for l in body)


# ---- the leading `by` --------------------------------------------------------

def test_a_leading_bare_by_is_stripped():
    lines = _body("by\nclassical\nexact h1")
    assert not any(l.strip() == "by" for l in lines)
    assert any(l.strip() == "classical" for l in lines)


def test_by_inside_the_closer_is_untouched():
    """Only a LEADING bare `by` is spurious."""
    lines = _body("refine ⟨?_, ?_⟩\n· exact (by simp)\n· exact h1")
    assert any("(by simp)" in l for l in lines)


# ---- unchanged behaviour -----------------------------------------------------

def test_single_line_closer_is_unchanged():
    assert _body("exact h1")[-1] == "  exact h1"


def test_have_bodies_still_indent_by_four():
    lines = _body("exact h1", haves=[HaveNode("h1", "True", "trivial", [])])
    hv = next(i for i, l in enumerate(lines) if "have h1" in l)
    assert lines[hv + 1] == "    trivial"


def test_the_line_map_still_covers_the_closer():
    """Attribution depends on it: a Lean error at body line N must map
    back to a segment, or repair blames the wrong leaf."""
    sk = Sketch(haves=[HaveNode("h1", "True", "trivial", [])],
                closer=NESTED_HAVE_CLOSER)
    body, segs = assemble_proof_with_map(sk)
    n = len(body.split("\n"))
    cl = [s for s in segs if s[0] == CLOSER_ID]
    assert len(cl) == 1
    _, start, end = cl[0]
    assert end == n and start <= end
