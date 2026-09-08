"""A proof may not cite the theorem it is proving.

Lean elaborates a declaration whose body mentions its own name as
RECURSIVE, and reports "fail to show termination / failed to infer
structural recursion". That is a HEADER error, which aborts the whole
attempt by design — attribution after a broken header is unreliable — so
one stray self-citation throws away the run.

Observed on p1967a2_v1, which lost BOTH remaining sketch attempts to it:

    931s   __header__: fail to show termination for putnam_1967_a2
    2747s  __header__: fail to show termination for putnam_1967_a2
    2747s  outcome FAILED

Caught before the compile: the verdict is already known, and a round is
worth more than the 200-600s it costs to be told.

The subtlety worth pinning: PutnamBench statements sit beside a companion
`..._solution` abbrev that proofs legitimately cite. A naive substring
check would reject every such proof.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.proof_dag import (                                  # noqa: E402
    _own_decl_name, references_own_name,
)

HEADER = ("set_option maxHeartbeats 1000000\n"
          "abbrev putnam_1967_a2_solution : ℕ := 1\n"
          "theorem putnam_1967_a2 (S : ℕ → ℤ) (hS0 : S 0 = 1) : True")


# ---- name extraction --------------------------------------------------------

def test_reads_the_final_declaration_name():
    assert _own_decl_name(HEADER) == "putnam_1967_a2"


def test_reads_through_modifiers():
    assert _own_decl_name("private theorem foo (n : ℕ) : True") == "foo"
    assert _own_decl_name("lemma bar : True") == "bar"


def test_no_declaration_yields_empty():
    assert _own_decl_name("import Mathlib") == ""


# ---- the guard --------------------------------------------------------------

def test_self_citation_is_caught():
    body = "  intro n\n  exact putnam_1967_a2 S hS0"
    assert references_own_name(HEADER, body) == "putnam_1967_a2"


def test_self_citation_anywhere_in_the_body_is_caught():
    body = "  simpa [putnam_1967_a2] using h"
    assert references_own_name(HEADER, body) == "putnam_1967_a2"


def test_the_companion_solution_abbrev_is_NOT_a_self_citation():
    """Every PutnamBench `find the value` problem cites `..._solution`.
    A substring check would reject all of them."""
    body = "  simp [putnam_1967_a2_solution]\n  norm_num"
    assert references_own_name(HEADER, body) is None


def test_an_unrelated_name_sharing_a_prefix_is_not_caught():
    body = "  exact putnam_1967_a2b_helper"
    assert references_own_name(HEADER, body) is None


def test_a_clean_proof_passes():
    body = "  intro n\n  simpa using hS0"
    assert references_own_name(HEADER, body) is None


def test_a_dotted_projection_is_not_a_self_citation():
    """`h.putnam_1967_a2` is a field access, not the theorem."""
    assert references_own_name(HEADER, "  exact h.putnam_1967_a2") is None


def test_a_root_qualified_citation_IS_caught():
    """`_root_.foo` is dotted, but it is exactly how one names the
    top-level declaration — a genuine self-citation.

    The dotted-prefix exclusion that protects `h.putnam_1967_a2` used to
    swallow this too. Observed on p1967a2_v4 (2026-08-12): the closer
    emitted `_root_.putnam_1967_a2` at t=10538s, passed this guard, and
    reached Lean — which only rejected it as `Unknown identifier` because
    the theorem is not in scope during its own elaboration."""
    assert (references_own_name(HEADER, "  exact _root_.putnam_1967_a2 S hS0")
            == "putnam_1967_a2")


def test_root_qualified_companion_solution_is_still_allowed():
    """The trailing-`_` rule must survive the `_root_.` addition."""
    assert references_own_name(
        HEADER, "  simp [_root_.putnam_1967_a2_solution]") is None


def test_root_qualified_unrelated_name_is_not_caught():
    assert references_own_name(
        HEADER, "  exact _root_.putnam_1967_a2b_helper") is None


def test_headerless_input_is_safe():
    assert references_own_name("", "exact foo") is None
    assert references_own_name(HEADER, "") is None


# ---- wiring -----------------------------------------------------------------

def test_the_guard_runs_before_the_compile():
    """The whole point is to spend no verify on a known-bad body."""
    import inspect
    import search.proof_dag as pd
    src = inspect.getsource(pd.attempt_dag_proof)
    i_guard = src.index("_self_ref = references_own_name")
    i_verify = src.index("res = verify_fn(theorem_header, body)")
    assert i_guard < i_verify


def test_the_feedback_names_the_actual_failure():
    """The model must be told WHY, not just that it failed — otherwise it
    re-emits the same self-citation. Fragments are checked individually
    because the message is wrapped across source lines."""
    import inspect
    import search.proof_dag as pd
    src = inspect.getsource(pd.attempt_dag_proof)
    for frag in ("fail to show termination", "RECURSIVE",
                 "the theorem being ", "never cite the "):
        assert frag in src, frag
