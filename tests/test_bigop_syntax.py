"""Big-operator binder drift, and surgical repair of parse errors.

Mathlib changed `∑ x in s, f x` to `∑ x ∈ s, f x`. The old spelling is a
hard PARSE error in the pin ("unexpected token 'in'; expected ','"), and
models emit it constantly because it was correct for most of their
training data.

Cost on putnam_2020_a2_v2: `aux_weighted_choose_to_lower` — the only
mathematically hard lemma in the problem — burned its whole retry budget
on it. Three attempts, 824s + 749s + 547s = 2,120s of Lean, every one
dying at the parser. Its Pascal argument was never evaluated.

Two defences, tested here:
  1. deterministic rewrite in `apply_pin_renames` (same class as the `₀`
     renames: pin-version drift, uniform, no problem knowledge);
  2. `prove_lemma_prompt(syntax_only=True)` — a MINIMAL-EDIT repair for
     parse errors, instead of regenerating a whole proof that tends to
     repeat the same bad token.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.proof_dag import (                                # noqa: E402
    apply_pin_renames, prove_lemma_prompt,
)


# ---- 1. deterministic rewrite ----------------------------------------------

def test_sum_binder_in_is_rewritten():
    assert apply_pin_renames("∑ j in Finset.range (n+1), f j") == \
        "∑ j ∈ Finset.range (n+1), f j"


def test_prod_binder_in_is_rewritten():
    assert apply_pin_renames("∏ i in s, g i") == "∏ i ∈ s, g i"


def test_the_exact_failing_statement_is_fixed():
    """Verbatim from the putnam_2020_a2_v1 theory proposal."""
    bad = ("lemma aux_weighted_choose_transform (m n : ℕ) : "
           "(∑ j in Finset.range (n + 1), 2 ^ (n - j) * Nat.choose (m + j) j)"
           " = ∑ j in Finset.range (n + 1), Nat.choose (m + n + 1) j")
    out = apply_pin_renames(bad)
    assert " in " not in out
    assert out.count("∈") == 2


def test_already_correct_is_untouched():
    good = "∑ j ∈ Finset.range n, f j"
    assert apply_pin_renames(good) == good


def test_word_containing_in_is_safe():
    """`interval`, `sin`, `index` must not be mangled."""
    s = "∑ x ∈ interval, Real.sin x"
    assert apply_pin_renames(s) == s


def test_binder_match_does_not_cross_a_comma():
    """An `in` in the BODY of a sum belongs to something else."""
    s = "∑ x ∈ s, foo (bar in baz)"
    assert apply_pin_renames(s) == s


def test_two_big_operators_both_rewritten():
    out = apply_pin_renames("∑ x in s, f x + ∏ y in t, g y")
    assert out == "∑ x ∈ s, f x + ∏ y ∈ t, g y"


def test_zero_renames_still_applied():
    """The pre-existing ₀ rewrite must keep working."""
    assert apply_pin_renames("div_le_div_iff") == "div_le_div_iff₀"
    assert apply_pin_renames("∑ i in s, div_le_iff") == "∑ i ∈ s, div_le_iff₀"


def test_multiline_binder_does_not_span_lines():
    s = "∑ x ∈ s, f x\nlet y := 1 in y"
    assert apply_pin_renames(s) == s


# ---- 2. minimal-edit parse repair ------------------------------------------

PROOF = "induction n with\n| zero => simp\n| succ k ih => ∑ j in s, f j"
ERR = "unexpected token 'in'; expected ','"


def test_syntax_only_prompt_returns_the_previous_proof():
    p = prove_lemma_prompt("lemma L : True", ERR, None,
                           prev_proof=PROOF, syntax_only=True)
    assert PROOF in p
    assert "MINIMAL edit" in p
    assert "SYNTAX error only" in p


def test_syntax_only_prompt_forbids_rethinking():
    p = prove_lemma_prompt("lemma L : True", ERR, None,
                           prev_proof=PROOF, syntax_only=True)
    assert "do not rethink the proof" in p
    assert "Change nothing else" in p


def test_syntax_only_needs_a_previous_proof():
    """With nothing to repair it must fall back to the normal prompt."""
    p = prove_lemma_prompt("lemma L : True", ERR, None,
                           prev_proof=None, syntax_only=True)
    assert "Lemma to prove" in p
    assert "MINIMAL edit" not in p


def test_normal_prompt_unchanged_when_not_syntax_only():
    p = prove_lemma_prompt("lemma L : True", ERR, ["Nat.succ_le"])
    assert "Lemma to prove" in p
    assert "Nat.succ_le" in p
    assert "MINIMAL edit" not in p
