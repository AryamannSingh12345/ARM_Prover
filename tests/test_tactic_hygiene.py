"""Regression tests for `harden_tactic_block` (2026-08-17).

Origin: the putnam_1967_b5 pipeline run (15.3 h, 64 compiles). Of 13 lost
lemma attempts, 9 — 69% — died of one of two non-mathematical causes:

  * `simp made no progress`, a hard ERROR in Lean 4, when a normalisation
    tactic lands on an already-normalised goal.  3 lemmas, including the
    crux (`aux_weighted_transform`) and both halves of its induction.
  * a trivial residual goal left unclosed, e.g. `1 + (1 + A) = 2 + A` or
    `∑ … + 1 = 1 + ∑ …`.  6 lemmas.  In one round the model had ALREADY
    proved `aux_one_add_one_add : 1 + (1 + A) = 2 + A` and still failed
    the same goal twice, because it never noticed the goal was open.

The safety property these tests pin: hardening cannot break a proof that
currently compiles, and it must not touch tactics whose failure is
load-bearing control flow.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.proof_dag import harden_tactic_block  # noqa: E402


# ------------------------------------------------------- no-op `simp`

def test_bare_simp_is_wrapped_in_try():
    out = harden_tactic_block("intro i\nsimp", sweep=False)
    assert out.splitlines() == ["intro i", "try simp"]


def test_simp_with_arguments_is_wrapped():
    out = harden_tactic_block("simp [Nat.choose_succ_succ, pow_succ]", sweep=False)
    assert out.strip().startswith("try simp [Nat.choose_succ_succ")


def test_all_noop_prone_tactics_are_covered():
    for tac in ("simp", "simp_all", "norm_num", "ring_nf",
                "push_cast", "field_simp", "dsimp", "norm_cast"):
        out = harden_tactic_block(tac, sweep=False)
        assert out.strip() == f"try {tac}", tac


def test_indentation_is_preserved():
    out = harden_tactic_block("  | succ n ih =>\n      simp", sweep=False)
    assert out.splitlines()[1] == "      try simp"


def test_case_marker_body_is_hardened_but_alternation_is_not():
    """`|` is overloaded in Lean; `=>` is the discriminator.

    A case body is ordinary tactics (harden it — this is where b5's
    `simp made no progress` failures actually lived). An alternation
    branch relies on failure for control flow (leave it alone).
    """
    case_body = harden_tactic_block(
        "induction m with\n| succ n ih =>\n  simp\n  exact ih", sweep=False)
    assert "  try simp" in case_body

    alternation = harden_tactic_block("first\n| simp\n| ring", sweep=False)
    assert "try" not in alternation


def test_non_noop_tactics_are_untouched():
    src = "induction m with\nexact foo\nrw [bar]\nomega"
    assert harden_tactic_block(src, sweep=False) == src


# ------------------------------- control flow must NOT be interfered with

def test_alternation_branch_is_not_wrapped():
    """`first | simp | ring` relies on simp FAILING to reach ring.

    Wrapping it in `try` would make the alternation always take its first
    branch — silently changing the proof's meaning.
    """
    src = "first\n| simp\n| ring"
    assert harden_tactic_block(src, sweep=False) == src


def test_seq_focus_line_is_not_wrapped():
    src = "constructor <;> simp"
    assert harden_tactic_block(src, sweep=False) == src


def test_inline_alternation_is_not_wrapped():
    src = "(simp | ring)"
    assert harden_tactic_block(src, sweep=False) == src


# -------------------------------------------------- the residual sweep

def test_sweep_is_appended_at_top_level_indent():
    out = harden_tactic_block("intro i\nexact foo").splitlines()
    assert out[:2] == ["intro i", "exact foo"]
    assert out[2:] == ["all_goals try omega", "all_goals try ring",
                       "all_goals try simp_all", "all_goals try ac_rfl"]


def test_sweep_matches_the_blocks_base_indent():
    out = harden_tactic_block("  intro i\n  exact foo").splitlines()
    assert all(l.startswith("  ") for l in out[-4:])
    assert out[-4] == "  all_goals try omega"


def test_sweep_can_be_disabled():
    assert "all_goals" not in harden_tactic_block("exact foo", sweep=False)


def test_empty_block_does_not_crash():
    harden_tactic_block("")
    harden_tactic_block(None)  # type: ignore[arg-type]


# ------------------------------------- the actual b5 failures, verbatim

def test_b5_crux_shape_gets_its_simp_guarded():
    """`aux_weighted_transform` died on exactly this shape, three times."""
    src = ("induction n with\n"
           "| zero => simp\n"
           "| succ n ih =>\n"
           "    rw [Finset.sum_range_succ]\n"
           "    simp")
    out = harden_tactic_block(src, sweep=False)
    # the bare trailing simp is guarded ...
    assert "    try simp" in out
    # ... but the one inside the `| zero =>` alternation branch is not
    assert "| zero => simp" in out


def test_residual_sweep_would_have_closed_the_add_comm_goals():
    """The six lost lemmas all ended with an unclosed commutativity goal.

    We cannot run Lean here, so we pin the mechanism: `omega`/`ring`/
    `ac_rfl` are present in the sweep, in that order, which is what
    closes `1 + (1 + A) = 2 + A` and `∑ … + 1 = 1 + ∑ …`.
    """
    out = harden_tactic_block("exact foo")
    assert "all_goals try omega" in out
    assert "all_goals try ac_rfl" in out


def test_hardening_is_idempotent():
    once = harden_tactic_block("simp", sweep=False)
    assert harden_tactic_block(once, sweep=False) == once
