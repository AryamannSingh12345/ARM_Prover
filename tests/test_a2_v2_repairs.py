"""Repairs for the two defects measured on p1963a2_dag_v2 (2026-08-18).

1. SPURIOUS-INTRO. In closer-stuck mode the closer obligation is posed
   standalone, so the model opens its tactic with `intro m hm` — but the
   sketch's `setup` has already introduced those binders, and Lean fails
   with "Tactic `introN` failed: There are no additional binders". This
   killed theory rounds 0 AND 4 of that run, and round 4 was the GOOD
   decomposition (elementary gap-persistence / dyadic fixed points).

2. PARTIAL COMMIT vs UNPROVED NAMES. The partial-commit probe reused the
   round's tactics verbatim, so a tactic citing a lemma that never proved
   died on `unknown identifier` instead of answering the question the
   probe exists to ask ("does the proved subset suffice?"). Measured
   cost: one 455s compile spent learning nothing.
"""
from __future__ import annotations

import re

import pytest

from search.proof_dag import (
    _INTRO_NO_BINDERS_RE,
    strip_leading_intro,
    tactic_cites,
)

# The verbatim Lean diagnostic from the run (v4.30.0-rc2).
LEAN_INTRO_ERR = (
    "Tactic `introN` failed: There are no additional binders or `let` "
    "bindings in the goal to introduce\nf : ℕ → ℕ\n"
    "hfpos : ∀ (n : ℕ), f n > 0"
)

# The verbatim round-4 closer tactic that was lost.
A2_ROUND4_CLOSER = (
    "intro m hm\n"
    "apply Nat.le_antisymm\n"
    "· by_contra hupper\n"
    "  have hgap : m < f m := Nat.lt_of_not_ge hupper\n"
    "  have hmk : m ≤ 2 ^ m := aux_nat_le_two_pow m\n"
    "  have hpersist : 2 ^ m < f (2 ^ m) := "
    "aux_nat_gap_persists f hfinc hm hmk hgap"
)


# ---------------------------------------------------------------- detection

def test_regex_matches_the_real_lean_diagnostic():
    assert _INTRO_NO_BINDERS_RE.search(LEAN_INTRO_ERR)


@pytest.mark.parametrize("msg", [
    "Tactic `intro` failed: There are no additional binders to introduce",
    "tactic 'introN' failed: there are no additional binders",
])
def test_regex_tolerates_spelling_variants(msg):
    assert _INTRO_NO_BINDERS_RE.search(msg)


@pytest.mark.parametrize("msg", [
    "unsolved goals\nf : ℕ → ℕ",
    "Tactic `rewrite` failed: Did not find an occurrence of the pattern",
    "omega could not prove the goal",
    # An intro failure for a DIFFERENT reason must not match — stripping
    # would not help and could destroy a correct tactic.
    "Tactic `introN` failed: expected a term",
])
def test_regex_does_not_fire_on_unrelated_errors(msg):
    assert not _INTRO_NO_BINDERS_RE.search(msg)


# ------------------------------------------------------------------- strip

def test_strips_the_real_lost_closer():
    out = strip_leading_intro(A2_ROUND4_CLOSER)
    assert out is not None
    assert out.startswith("apply Nat.le_antisymm")
    assert "intro m hm" not in out
    # Everything else must survive verbatim.
    for keep in ("by_contra hupper", "aux_nat_le_two_pow m",
                 "aux_nat_gap_persists f hfinc hm hmk hgap"):
        assert keep in out


def test_strips_intros_plural_and_leading_blank_lines():
    assert strip_leading_intro("\n\n  intros a b\n  simp") == "  simp"


def test_returns_none_when_first_line_is_not_an_intro():
    for t in ("simp\nintro x", "apply foo\n  intro y", "  omega"):
        assert strip_leading_intro(t) is None


def test_returns_none_rather_than_producing_an_empty_tactic():
    # Stripping an intro-only block would leave a parse error, which is
    # worse than the original failure.
    assert strip_leading_intro("intro n hn") is None
    assert strip_leading_intro("  intro n hn  \n\n") is None


def test_only_the_first_intro_is_removed():
    out = strip_leading_intro("intro a\nrefine ?_\nintro b\nsimp")
    assert out == "refine ?_\nintro b\nsimp"


def test_strip_is_idempotent_in_the_sense_that_it_stops():
    # Applying twice must not keep eating the proof.
    once = strip_leading_intro("intro a\nintro b\nsimp")
    assert once == "intro b\nsimp"
    twice = strip_leading_intro(once)
    assert twice == "simp"
    assert strip_leading_intro(twice) is None


# ------------------------------------------------------- unproved-name cite

def test_cites_the_real_unproved_lemma():
    unproved = ["aux_monotone_multiplicative_power_classification"]
    closer = ("obtain ⟨c, hc⟩ := "
              "aux_monotone_multiplicative_power_classification f hfpos")
    assert tactic_cites(closer, unproved)


def test_does_not_cite_when_only_proved_lemmas_appear():
    unproved = ["aux_monotone_multiplicative_power_classification"]
    assert not tactic_cites(A2_ROUND4_CLOSER, unproved)


def test_substring_names_do_not_false_positive():
    # `aux_foo` must not match inside `aux_foo_bar`.
    assert not tactic_cites("exact aux_foo_bar x", ["aux_foo"])
    assert not tactic_cites("exact my_aux_foo x", ["aux_foo"])
    assert tactic_cites("exact aux_foo x", ["aux_foo"])


def test_dotted_qualified_use_is_not_a_bare_cite():
    # `Foo.aux_bar` is a different declaration from `aux_bar`.
    assert not tactic_cites("exact Foo.aux_bar", ["aux_bar"])


def test_empty_and_degenerate_inputs_are_safe():
    assert not tactic_cites("simp", [])
    assert not tactic_cites("simp", [""])
    assert not tactic_cites("", ["aux_x"])


def test_regex_metacharacters_in_a_name_are_escaped():
    # A malformed name must not raise or match everything.
    assert not tactic_cites("simp", ["aux.*"])
    assert tactic_cites("exact aux.*", ["aux.*"]) or True  # must not raise


# --------------------------------------------------------------- integration

def test_the_a2_scenario_end_to_end():
    """The exact p1963a2_dag_v2 situation, in both halves."""
    # Half 1: the closer that was lost is now repairable.
    fixed = strip_leading_intro(A2_ROUND4_CLOSER)
    assert fixed is not None and not fixed.startswith("intro")

    # Half 2: the round-3 closer cited the lemma that never proved, so
    # the partial commit must drop it instead of buying a 455s probe.
    unproved = ["aux_monotone_multiplicative_power_classification"]
    tactics = {
        "__closer__": ("obtain ⟨c, hc⟩ := "
                       "aux_monotone_multiplicative_power_classification f"),
    }
    survivors = {k: v for k, v in tactics.items()
                 if not tactic_cites(v, unproved)}
    assert survivors == {}, "nothing should survive; probe must be skipped"


def test_source_carries_the_measurement_not_just_the_fix():
    """The repair must stay explainable: both defects are keyed to a run."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src/search/proof_dag.py"
    text = src.read_text(encoding="utf-8")
    assert "p1963a2_dag_v2" in text
    # The intro repair must remain error-driven, never speculative.
    i = text.index("def strip_leading_intro")
    assert "ERROR-DRIVEN" in text[:i]
    assert re.search(r"_intro_retry_ok\s*=\s*False", text), \
        "the re-probe must be capped at one retry"
