"""`--store-feedback` must never splice two defs under one name.

MEASURED on p1965a2_dag_v1 (2026-08-22). The theory revised its auxiliary
def between rounds:

    round 0: def aux_chooseDeviationTerm (n r : N) : Z := (...) ^ 2
    round 1: def aux_chooseDeviationTerm (n r : N) : Z := ((... : Z) ^ (2 : N))

Both bodies were stored, attached to lemmas proved in different rounds.
`required_defs()` deduplicated by SOURCE TEXT, so both survived; the
caller filtered against the header, which contained neither; and both
were spliced into sketch attempt 3's header at t=40799s. Lean reported
`aux_chooseDeviationTerm has already been declared` — a HEADER error,
which stops the repair loop by design — and the attempt died on its first
compile with three kernel-proved lemmas in hand. The run had already spent
11 hours.

Two properties are pinned here:
  1. one declaration per NAME reaches the header, whichever route;
  2. a lemma requiring a SUPERSEDED body is withheld rather than surfaced
     into a header where its def means something else.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.dag.lemma_store import LemmaStore, decl_name  # noqa: E402

PRELUDE = "import Mathlib\n"

DEF_V0 = ("def aux_chooseDeviationTerm (n r : ℕ) : ℤ := "
          "(((n : ℤ) - 2 * (r : ℤ)) * (Nat.choose n r : ℤ)) ^ 2")
DEF_V1 = ("def aux_chooseDeviationTerm (n r : ℕ) : ℤ := "
          "(((((n : ℤ) - 2 * (r : ℤ)) * (Nat.choose n r : ℤ)) : ℤ) ^ (2 : ℕ))")
DEF_OTHER = ("def aux_prevChoose (m r : ℕ) : ℤ := "
             "if r = 0 then 0 else (Nat.choose m (r - 1) : ℤ)")


def _store_with_revised_def() -> LemmaStore:
    """The exact p1965a2 situation: two bodies, one name, two lemmas."""
    s = LemmaStore()
    s.put("lemma aux_a : True", "aux_a", "lemma aux_a : True := by trivial",
          PRELUDE, defs=(DEF_V0,), theory_round=0)
    s.put("lemma aux_b : True", "aux_b", "lemma aux_b : True := by trivial",
          PRELUDE, defs=(DEF_V1,), theory_round=1)
    return s


# ---------------------------------------------------------------- decl_name

def test_decl_name_extracts_the_identifier():
    assert decl_name(DEF_V0) == "aux_chooseDeviationTerm"
    assert decl_name(DEF_V1) == "aux_chooseDeviationTerm"
    assert decl_name(DEF_OTHER) == "aux_prevChoose"


def test_decl_name_handles_modifiers_and_attributes():
    assert decl_name("noncomputable def f (x : ℕ) : ℕ := x") == "f"
    assert decl_name("private abbrev g := 1") == "g"
    assert decl_name("@[simp] def h : ℕ := 0") == "h"
    assert decl_name("structure S where\n  x : ℕ") == "S"


def test_decl_name_is_empty_for_non_declarations():
    assert decl_name("") == ""
    assert decl_name("-- just a comment") == ""
    assert decl_name("lemma foo : True := by trivial") == ""


# ------------------------------------------------------- required_defs

def test_two_bodies_one_name_yield_ONE_declaration():
    """The regression. Text dedup let both through; name dedup must not."""
    defs = _store_with_revised_def().required_defs()
    names = [decl_name(d) for d in defs]
    assert names.count("aux_chooseDeviationTerm") == 1, (
        f"duplicate declaration would be spliced: {names}")
    assert len(defs) == 1


def test_first_seen_body_wins():
    assert _store_with_revised_def().required_defs() == [DEF_V0]


def test_distinct_names_are_all_kept():
    s = LemmaStore()
    s.put("lemma a : True", "a", "lemma a : True := by trivial",
          PRELUDE, defs=(DEF_V0,))
    s.put("lemma b : True", "b", "lemma b : True := by trivial",
          PRELUDE, defs=(DEF_OTHER,))
    assert set(map(decl_name, s.required_defs())) == {
        "aux_chooseDeviationTerm", "aux_prevChoose"}


def test_identical_body_reused_is_still_deduplicated():
    s = LemmaStore()
    s.put("lemma a : True", "a", "lemma a : True := by trivial",
          PRELUDE, defs=(DEF_V0,))
    s.put("lemma b : True", "b", "lemma b : True := by trivial",
          PRELUDE, defs=(DEF_V0,))
    assert s.required_defs() == [DEF_V0]


def test_no_defs_means_no_declarations():
    s = LemmaStore()
    s.put("lemma a : True", "a", "lemma a : True := by trivial", PRELUDE)
    assert s.required_defs() == []


# --------------------------------------------------- incompatible_names

def test_lemma_needing_the_superseded_body_is_flagged():
    bad = _store_with_revised_def().incompatible_names()
    assert bad == {"aux_b"}, (
        "aux_b was proved under DEF_V1; surfacing it beside DEF_V0 would "
        "put it in a header where its def means something else")


def test_nothing_is_flagged_when_defs_are_consistent():
    s = LemmaStore()
    s.put("lemma a : True", "a", "lemma a : True := by trivial",
          PRELUDE, defs=(DEF_V0,))
    s.put("lemma b : True", "b", "lemma b : True := by trivial",
          PRELUDE, defs=(DEF_V0, DEF_OTHER))
    assert s.incompatible_names() == set()


def test_defless_lemmas_are_never_flagged():
    s = _store_with_revised_def()
    s.put("lemma aux_c : True", "aux_c", "lemma aux_c : True := by trivial",
          PRELUDE)
    assert "aux_c" not in s.incompatible_names()


# ------------------------------------------------------------- wiring

def test_caller_guards_within_a_single_batch():
    """`_declared` is the header BEFORE the splice, so it cannot stop two
    same-named defs arriving together. The batch guard must."""
    src = (Path(__file__).resolve().parents[1]
           / "src" / "search" / "proof_dag.py").read_text(encoding="utf-8")
    i = src.index("def _surface_store_lemmas")
    j = src.index("for attempt in range(1, sketch_attempts + 1)", i)
    body = src[i:j]
    assert "_batch" in body, "no within-batch name guard"
    assert "nm in _declared or nm in _batch" in body
    assert "incompatible_names()" in body, "withheld records not applied"


def test_motivation_is_recorded():
    for rel in ("src/search/dag/lemma_store.py",):
        t = (Path(__file__).resolve().parents[1] / rel).read_text(encoding="utf-8")
        assert "p1965a2_dag_v1" in t
        assert "has already been declared" in t
