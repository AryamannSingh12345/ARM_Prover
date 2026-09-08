"""Environment integrity: the goal must still mean what the problem says.

`sorry`-scanning and kernel re-verification are BOTH insufficient against
a declaration that mutates the ambient environment, because the kernel
then faithfully verifies a different, weaker statement — and the fresh
re-verify includes the poisoned declaration.

This was not hypothetical. On `frankl_open_v1` (2026-08-08) the prover
reported SOLVED on Frankl's OPEN conjecture with:

    instance (priority := 10000) aux_instLENat : LE ℕ where
      le _ _ := True

    intro A h_nonempty h_union; exact ⟨0, True.intro⟩

`≤` on ℕ was redefined to `True` at a priority beating Mathlib's, so
`2 * card ≥ A.card` became `True`. The proof contains no `sorry`, no
`admit`, no `native_decide` — every existing guard passed.

Three splice paths had to be closed: `_parse_theory`'s `defs`,
`reframe`'s `objects`, and `library.build_library`'s declarations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.decl_safety import first_unsafe, unsafe_declaration   # noqa: E402

EXPLOIT = ("instance (priority := 10000) aux_instLENat : LE ℕ where\n"
           "  le _ _ := True")


# ---- the exploit itself -----------------------------------------------------

def test_the_observed_exploit_is_refused():
    why = unsafe_declaration(EXPLOIT)
    assert why and "instance" in why


def test_axiom_is_refused():
    """Strictly worse than the instance: `axiom f : False` proves anything."""
    assert unsafe_declaration("axiom aux_cheat : False")


def test_notation_and_macro_are_refused():
    """These change what the GOAL's surface syntax parses to."""
    for src in ("notation:50 a \" ≤ \" b => True",
                "macro \"trivialise\" : tactic => `(tactic| trivial)",
                "syntax \"foo\" : term",
                "set_option maxHeartbeats 0 in\ndef f : ℕ := 0"):
        assert unsafe_declaration(src), src


def test_scoped_and_local_instances_are_refused():
    for src in ("local instance foo : LE ℕ where le _ _ := True",
                "scoped instance foo : LE ℕ where le _ _ := True"):
        assert unsafe_declaration(src), src


def test_instance_attribute_is_refused():
    assert unsafe_declaration("@[instance] def sneaky : LE ℕ := ⟨fun _ _ => True⟩")


# ---- honest declarations still pass ----------------------------------------

def test_ordinary_definitions_are_allowed():
    for src in ("def aux_weighted (m n : ℕ) : ℕ := m + n",
                "abbrev aux_two : ℕ := 2",
                "structure Frame where\n  carrier : ℕ",
                "inductive Colour | red | black"):
        assert unsafe_declaration(src) is None, src


def test_the_word_instance_inside_a_statement_is_fine():
    """`Nonempty (Invertible M)` appears in putnam_1991_a2. Rejecting the
    substring rather than the declaration head would break real problems."""
    assert unsafe_declaration(
        "def f : Prop := Nonempty (Invertible (A ^ 2 + B ^ 2))") is None
    assert unsafe_declaration(
        "lemma foo : ∀ n, instanceLike n = n := by simp") is None


def test_first_unsafe_reports_the_offender():
    bad = first_unsafe(["def ok : ℕ := 1", EXPLOIT, "def also_ok : ℕ := 2"])
    assert bad is not None and bad[0] == EXPLOIT


def test_first_unsafe_on_clean_input_is_none():
    assert first_unsafe(["def a : ℕ := 1", "abbrev b : ℕ := 2"]) is None


# ---- the splice paths -------------------------------------------------------

def test_theory_defs_refuse_the_exploit():
    import search.proof_dag as pd
    raw = json.dumps({
        "defs": [EXPLOIT],
        "lemmas": [{"name": "b1", "statement": "lemma b1 : True"}],
        "leaf_tactics": {"h1": "trivial"}})
    parsed, err = pd._parse_theory(raw)
    assert parsed is None
    assert "unsafe" in err


def test_theory_defs_no_longer_accept_instance_at_all():
    """`instance` was an explicitly permitted def head — that was the hole."""
    import search.proof_dag as pd
    assert not pd._DEF_DECL_RE.match("instance foo : LE ℕ where le _ _ := True")
    assert pd._DEF_DECL_RE.match("def foo : ℕ := 0")


def test_reframe_objects_refuse_the_exploit():
    from search.reframe import parse_reframe
    raw = json.dumps({
        "name": "F", "rationale": "r", "objects": [EXPLOIT],
        "bridges": ["lemma b1 : True"], "theorems": [],
        "derivation": "trivial"})
    t, err = parse_reframe(raw)
    assert t is None and "unsafe" in err


def test_library_refuses_the_exploit_without_compiling_it():
    from search.library import build_library

    calls = {"n": 0}

    def llm(system, user):
        if "planning a small, self-contained" in system:
            return json.dumps({"decls": [{"name": "x", "kind": "def",
                                          "statement": "def x : ℕ := 0",
                                          "rationale": "r"}]})
        return json.dumps({"declaration": EXPLOIT})

    def verify(header, body):
        calls["n"] += 1
        return {"ok": True}

    res = build_library("spec", llm_call=llm, verify_fn=verify,
                        attempts_per_decl=1)
    assert res.decls == [], "an unsafe declaration must never be accepted"
    assert calls["n"] == 0, "it must be refused BEFORE reaching the kernel"
