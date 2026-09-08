"""Point 7a — α-invariant circularity fingerprint + defeq-probe seam.

Covers `search.goal_fingerprint` (text vs syntactic keys, alpha
invariance, the elaborated defeq escalation) and the wiring into ARM's
circularity guard (mode='text' reproduces legacy behaviour; 'syntactic'
catches renamed-binder restatements that 'text' misses).
"""
from __future__ import annotations

from search.goal_fingerprint import (
    conclusion_fingerprint, text_fingerprint, is_circular, statement_to_prop,
)
from search.proof_dag import split_theorem_header as SPLIT


def _fp_syn(s):
    return conclusion_fingerprint(s, split_fn=SPLIT)


def _fp_txt(s):
    return text_fingerprint(s, split_fn=SPLIT)


# ---------------- fingerprint semantics -------------------------------------

def test_text_key_matches_norm_ws_of_conclusion():
    from search.proof_dag import _norm_ws
    stmt = "theorem t (x : ℝ) : x + x = 2 * x"
    assert _fp_txt(stmt) == _norm_ws("x + x = 2 * x")


def test_syntactic_is_alpha_invariant():
    a = "lemma f (x : ℝ) : x + x = 2 * x"
    b = "lemma g (a : ℝ) : a + a = 2 * a"
    # Different variable names → text keys differ, α-invariant keys match.
    assert _fp_txt(a) != _fp_txt(b)
    assert _fp_syn(a) == _fp_syn(b)


def test_syntactic_forall_binder_alpha_invariant():
    a = "lemma f : ∀ x : ℝ, x = x"
    b = "lemma g : ∀ y : ℝ, y = y"
    assert _fp_syn(a) == _fp_syn(b)


def test_syntactic_distinguishes_real_difference():
    a = "lemma f (x : ℝ) : x + x = 2 * x"
    b = "lemma g (x : ℝ) : x * x = x ^ 2"
    assert _fp_syn(a) != _fp_syn(b)


def test_free_names_are_not_renamed():
    # `Real.pi` is a constant, not a binder — must survive fingerprinting
    # and keep two genuinely different goals distinct.
    a = "lemma f (x : ℝ) : x = Real.pi"
    b = "lemma g (x : ℝ) : x = Real.exp 1"
    assert _fp_syn(a) != _fp_syn(b)


# ---------------- is_circular ------------------------------------------------

def test_is_circular_text_misses_renamed_binder():
    goal = "theorem t (x : ℝ) : x + x = 2 * x"
    lemma = "lemma aux (a : ℝ) : a + a = 2 * a"
    assert is_circular(lemma, [goal], mode="text", split_fn=SPLIT) is False
    assert is_circular(lemma, [goal], mode="syntactic",
                       split_fn=SPLIT) is True


def test_is_circular_text_still_catches_exact():
    goal = "theorem t (x : ℝ) : x + x = 2 * x"
    lemma = "lemma aux (x : ℝ) : x + x = 2 * x"
    assert is_circular(lemma, [goal], mode="text", split_fn=SPLIT) is True


def test_is_circular_unsplittable_never_matches():
    # A statement with no top-level colon does not split → never circular
    # (legacy-faithful; no false positive).
    assert is_circular("garbage no colon", ["also garbage"],
                       mode="syntactic", split_fn=SPLIT) is False


def test_is_circular_genuine_lemma_not_flagged():
    goal = "theorem t (x : ℝ) : x + x = 2 * x"
    lemma = "lemma aux (x : ℝ) : 0 < x → x < 2 * x"
    assert is_circular(lemma, [goal], mode="syntactic",
                       split_fn=SPLIT) is False


# ---------------- elaborated defeq escalation (callback) --------------------

def test_elaborated_escalation_uses_probe():
    goal = "theorem t (x : ℝ) : x + x = 2 * x"
    # Syntactically distinct, but the (fake) defeq probe declares them
    # definitionally equal → circular only in elaborated mode.
    lemma = "lemma aux (x : ℝ) : 2 * x = x + x"
    assert is_circular(lemma, [goal], mode="syntactic",
                       split_fn=SPLIT) is False

    def probe(a, b):
        return True  # pretend Lean says defeq

    assert is_circular(lemma, [goal], mode="elaborated",
                       split_fn=SPLIT, defeq_probe=probe) is True


def test_elaborated_none_probe_is_safe():
    goal = "theorem t (x : ℝ) : x + x = 2 * x"
    lemma = "lemma aux (x : ℝ) : 2 * x = x + x"

    def probe(a, b):
        return None  # undecidable → must be treated as NOT circular

    assert is_circular(lemma, [goal], mode="elaborated",
                       split_fn=SPLIT, defeq_probe=probe) is False


def test_elaborated_probe_exception_is_safe():
    goal = "theorem t (x : ℝ) : x + x = 2 * x"
    lemma = "lemma aux (x : ℝ) : 2 * x = x + x"

    def probe(a, b):
        raise RuntimeError("lean fell over")

    assert is_circular(lemma, [goal], mode="elaborated",
                       split_fn=SPLIT, defeq_probe=probe) is False


# ---------------- statement_to_prop (elaborated-probe input) ----------------

def test_statement_to_prop_telescopes_binders():
    p = statement_to_prop("theorem foo (x : R) (h : 0 < x) : x + x = 2 * x",
                          split_fn=SPLIT)
    assert p == "∀ (x : R) (h : 0 < x), x + x = 2 * x"


def test_statement_to_prop_no_binders():
    assert statement_to_prop("lemma bar : True", split_fn=SPLIT) == "True"


def test_statement_to_prop_implicit_and_instance_binders():
    p = statement_to_prop("theorem t {a : N} [Foo a] : a = a", split_fn=SPLIT)
    assert p == "∀ {a : N} [Foo a], a = a"


def test_statement_to_prop_unsplittable_is_none():
    assert statement_to_prop("no colon here", split_fn=SPLIT) is None


# NOTE: the elaborated-mode Lean defeq probe (`example : Pₐ ↔ P_b :=
# Iff.rfl`) was validated on a real `lake env lean` compile (core-Nat, no
# Mathlib): an α-renamed pair compiles (defeq → circular) and a reordered
# pair fails with an Iff.rfl type mismatch (not defeq → not circular). That
# check is intentionally NOT in this fast suite (it spawns Lean); the
# escalation LOGIC is covered by test_elaborated_escalation_uses_probe with
# a fake probe.


# ---------------- wiring into ARM -------------------------------------------

def _run(circularity_mode: str):
    """Drive the theory loop with a lemma that restates the goal only up
    to a binder rename. text mode must MISS it (theory proceeds);
    syntactic mode must FLAG it circular."""
    import json
    from search import proof_dag

    header = ("theorem tgt (x : ℝ) (h : 0 < x) : x + x = 2 * x := by sorry")

    def fake_sketch(system, user):
        if system == proof_dag.SKETCH_SYSTEM:
            return json.dumps({
                "setup": [],
                "haves": [{"id": "h0", "type": "x + x = 2 * x",
                           "tactic": "nlinarith", "depends": []}],
                "closer": "exact h0"})
        if system == proof_dag.THEORY_SYSTEM:
            # Restates the h0 goal with a renamed binder (a for x).
            return json.dumps({
                "defs": [],
                "lemmas": [{"name": "aux_ren",
                            "statement": "lemma aux_ren (a : ℝ) "
                                         "(h : 0 < a) : a + a = 2 * a"}],
                "leaf_tactics": {"h0": "exact aux_ren x h"}})
        return json.dumps({"proof": "ring"})

    def fail(hdr, body):
        return {"ok": False, "errors": "error: unsolved goals",
                "body_line_offset": 1}

    return proof_dag.attempt_dag_proof(
        header, sketch_llm_call=fake_sketch, verify_fn=fail, probe_fn=fail,
        sketch_attempts=1, repair_rounds=1, abduce_lemmas=True,
        abduce_mode="theory", abduce_theory_rounds=0,
        abduce_theory_trigger="always", circularity_mode=circularity_mode)


def test_wiring_text_mode_misses_renamed_restatement():
    res = _run("text")
    assert not any(e.startswith("theory_circular") for e in res.repair_errors)


def test_wiring_syntactic_mode_flags_renamed_restatement():
    res = _run("syntactic")
    assert any(e.startswith("theory_circular") for e in res.repair_errors)


# ---------------- elaborated probe similarity gate (Point 7a cost fix) -------

def test_elaborated_skips_probe_for_dissimilar_goal():
    """The Lean defeq probe must NOT fire on obviously-unrelated pairs
    (amc12a_2021_p25 ran ~200 pointless compiles). Low token overlap →
    skip the compile entirely."""
    goal = "theorem t (x : ℝ) : x + x = 2 * x"
    lemma = "lemma aux (a b c : ℕ) : Nat.gcd a b ∣ c"   # unrelated vocabulary
    calls = []

    def probe(a, b):
        calls.append((a, b))
        return True  # would report circular IF it were ever called

    assert is_circular(lemma, [goal], mode="elaborated", split_fn=SPLIT,
                       defeq_probe=probe) is False
    assert calls == []  # probe was skipped by the similarity gate


def test_elaborated_probes_similar_goal():
    """A textually-similar-but-not-syntactically-equal lemma DOES get the
    probe (this is exactly the notation/reducible case the mode is for)."""
    goal = "theorem t (x : ℝ) : x + x = 2 * x"
    lemma = "lemma aux (x : ℝ) : x + x = 2 * x + 0"  # same tokens, +0
    calls = []

    def probe(a, b):
        calls.append((a, b))
        return True

    assert is_circular(lemma, [goal], mode="elaborated", split_fn=SPLIT,
                       defeq_probe=probe) is True
    assert len(calls) == 1
