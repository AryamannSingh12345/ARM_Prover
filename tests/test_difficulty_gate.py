"""Point 4 — difficulty heuristic + decomposition-quality gate.

Covers `search.difficulty` (the surface-syntax difficulty proxy and the
`theory_reduces_difficulty` gate decision) and its wiring into ARM's
theory-abduction loop (flag-gated, off by default).
"""
from __future__ import annotations

from search.difficulty import (
    estimate_difficulty, difficulty_features, theory_reduces_difficulty,
)


# ---------------- the difficulty proxy --------------------------------------

def test_estimate_monotone_in_size():
    small = estimate_difficulty("a = b")
    big = estimate_difficulty("a = b " * 40)
    assert big > small


def test_quantifiers_and_bigops_increase_difficulty():
    plain = estimate_difficulty("x + y = y + x")
    quant = estimate_difficulty("∀ x, ∃ y, x + y = y + x")
    bigop = estimate_difficulty("∑ i in Finset.range n, f i = g n")
    assert quant > plain
    assert bigop > plain


def test_geometric_markers_register():
    f = difficulty_features(
        "Collinear ℝ {a, b, c} ∧ dist a b = dist b c")
    assert f.n_geometric >= 2
    assert f.n_connectives >= 1


def test_provability_prior_reduces_estimate():
    g = "∀ x : ℝ, ∃ y, x + y = 0"
    base = estimate_difficulty(g)
    with_prior = estimate_difficulty(g, provability_prior=0.9)
    assert with_prior < base
    # prior=1.0 (always closes) drives it to ~0.
    assert estimate_difficulty(g, provability_prior=1.0) == 0.0


# ---------------- the gate decision -----------------------------------------

_GOAL = ("∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → "
         "x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + "
         "z / Real.sqrt (x + y) ≥ 3 / Real.sqrt 2")


def test_gate_rejects_near_restatement():
    """A lemma whose conclusion is (essentially) the whole goal is nearly
    as hard as the goal → gate fails."""
    r = theory_reduces_difficulty(
        [_GOAL],
        [("aux_restate", _GOAL)],
        factor=0.9)
    assert r.ok is False
    assert "aux_restate" in r.offenders


def test_gate_passes_genuine_decomposition():
    """Small, structurally simpler lemmas → gate passes."""
    r = theory_reduces_difficulty(
        [_GOAL],
        [("aux_tangent", "Real.sqrt (y + z) ≤ (y + z + 2) / 2"),
         ("aux_pos", "0 < Real.sqrt 2")],
        factor=0.9)
    assert r.ok is True
    assert r.offenders == []


def test_gate_no_lemmas_passes():
    r = theory_reduces_difficulty([_GOAL], [], factor=0.9)
    assert r.ok is True


def test_gate_abstains_on_trivial_goal():
    # D(goal) ~ 0 → cannot assess a reduction → abstain (pass).
    r = theory_reduces_difficulty(["", "  "], [("l", "x")], factor=0.9)
    assert r.ok is True


def test_stricter_factor_rejects_more():
    goal = ["∀ x : ℝ, P x ∧ Q x → R x"]
    lemma = [("aux", "∀ x : ℝ, P x → R x")]
    lenient = theory_reduces_difficulty(goal, lemma, factor=0.95)
    strict = theory_reduces_difficulty(goal, lemma, factor=0.5)
    # A lower factor is at least as strict (never rejects fewer).
    assert not (lenient.ok is False and strict.ok is True)


# ---------------- wiring into ARM (flag-gated) ------------------------------

def _run_theory(monkeypatch, *, quality_gate: bool, factor: float = 0.9):
    """Drive attempt_dag_proof so the theory loop proposes ONE lemma that
    restates the goal, and capture whether the quality gate rejected it.
    Returns (repair_errors, proposed_count)."""
    import json
    from search import proof_dag

    header = ("theorem tgt (x : ℝ) (h : 0 < x) : "
              "x + x = 2 * x := by sorry")

    calls = {"n": 0}

    def fake_sketch(system, user):
        calls["n"] += 1
        if system == proof_dag.SKETCH_SYSTEM:
            # One have with a plausible (non-forbidden) tactic that our
            # fake verifier reports as failing → a stuck leaf.
            return json.dumps({
                "setup": [],
                "haves": [{"id": "h0", "type": "x + x = 2 * x",
                           "tactic": "nlinarith", "depends": []}],
                "closer": "exact h0",
            })
        if system == proof_dag.THEORY_SYSTEM:
            # Propose a lemma that is NON-circular (conclusion differs from
            # the goal text, so the circularity guard passes) but is just
            # as hard as the obligation — a reordered restatement. This is
            # exactly the complexity-preservation case the quality gate is
            # meant to catch that circularity misses.
            return json.dumps({
                "defs": [],
                "lemmas": [{"name": "aux_reorder",
                            "statement": "lemma aux_reorder (x : ℝ) "
                                         "(h : 0 < x) : 2 * x = x + x"}],
                "leaf_tactics": {"h0": "linarith [aux_reorder x h]"},
            })
        # prove-lemma / repair: never reached in this test.
        return json.dumps({"proof": "ring"})

    # verify: the have body 'sorry' → the leaf is broken; nothing verifies.
    def fake_verify(hdr, body):
        return {"ok": False, "errors": "error: unsolved goals",
                "body_line_offset": 1}

    def fake_probe(hdr, body):
        return {"ok": False, "errors": "error: unsolved goals",
                "body_line_offset": 1}

    res = proof_dag.attempt_dag_proof(
        header,
        sketch_llm_call=fake_sketch,
        verify_fn=fake_verify,
        probe_fn=fake_probe,
        sketch_attempts=1,
        repair_rounds=1,
        abduce_lemmas=True,
        abduce_mode="theory",
        abduce_theory_rounds=0,
        abduce_theory_trigger="always",
        abduce_quality_gate=quality_gate,
        abduce_quality_factor=factor,
    )
    return res


def test_quality_gate_fires_when_enabled(monkeypatch):
    res = _run_theory(monkeypatch, quality_gate=True)
    assert any(e.startswith("theory_low_quality") for e in res.repair_errors)


def test_quality_gate_silent_when_disabled(monkeypatch):
    res = _run_theory(monkeypatch, quality_gate=False)
    assert not any(e.startswith("theory_low_quality")
                   for e in res.repair_errors)


# ---------------- provability prior in the gate (Point 4) -------------------

def test_prior_fn_lowers_lemma_difficulty_to_pass():
    """A lemma that would fail the gate passes when the prior says its
    shape usually closes (prior applied to the lemma side)."""
    goal = ["∀ x : ℝ, P x ∧ Q x → R x"]
    lemma = [("aux", "∀ x : ℝ, P x ∧ Q x → R x ∨ R x")]
    base = theory_reduces_difficulty(goal, lemma, factor=0.9)
    assert base.ok is False  # lemma ≈ goal difficulty

    def prior(g):
        # High closure confidence for the lemma's shape only.
        return 0.9 if "∨" in g else None

    with_prior = theory_reduces_difficulty(goal, lemma, factor=0.9,
                                           prior_fn=prior)
    assert with_prior.ok is True


def test_prior_fn_none_and_raising_are_soft():
    goal = ["∀ x : ℝ, P x → R x"]
    lemma = [("aux", "P 0")]
    r_none = theory_reduces_difficulty(goal, lemma, factor=0.9,
                                       prior_fn=lambda g: None)
    r_boom = theory_reduces_difficulty(
        goal, lemma, factor=0.9,
        prior_fn=lambda g: (_ for _ in ()).throw(RuntimeError()))
    r_off = theory_reduces_difficulty(goal, lemma, factor=0.9)
    assert r_none.ok == r_boom.ok == r_off.ok
    assert r_none.lemma_difficulty == r_off.lemma_difficulty


# ---------------- quality gate prunes offenders (not wholesale reject) -------

def test_quality_gate_prunes_offender_keeps_good_lemmas():
    """amc12a_2021_p25 regression: the gate rejected the WHOLE proposal
    when it bundled one goal-sized capstone with good building blocks.
    Now it drops only the offender and keeps the rest."""
    import json
    from search import proof_dag
    header = "theorem tgt (x : ℝ) (h : 0 < x) : x + x = 2 * x := by sorry"
    evs = []

    def fake_sketch(system, user):
        if system == proof_dag.SKETCH_SYSTEM:
            return json.dumps({"setup": [], "haves": [
                {"id": "h0", "type": "x + x = 2 * x",
                 "tactic": "nlinarith", "depends": []}],
                "closer": "exact h0"})
        if system == proof_dag.THEORY_SYSTEM:
            return json.dumps({"defs": [], "lemmas": [
                {"name": "aux_big",
                 "statement": "lemma aux_big (x : ℝ) (h : 0 < x) : "
                              "2 * x = x + x"},          # goal-sized offender
                {"name": "aux_small",
                 "statement": "lemma aux_small : True"}],  # tiny, good
                "leaf_tactics": {"h0": "linarith [aux_big x h]"}})
        return json.dumps({"proof": "ring"})

    def fail(hdr, body):
        return {"ok": False, "errors": "error: unsolved goals",
                "body_line_offset": 1}

    proof_dag.attempt_dag_proof(
        header, sketch_llm_call=fake_sketch, verify_fn=fail, probe_fn=fail,
        sketch_attempts=1, repair_rounds=1, abduce_lemmas=True,
        abduce_mode="theory", abduce_theory_rounds=0,
        abduce_theory_trigger="always", abduce_quality_gate=True,
        abduce_quality_factor=0.9,
        trace=lambda k, **p: evs.append((k, p.get("stage"), p)))

    pruned = [p for (k, s, p) in evs if s == "pruned (low quality)"]
    rejected = [p for (k, s, p) in evs
                if s == "rejected (all lemmas low quality)"]
    assert pruned and not rejected, "should prune offender, not reject all"
    assert "aux_big" in pruned[0].get("dropped", [])
    assert "aux_small" in pruned[0].get("kept", [])
