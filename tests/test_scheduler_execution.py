"""Point 5 execution — scheduler-engine repair loop (behind the flag).

The contract: `dag_repair_engine="scheduler"` routes every repair-stage
decision through the RepairPolicy object, and for the DEFAULT policy this
is parity-identical to legacy. These tests run identical scenarios under
both engines and assert the DagResult matches field-for-field, across the
repair paths (leaf-closer, llm-repair, theory abduction). Also checks the
engine validation.
"""
from __future__ import annotations

import json

import pytest

from search.proof_dag import attempt_dag_proof


def _sketch_json(haves, closer):
    return json.dumps({"haves": haves, "closer": closer})


def _run(engine, *, llm, verify, **kw):
    return attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=llm, verify_fn=verify, sketch_attempts=1,
        repair_rounds=2, dag_repair_engine=engine, **kw)


def _fields(r) -> dict:
    """The outcome fields parity must preserve."""
    return {
        "verified": r.verified,
        "failure_stage": r.failure_stage,
        "repair_rounds_used": r.repair_rounds_used,
        "repaired_ids": sorted(r.repaired_ids),
        "leaf_closer_fixed_ids": sorted(r.leaf_closer_fixed_ids),
        "decomposed_ids": sorted(r.decomposed_ids),
        "abduced_ids": sorted(r.abduced_ids),
        "final_tactics": sorted((h.id, h.tactic)
                                for h in (r.sketch.haves if r.sketch else [])),
    }


# ---------------- scenario builders (deterministic fakes) -------------------

def _repair_scenario():
    good = _sketch_json(
        [{"id": "h1", "type": "0 ≤ (a - b)^2", "tactic": "positivity",
          "depends": []},
         {"id": "h2", "type": "a^2 + b^2 ≥ 2*a*b", "tactic": "nlinarith [h1]",
          "depends": ["h1"]}],
        "linarith [h2]")

    def llm(system, user):
        from search.proof_dag import SKETCH_SYSTEM
        if system == SKETCH_SYSTEM:
            return good
        return json.dumps({"repairs": [
            {"id": "h1", "type": "0 ≤ (a - b)^2",
             "tactic": "nlinarith [sq_nonneg (a - b)]"}]})

    def verify(header, body):
        if "nlinarith [sq_nonneg (a - b)]" in body:
            return {"ok": True, "errors": None, "body_line_offset": 3}
        return {"ok": False,
                "errors": "Try_x.lean:5:4: error: positivity failed",
                "body_line_offset": 3}
    return llm, verify


def _leaf_closer_scenario():
    good = _sketch_json(
        [{"id": "h1", "type": "0 ≤ (a - b)^2", "tactic": "positivity",
          "depends": []}],
        "nlinarith [h1]")

    def llm(system, user):
        return good  # only the sketch is ever needed

    def verify(header, body):
        if "simp [sq_nonneg]" in body:
            return {"ok": True, "errors": None, "body_line_offset": 3}
        return {"ok": False,
                "errors": "Try_x.lean:5:4: error: positivity failed",
                "body_line_offset": 3}

    def closer(stmt):
        return "simp [sq_nonneg]"
    return llm, verify, closer


# ---------------- parity across engines -------------------------------------

def test_parity_plain_llm_repair():
    llm, verify = _repair_scenario()
    legacy = _fields(_run("legacy", llm=_repair_scenario()[0],
                          verify=_repair_scenario()[1]))
    sched = _fields(_run("scheduler", llm=_repair_scenario()[0],
                         verify=_repair_scenario()[1]))
    assert legacy == sched
    assert sched["verified"] is True
    assert sched["repaired_ids"] == ["h1"]


def test_parity_leaf_closer():
    l1, v1, c1 = _leaf_closer_scenario()
    l2, v2, c2 = _leaf_closer_scenario()
    legacy = _fields(_run("legacy", llm=l1, verify=v1, leaf_closer_call=c1))
    sched = _fields(_run("scheduler", llm=l2, verify=v2, leaf_closer_call=c2))
    assert legacy == sched
    assert sched["verified"] is True
    assert sched["leaf_closer_fixed_ids"] == ["h1"]


def test_parity_theory_abduction():
    """The theory path (bare-timeout / theory-leaf + closer-stuck triggers)
    must also match across engines."""
    header = "theorem tgt (x : ℝ) (h : 0 < x) : x + x = 2 * x := by sorry"

    def make():
        def llm(system, user):
            from search.proof_dag import SKETCH_SYSTEM, THEORY_SYSTEM
            if system == SKETCH_SYSTEM:
                return json.dumps({
                    "setup": [], "haves": [
                        {"id": "h0", "type": "x + x = 2 * x",
                         "tactic": "nlinarith", "depends": []}],
                    "closer": "exact h0"})
            if system == THEORY_SYSTEM:
                return json.dumps({
                    "defs": [], "lemmas": [
                        {"name": "aux_dbl",
                         "statement": "lemma aux_dbl (y : ℝ) : y + y = 2 * y"}],
                    "leaf_tactics": {"h0": "exact aux_dbl x"}})
            return json.dumps({"proof": "ring"})

        def verify(hdr, body):
            if "aux_" in hdr and "sorry" not in body:
                return {"ok": True, "errors": None}
            return {"ok": False, "errors": "error: unsolved goals",
                    "body_line_offset": 1}

        def probe(hdr, body):
            return {"ok": True, "errors": None} if "sorry" in hdr else \
                   {"ok": False, "errors": "error: unsolved goals",
                    "body_line_offset": 1}
        return llm, verify, probe

    def go(engine):
        llm, verify, probe = make()
        return _fields(attempt_dag_proof(
            header, sketch_llm_call=llm, verify_fn=verify, probe_fn=probe,
            sketch_attempts=1, repair_rounds=1, abduce_lemmas=True,
            abduce_mode="theory", abduce_theory_rounds=0,
            abduce_theory_trigger="always", dag_repair_engine=engine))

    assert go("legacy") == go("scheduler")


def test_unknown_engine_raises():
    llm, verify = _repair_scenario()
    with pytest.raises(ValueError):
        _run("bogus", llm=llm, verify=verify)


def test_legacy_is_default():
    # Not passing the flag == legacy; identical to explicit legacy.
    llm, verify = _repair_scenario()
    r = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=_repair_scenario()[0], verify_fn=_repair_scenario()[1],
        sketch_attempts=1, repair_rounds=2)
    assert _fields(r) == _fields(_run("legacy", llm=_repair_scenario()[0],
                                      verify=_repair_scenario()[1]))
