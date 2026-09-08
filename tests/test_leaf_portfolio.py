"""Point 3 — deterministic leaf-candidate portfolio executing in DAG leaves.

The portfolio pass (Pass 1.25) fires only when `leaf_solvers` is non-empty:
candidates from the providers are combined into one `first | …` tactic on
the broken leaf so a single assembled-proof compile tests them all. One
shot per leaf; failures fall through to the normal ladder. Default (no
solvers) is byte-identical to legacy.
"""
from __future__ import annotations

import json

from search.proof_dag import attempt_dag_proof, SKETCH_SYSTEM
from search.dag.model import CandidateMove
from search.dag.solvers import SolverContext


class _FakeProvider:
    def __init__(self, name, moves):
        self.name = name
        self._moves = moves
        self.calls = 0

    def candidates(self, obligation, context):
        self.calls += 1
        return list(self._moves)


def _sketch():
    return json.dumps({"haves": [
        {"id": "h1", "type": "0 ≤ (a - b)^2", "tactic": "positivity",
         "depends": []}], "closer": "nlinarith [h1]"})


def _mk(llm_repair_response=None):
    calls = {"repair": 0}

    def llm(system, user):
        if system == SKETCH_SYSTEM:
            return _sketch()
        calls["repair"] += 1
        return llm_repair_response or json.dumps({"repairs": []})

    def verify(header, body):
        if "sq_nonneg" in body:
            return {"ok": True, "errors": None, "body_line_offset": 3}
        return {"ok": False,
                "errors": "Try_x.lean:5:4: error: positivity failed",
                "body_line_offset": 3}
    return llm, verify, calls


def test_portfolio_candidate_solves_leaf():
    llm, verify, calls = _mk()
    prov = _FakeProvider("proof_prior", [
        CandidateMove(source="proof_prior", tactic="norm_num", cost=0.9),
        CandidateMove(source="proof_prior",
                      tactic="nlinarith [sq_nonneg (a - b)]", cost=0.5),
    ])
    r = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=llm, verify_fn=verify, sketch_attempts=1,
        repair_rounds=2, leaf_solvers=[prov],
        solver_context=SolverContext(theorem_header="t", premises=()))
    assert r.verified
    assert prov.calls == 1
    # Cheapest-first inside the first| combination.
    h1 = next(h for h in r.sketch.haves if h.id == "h1")
    assert h1.tactic.startswith("first | nlinarith [sq_nonneg (a - b)]")
    # Solved by the portfolio, not by LLM repair.
    assert calls["repair"] == 0
    assert "h1" in r.repaired_ids


def test_no_solvers_is_legacy():
    llm, verify, calls = _mk(json.dumps({"repairs": [
        {"id": "h1", "type": "0 ≤ (a - b)^2",
         "tactic": "nlinarith [sq_nonneg (a - b)]"}]}))
    r = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=llm, verify_fn=verify, sketch_attempts=1,
        repair_rounds=2)
    assert r.verified
    assert calls["repair"] == 1  # LLM repair did the work, as before


def test_forbidden_and_malformed_candidates_filtered():
    llm, verify, calls = _mk(json.dumps({"repairs": [
        {"id": "h1", "type": "0 ≤ (a - b)^2",
         "tactic": "nlinarith [sq_nonneg (a - b)]"}]}))
    prov = _FakeProvider("template_tactics", [
        CandidateMove(source="t", tactic="sorry", cost=0.1),
        CandidateMove(source="t", tactic="exact ?_", cost=0.2),
        CandidateMove(source="t", tactic="line1\nline2", cost=0.3),
        CandidateMove(source="t", tactic="", cost=0.4),
    ])
    r = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=llm, verify_fn=verify, sketch_attempts=1,
        repair_rounds=2, leaf_solvers=[prov],
        solver_context=SolverContext(theorem_header="t", premises=()))
    # Every candidate was rejected → leaf fell through to LLM repair.
    assert r.verified
    assert calls["repair"] == 1


def test_portfolio_one_shot_per_leaf():
    llm, verify, calls = _mk(json.dumps({"repairs": [
        {"id": "h1", "type": "0 ≤ (a - b)^2",
         "tactic": "nlinarith [sq_nonneg (a - b)]"}]}))
    prov = _FakeProvider("dependency_exploration", [
        CandidateMove(source="d", tactic="norm_num", cost=0.5)])  # fails
    r = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=llm, verify_fn=verify, sketch_attempts=1,
        repair_rounds=3, leaf_solvers=[prov],
        solver_context=SolverContext(theorem_header="t", premises=()))
    assert r.verified
    assert prov.calls == 1          # not retried on the same leaf
    assert calls["repair"] >= 1     # LLM repair finished the job


def test_provider_exception_is_soft():
    class _Boom:
        name = "boom"

        def candidates(self, ob, ctx):
            raise RuntimeError("nope")

    llm, verify, calls = _mk(json.dumps({"repairs": [
        {"id": "h1", "type": "0 ≤ (a - b)^2",
         "tactic": "nlinarith [sq_nonneg (a - b)]"}]}))
    r = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=llm, verify_fn=verify, sketch_attempts=1,
        repair_rounds=2, leaf_solvers=[_Boom()],
        solver_context=SolverContext(theorem_header="t", premises=()))
    assert r.verified  # ladder still finished the proof


def test_garbage_candidate_does_not_consume_leaf():
    """Bug-2 regression (ablate_easy_portfolioON): a portfolio candidate
    that does NOT close the leaf standalone must be discarded, leaving the
    leaf broken for the LLM repair — never consumed with a failing tactic."""
    calls = {"repair": 0, "probe": 0}

    def llm(system, user):
        if system == SKETCH_SYSTEM:
            return _sketch()
        calls["repair"] += 1
        return json.dumps({"repairs": [
            {"id": "h1", "type": "0 ≤ (a - b)^2",
             "tactic": "nlinarith [sq_nonneg (a - b)]"}]})

    def verify(header, body):
        if "sq_nonneg (a - b)" in body:
            return {"ok": True, "errors": None, "body_line_offset": 3}
        return {"ok": False,
                "errors": "Try_x.lean:5:4: error: unknown tactic",
                "body_line_offset": 3}

    def probe(header, body):
        # Standalone leaf probe: the garbage candidate never closes it.
        calls["probe"] += 1
        return {"ok": False, "errors": "error: unknown tactic",
                "body_line_offset": 1}

    # Provider emits a plausible-looking but wrong candidate (mirrors the
    # topology-noise `rw [...]` that broke the real run).
    prov = _FakeProvider("template_tactics", [
        CandidateMove(source="template_tactics",
                      tactic="rw [geometric_hahn_banach_open_open]", cost=0.3)])
    r = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=llm, verify_fn=verify, probe_fn=probe,
        sketch_attempts=1, repair_rounds=2, leaf_solvers=[prov],
        solver_context=SolverContext(theorem_header="t", premises=()))
    assert calls["probe"] >= 1          # portfolio DID probe the candidate
    assert r.verified                   # ...but the leaf was still repaired
    assert calls["repair"] >= 1         # by the LLM, not consumed by garbage
    h1 = next(h for h in r.sketch.haves if h.id == "h1")
    assert "geometric_hahn_banach" not in h1.tactic   # garbage discarded
