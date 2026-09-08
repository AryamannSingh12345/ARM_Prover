"""Phase 1 — first-class proof-obligation model.

Contracts pinned here:
- ObligationState / ObligationAttempt / CandidateMove serialize round-trip.
- Obligations are constructed from a Sketch: one per have (source_kind
  'leaf'), plus a 'setup' obligation and a 'closer' pseudo-obligation
  that uses the SAME abstraction as ordinary leaves.
- The HaveNode/Sketch JSON contract is unchanged (parse_sketch still
  works; obligation construction is additive).
- Recording attempts accumulates history but drops an identical repeat
  on an unchanged obligation.
- A verified obligation stays verified when an unrelated obligation is
  repaired.
- The reserved ids match proof_dag's (no divergence during the split).

Pure Python — no Lean, no LLM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.dag.model import (  # noqa: E402
    ObligationState, ObligationAttempt, CandidateMove,
    SETUP_ID, CLOSER_ID,
)
from search.dag.obligations import (  # noqa: E402
    build_obligations_from_sketch, record_attempt, mark_verified,
    classify_failure,
)
from search.proof_dag import (  # noqa: E402
    parse_sketch,
    SETUP_ID as PD_SETUP_ID,
    CLOSER_ID as PD_CLOSER_ID,
)

SKETCH_JSON = json.dumps({
    "setup": ["classical"],
    "haves": [
        {"id": "h1", "type": "0 < a", "tactic": "positivity", "depends": []},
        {"id": "h2", "type": "a < b", "tactic": "linarith",
         "depends": ["h1"]},
    ],
    "closer": "exact lt_trans h1 h2",
})
MAIN_GOAL = "0 < b"


# ---------- reserved ids stay in sync -----------------------------------------

def test_reserved_ids_match_proof_dag():
    assert SETUP_ID == PD_SETUP_ID
    assert CLOSER_ID == PD_CLOSER_ID


# ---------- serialization -----------------------------------------------------

def test_attempt_roundtrip():
    a = ObligationAttempt(
        solver="fallback_ladder", candidate="nlinarith", result="failed",
        error="linarith failed", wall_s=1.5,
        token_usage={"in": 10, "out": 5}, verifier_backend="compile")
    assert ObligationAttempt.from_dict(a.to_dict()) == a


def test_candidate_move_roundtrip():
    c = CandidateMove(source="proof_prior", tactic="simp", cost=0.3,
                      metadata={"tactic_class": "simp"})
    assert CandidateMove.from_dict(c.to_dict()) == c


def test_obligation_roundtrip():
    ob = ObligationState(
        id="h1", goal_text="0 < a", dependencies=("h0",),
        local_context=None, source_kind="leaf", status="pending")
    ob.candidate_moves.append(
        CandidateMove(source="template", tactic="positivity", cost=0.1))
    record_attempt(ob, ObligationAttempt(
        solver="fallback_ladder", candidate="positivity", result="failed",
        error="e", wall_s=0.2, token_usage=None, verifier_backend="compile"))
    back = ObligationState.from_dict(ob.to_dict())
    assert back == ob
    # survives a JSON text round-trip too
    assert ObligationState.from_dict(json.loads(json.dumps(ob.to_dict()))) == ob


# ---------- construction from a sketch ----------------------------------------

def test_build_from_sketch_covers_leaves_setup_closer():
    sketch, err = parse_sketch(SKETCH_JSON)
    assert err is None and sketch is not None
    obs = build_obligations_from_sketch(sketch, main_goal=MAIN_GOAL)
    assert set(obs) == {"h1", "h2", SETUP_ID, CLOSER_ID}
    assert obs["h1"].source_kind == "leaf"
    assert obs["h2"].dependencies == ("h1",)
    assert obs[SETUP_ID].source_kind == "setup"
    # the closer is tracked like a leaf: source_kind 'closer', goal = main
    # goal, and it depends on every have.
    assert obs[CLOSER_ID].source_kind == "closer"
    assert obs[CLOSER_ID].goal_text == MAIN_GOAL
    assert set(obs[CLOSER_ID].dependencies) == {"h1", "h2"}
    assert all(o.status == "pending" for o in obs.values())


def test_haveNode_contract_unchanged():
    # Building obligations must not mutate the sketch (additive only).
    sketch, _ = parse_sketch(SKETCH_JSON)
    before = [(h.id, h.type_text, h.tactic, tuple(h.depends))
              for h in sketch.haves]
    build_obligations_from_sketch(sketch, main_goal=MAIN_GOAL)
    after = [(h.id, h.type_text, h.tactic, tuple(h.depends))
             for h in sketch.haves]
    assert before == after


# ---------- attempt history + dedup -------------------------------------------

def _attempt(cand, result="failed", solver="llm_repair"):
    return ObligationAttempt(
        solver=solver, candidate=cand, result=result, error="e",
        wall_s=0.1, token_usage=None, verifier_backend="compile")


def test_identical_attempt_deduped():
    ob = ObligationState(id="h1", goal_text="g", dependencies=(),
                         local_context=None, source_kind="leaf",
                         status="pending")
    assert record_attempt(ob, _attempt("nlinarith")) is True
    # identical solver+candidate+result on an unchanged obligation: dropped
    assert record_attempt(ob, _attempt("nlinarith")) is False
    # a different candidate is kept
    assert record_attempt(ob, _attempt("linarith")) is True
    assert len(ob.attempts) == 2


def test_verified_obligation_survives_unrelated_repair():
    obs = build_obligations_from_sketch(
        parse_sketch(SKETCH_JSON)[0], main_goal=MAIN_GOAL)
    mark_verified(obs["h1"], "positivity")
    assert obs["h1"].status == "verified"
    assert obs["h1"].verified_tactic == "positivity"
    # repairing h2 must not touch h1
    record_attempt(obs["h2"], _attempt("nlinarith"))
    mark_verified(obs["h2"], "nlinarith")
    assert obs["h1"].status == "verified"
    assert obs["h1"].verified_tactic == "positivity"


# ---------- failure classification --------------------------------------------

def test_classify_failure_extracts_classes_and_unknowns():
    err = ("f.lean:5:2: error(lean.unknownIdentifier): Unknown identifier "
           "`mem_segment_iff_wbtw`\n")
    classes, unknowns = classify_failure(err)
    assert "unknown_identifier" in classes
    assert "mem_segment_iff_wbtw" in unknowns


def test_classify_failure_timeout_and_unsolved():
    ct, _ = classify_failure("timeout after 600s")
    assert "timeout" in ct
    cu, _ = classify_failure("f.lean:5:2: error: unsolved goals\n⊢ True")
    assert "unsolved_goals" in cu
