"""Point 7 — causal (probe-isolated) error attribution, flag-gated.

Line-mapped blame is syntactic; with --causal-attribution each blamed
leaf is probe-compiled standalone in its dependency context, and a leaf
that closes on its own is exonerated (blame reroutes to the closer).
Off by default = attribution unchanged.
"""
from __future__ import annotations

import json

from search.proof_dag import attempt_dag_proof, SKETCH_SYSTEM


def _mk(*, causal: bool):
    """h1's tactic is CORRECT (closes standalone) but the assembled-proof
    error is line-mapped onto h1's segment (collateral of the closer).
    The repair LLM echoes back whatever ids it is asked to fix — so the
    final repaired ids reveal WHO got blamed."""
    calls = {"repair_prompts": []}

    def llm(system, user):
        if system == SKETCH_SYSTEM:
            return json.dumps({"haves": [
                {"id": "h1", "type": "0 ≤ (a - b)^2",
                 "tactic": "positivity", "depends": []}],
                "closer": "nlinarith [h1]"})
        calls["repair_prompts"].append(user)
        # Fix the closer (the real culprit) when asked about it.
        return json.dumps({"repairs": [],
                           "closer": "nlinarith [h1, sq_nonneg (a - b)]"})

    def verify(header, body):
        if "sq_nonneg (a - b)" in body:
            return {"ok": True, "errors": None, "body_line_offset": 3}
        # Error on file line 5 → body line 2 → inside h1's segment even
        # though h1 is healthy (collateral blame).
        return {"ok": False,
                "errors": "Try_x.lean:5:4: error: linarith failed",
                "body_line_offset": 3}

    def probe(header, body):
        # Standalone leaf probe: h1's positivity closes fine.
        if "positivity" in body and "theorem" in header:
            return {"ok": True, "errors": None, "body_line_offset": 1}
        return {"ok": False, "errors": "error: x", "body_line_offset": 1}

    res = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=llm, verify_fn=verify, probe_fn=probe,
        sketch_attempts=1, repair_rounds=2, causal_attribution=causal)
    return res, calls


def test_causal_exonerates_healthy_leaf():
    res, calls = _mk(causal=True)
    assert res.verified
    # The repair prompt saw a BROKEN CLOSER, not a broken h1.
    joined = "\n".join(calls["repair_prompts"])
    assert "BROKEN closer" in joined
    assert "id: h1" not in joined
    # h1's original tactic survived untouched.
    h1 = next(h for h in res.sketch.haves if h.id == "h1")
    assert h1.tactic == "positivity"


def test_default_off_keeps_line_blame():
    res, calls = _mk(causal=False)
    # Legacy behaviour: h1 is blamed (line-mapped) and shown as broken.
    joined = "\n".join(calls["repair_prompts"])
    assert "id: h1" in joined


def test_probe_failure_never_exonerates():
    def llm(system, user):
        if system == SKETCH_SYSTEM:
            return json.dumps({"haves": [
                {"id": "h1", "type": "0 ≤ (a - b)^2",
                 "tactic": "positivity", "depends": []}],
                "closer": "nlinarith [h1]"})
        return json.dumps({"repairs": [
            {"id": "h1", "type": "0 ≤ (a - b)^2",
             "tactic": "nlinarith [sq_nonneg (a - b)]"}]})

    def verify(header, body):
        if "sq_nonneg" in body:
            return {"ok": True, "errors": None, "body_line_offset": 3}
        return {"ok": False,
                "errors": "Try_x.lean:5:4: error: positivity failed",
                "body_line_offset": 3}

    def probe(header, body):
        raise RuntimeError("probe fell over")

    res = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=llm, verify_fn=verify, probe_fn=probe,
        sketch_attempts=1, repair_rounds=2, causal_attribution=True)
    # Raising probe → no exoneration → h1 stays blamed and gets repaired.
    assert res.verified
    assert "h1" in res.repaired_ids
