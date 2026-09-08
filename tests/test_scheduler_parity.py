"""Migration parity — the shadow scheduler must reproduce the EXACT
as-built repair order.

Two layers:
1. `plan_round` unit tests: the encoded order matches the legacy gating,
   stage by stage.
2. live shadow demo: run the REAL legacy `attempt_dag_proof` on mocked
   scenarios via an EventRecorder, reconstruct each round's context, and
   assert the scheduler diverges nowhere (compare_round is None).

Pure Python except the live-legacy runs, which mock Lean and the LLM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.dag.scheduler import (  # noqa: E402
    RoundContext, plan_round, compare_round, RepairEngine, Divergence,
)
from search.dag.events import (  # noqa: E402
    EventRecorder,
    STAGE_BARE_TIMEOUT_THEORY, STAGE_LEAF_CLOSER, STAGE_THEORY_LEAF,
    STAGE_THEORY_CLOSER, STAGE_EAGER_ABDUCE, STAGE_DECOMPOSE,
    STAGE_LLM_REPAIR,
)
from search.dag.model import CLOSER_ID
from search.proof_dag import attempt_dag_proof, THEORY_SYSTEM, PROVE_LEMMA_SYSTEM  # noqa: E402


# ---------- plan_round: exact as-built order ----------------------------------

def _ctx(**kw):
    base = dict(has_attributable_errors=True, setup_broken=False,
               closer_broken=False, broken_leaf_ids=())
    base.update(kw)
    return RoundContext(**base)


def test_plan_plain_leaf_is_llm_repair_only():
    assert plan_round(_ctx(broken_leaf_ids=("h1",))) == [STAGE_LLM_REPAIR]


def test_plan_leaf_closer_precedes_llm():
    p = plan_round(_ctx(broken_leaf_ids=("h1",), has_leaf_closer=True))
    assert p == [STAGE_LEAF_CLOSER, STAGE_LLM_REPAIR]


def test_plan_theory_leaf_when_stuck():
    p = plan_round(_ctx(broken_leaf_ids=("h1",), abduce_lemmas=True,
                        abduce_mode="theory", streaks={"h1": 2}))
    assert p == [STAGE_THEORY_LEAF, STAGE_LLM_REPAIR]


def test_plan_theory_leaf_not_fired_below_streak():
    p = plan_round(_ctx(broken_leaf_ids=("h1",), abduce_lemmas=True,
                        abduce_mode="theory", streaks={"h1": 1}))
    assert p == [STAGE_LLM_REPAIR]


def test_plan_theory_leaf_fires_on_always_trigger():
    p = plan_round(_ctx(broken_leaf_ids=("h1",), abduce_lemmas=True,
                        abduce_mode="theory", trigger="always"))
    assert STAGE_THEORY_LEAF in p


def test_plan_theory_closer_stuck():
    p = plan_round(_ctx(broken_leaf_ids=(), closer_broken=True,
                        abduce_lemmas=True, abduce_mode="theory",
                        has_probe=True, streaks={CLOSER_ID: 2}))
    assert p == [STAGE_THEORY_CLOSER, STAGE_LLM_REPAIR]


def test_plan_eager_abduce():
    p = plan_round(_ctx(broken_leaf_ids=("h1",), abduce_lemmas=True,
                        abduce_mode="eager"))
    assert p == [STAGE_EAGER_ABDUCE, STAGE_LLM_REPAIR]


def test_plan_decompose_when_streak_and_depth():
    p = plan_round(_ctx(broken_leaf_ids=("h1",), decompose_depth=1,
                        streaks={"h1": 2}))
    assert p == [STAGE_DECOMPOSE, STAGE_LLM_REPAIR]


def test_plan_setup_only_is_llm_repair():
    assert plan_round(_ctx(broken_leaf_ids=(), setup_broken=True)) == \
        [STAGE_LLM_REPAIR]


def test_plan_bare_timeout_theory():
    p = plan_round(_ctx(has_attributable_errors=False, broken_leaf_ids=(),
                        abduce_lemmas=True, abduce_mode="theory",
                        has_probe=True))
    assert p == [STAGE_BARE_TIMEOUT_THEORY]


def test_plan_bare_timeout_no_theory_breaks():
    assert plan_round(_ctx(has_attributable_errors=False,
                           broken_leaf_ids=())) == []


def test_plan_full_stack_order():
    # leaf closer + theory-leaf + decompose + llm, in that order
    p = plan_round(_ctx(broken_leaf_ids=("h1",), has_leaf_closer=True,
                        abduce_lemmas=True, abduce_mode="theory",
                        decompose_depth=1, streaks={"h1": 2}))
    assert p == [STAGE_LEAF_CLOSER, STAGE_THEORY_LEAF, STAGE_DECOMPOSE,
                 STAGE_LLM_REPAIR]


# ---------- engine modes ------------------------------------------------------

def test_scheduler_mode_is_gated():
    RepairEngine("legacy")           # ok
    RepairEngine("shadow")           # ok
    try:
        RepairEngine("scheduler")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("scheduler mode must be gated behind review")


def test_shadow_records_divergence_only_in_shadow():
    ctx = _ctx(broken_leaf_ids=("h1",))
    legacy = RepairEngine("legacy")
    legacy.observe(1, [STAGE_THEORY_LEAF], ctx)  # wrong, but legacy ignores
    assert legacy.divergences == []
    shadow = RepairEngine("shadow")
    shadow.observe(1, [STAGE_THEORY_LEAF], ctx)  # != plan [LLM_REPAIR]
    assert len(shadow.divergences) == 1


# ---------- live shadow demo: legacy behaviour, zero divergence ---------------

HEADER = "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b"


def _run_legacy(llm, verify, *, probe=None, **kw):
    rec = EventRecorder()
    res = attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify, probe_fn=probe,
        sketch_attempts=1, repair_rounds=3, leaf_fallbacks=(),
        decompose_depth=0, trace=rec, **kw)
    return res, rec


def _static_config(**over):
    cfg = dict(abduce_lemmas=False, abduce_mode=None, has_leaf_closer=False,
               has_probe=False, decompose_depth=0, trigger="stuck",
               has_haves=True, header_splittable=True)
    cfg.update(over)
    return cfg


def _ctx_from_round(rd, cfg, theory_spent):
    return RoundContext(
        has_attributable_errors=bool(rd.broken) or rd.closer_broken,
        setup_broken=rd.setup_broken, closer_broken=rd.closer_broken,
        broken_leaf_ids=rd.broken, theory_spent=theory_spent, **cfg)


def test_shadow_zero_divergence_on_plain_repair():
    sketch = json.dumps({
        "haves": [{"id": "h1", "type": "a^2 + b^2 ≥ 2*a*b",
                   "tactic": "positivity", "depends": []}],
        "closer": "exact h1"})
    repaired = json.dumps({"repairs": [
        {"id": "h1", "tactic": "nlinarith [sq_nonneg (a-b)]"}]})
    state = {"n": 0}

    def llm(system, user):
        return repaired if "BROKEN steps" in user else sketch

    def verify(header, body):
        state["n"] += 1
        if "nlinarith" in body:
            return {"ok": True, "errors": None, "body_line_offset": 3}
        return {"ok": False, "errors": "f.lean:4:2: error: unsolved goals",
                "body_line_offset": 3}

    res, rec = _run_legacy(llm, verify)
    assert res.verified
    cfg = _static_config()
    # theory never spent in this scenario
    divs = [compare_round(rd.round_no, rd.stages_fired,
                          _ctx_from_round(rd, cfg, theory_spent=False))
            for rd in rec.round_decisions()]
    assert rec.round_decisions()            # at least one repair round ran
    assert [d for d in divs if d is not None] == []


def test_shadow_zero_divergence_on_arm_closer_commit():
    sketch = json.dumps({
        "haves": [{"id": "h1", "type": "a^2 + b^2 ≥ 2*a*b",
                   "tactic": "nlinarith [sq_nonneg (a-b)]", "depends": []}],
        "closer": "exact h1"})
    theory = json.dumps({
        "defs": [], "lemmas": [{"name": "aux_bridge",
                   "statement": "lemma aux_bridge (x y : ℝ) : "
                                "x^2 + y^2 ≥ 2*x*y"}],
        "leaf_tactics": {"__closer__": "exact aux_bridge a b"}})

    def llm(system, user):
        if system == THEORY_SYSTEM:
            return theory
        if system == PROVE_LEMMA_SYSTEM:
            return json.dumps({"proof": "GOODPROOF"})
        return sketch

    def verify(header, body):
        if "theorem t" not in header:
            return {"ok": "GOODPROOF" in body, "errors": "e",
                    "body_line_offset": 1}
        ok = "aux_bridge" in header and "exact aux_bridge a b" in body
        return {"ok": ok, "errors": None if ok else
                "f.lean:6:2: error: unsolved goals\n⊢ nope",
                "body_line_offset": 3}

    def probe(header, body):
        return {"ok": ":= by sorry" in header and
                "exact aux_bridge a b" in body, "errors": "e",
                "body_line_offset": 3}

    res, rec = _run_legacy(
        llm, verify, probe=probe, abduce_lemmas=True, abduce_mode="theory",
        abduce_theory_trigger="always")
    assert res.verified and CLOSER_ID in res.abduced_ids
    cfg = _static_config(abduce_lemmas=True, abduce_mode="theory",
                         has_probe=True, trigger="always")
    # Reconstruct theory_spent: it flips True the first round a theory
    # stage fires; feed that forward.
    spent = False
    seen_div: list[Divergence] = []
    for rd in rec.round_decisions():
        d = compare_round(rd.round_no, rd.stages_fired,
                          _ctx_from_round(rd, cfg, spent))
        if d is not None:
            seen_div.append(d)
        if STAGE_THEORY_CLOSER in rd.stages_fired or \
                STAGE_THEORY_LEAF in rd.stages_fired:
            spent = True
    assert seen_div == []
