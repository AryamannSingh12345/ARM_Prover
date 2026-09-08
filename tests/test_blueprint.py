"""Blueprint-first sketching + the sufficiency probe.

Two properties carry the design:

1. **Legacy-when-off.** `sketch_mode="tactics"` (the default) must go down
   the original path — same prompt, same parser, no probe.
2. **The probe checks the PLAN, not the tactics.** With every claim granted
   for free (sorry stubs), does the closer close the goal? A blueprint that
   fails is regenerated with the Lean errors fed back, before anything is
   proved.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search import blueprint as bp                          # noqa: E402
from search import proof_dag                                # noqa: E402


def _bp_json(claims=None, closer="exact h1", setup=None):
    claims = claims if claims is not None else [
        {"id": "h1", "type": "n + 0 = n", "depends": []}]
    return json.dumps({"setup": setup or [], "claims": claims,
                       "closer": closer})


# --------------------------------------------------------------------------
# parsing / validation
# --------------------------------------------------------------------------

def test_parse_minimal_blueprint():
    obj, err = bp.parse_blueprint(_bp_json())
    assert err is None
    assert obj["claims"][0]["id"] == "h1" and obj["closer"] == "exact h1"


def test_parse_rejects_missing_closer():
    obj, err = bp.parse_blueprint(_bp_json(closer=""))
    assert obj is None and "closer" in err


def test_parse_rejects_empty_claims():
    obj, err = bp.parse_blueprint(_bp_json(claims=[]))
    assert obj is None and "claims" in err


def test_parse_rejects_bad_identifier():
    obj, err = bp.parse_blueprint(
        _bp_json(claims=[{"id": "not an id", "type": "True", "depends": []}]))
    assert obj is None and "malformed" in err


def test_parse_rejects_forbidden_tactic_in_closer():
    obj, err = bp.parse_blueprint(_bp_json(closer="sorry"))
    assert obj is None and "forbidden" in err


def test_parse_ignores_any_tactic_field():
    """Tactics are not part of this contract."""
    obj, _ = bp.parse_blueprint(_bp_json(
        claims=[{"id": "h1", "type": "True", "depends": [],
                 "tactic": "trivial"}]))
    assert "tactic" not in obj["claims"][0]


def test_parse_tolerates_fences_and_prose():
    obj, err = bp.parse_blueprint("Here:\n```json\n" + _bp_json() + "\n```")
    assert err is None and obj["claims"]


def test_validate_catches_dangling_reference():
    err = bp.validate_references(
        [{"id": "a", "type": "T", "depends": ["ghost"]}])
    assert err and "ghost" in err


def test_validate_catches_forward_reference():
    err = bp.validate_references([
        {"id": "a", "type": "T", "depends": ["b"]},
        {"id": "b", "type": "T", "depends": []}])
    assert err and "`b`" in err


def test_validate_catches_duplicate_id():
    err = bp.validate_references([
        {"id": "a", "type": "T", "depends": []},
        {"id": "a", "type": "U", "depends": []}])
    assert err and "duplicate" in err


def test_validate_accepts_well_ordered_chain():
    assert bp.validate_references([
        {"id": "a", "type": "T", "depends": []},
        {"id": "b", "type": "U", "depends": ["a"]}]) is None


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def test_generate_one_shot():
    out, err = bp.generate_blueprint("theorem t : True",
                                     llm_call=lambda s, u: _bp_json())
    assert err is None and len(out["claims"]) == 1


def test_generate_rejects_dangling_at_generation_time():
    bad = _bp_json(claims=[{"id": "a", "type": "T", "depends": ["ghost"]}])
    out, err = bp.generate_blueprint("theorem t : True",
                                     llm_call=lambda s, u: bad)
    assert out is None and "ghost" in err


def test_generate_survives_llm_exception():
    def boom(s, u):
        raise RuntimeError("down")
    out, err = bp.generate_blueprint("theorem t : True", llm_call=boom)
    assert out is None and "llm error" in err


def test_sequential_accumulates_then_completes():
    steps = [
        json.dumps({"done": False,
                    "claim": {"id": "a", "type": "P", "depends": []},
                    "closer": ""}),
        json.dumps({"done": False,
                    "claim": {"id": "b", "type": "Q", "depends": ["a"]},
                    "closer": ""}),
        json.dumps({"done": True, "claim": None, "closer": "exact b"}),
    ]
    it = iter(steps)
    out, err = bp.generate_blueprint("theorem t : True",
                                     llm_call=lambda s, u: next(it),
                                     sequential=True)
    assert err is None
    assert [c["id"] for c in out["claims"]] == ["a", "b"]
    assert out["closer"] == "exact b"


def test_sequential_conditions_on_established_claims():
    seen: list[str] = []

    def llm(system, user):
        seen.append(user)
        if len(seen) == 1:
            return json.dumps({"done": False,
                               "claim": {"id": "a", "type": "P",
                                         "depends": []}, "closer": ""})
        return json.dumps({"done": True, "claim": None, "closer": "exact a"})

    bp.generate_blueprint("theorem t : True", llm_call=llm, sequential=True)
    assert "No claims established yet" in seen[0]
    assert "a : P" in seen[1]          # step 2 saw step 1's claim


def test_sequential_refuses_done_with_no_claims():
    out, err = bp.generate_blueprint(
        "theorem t : True", sequential=True,
        llm_call=lambda s, u: json.dumps(
            {"done": True, "claim": None, "closer": "trivial"}))
    assert out is None and "no claims" in err


def test_sequential_is_bounded():
    """A model that never says done must not loop forever."""
    n = {"i": 0}

    def llm(system, user):
        n["i"] += 1
        return json.dumps({"done": False,
                           "claim": {"id": f"c{n['i']}", "type": "P",
                                     "depends": []}, "closer": ""})
    out, err = bp.generate_blueprint("theorem t : True", llm_call=llm,
                                     sequential=True, max_claims=4)
    assert out is None and "exceeded" in err
    assert n["i"] <= 5


# --------------------------------------------------------------------------
# end-to-end through attempt_dag_proof
# --------------------------------------------------------------------------

HEADER = "import Mathlib\n\ntheorem tgt (n : ℕ) : n + 0 = n"


class _H:
    """Records which system prompt was used and every probe/verify body."""

    def __init__(self, *, probe_ok, verify_ok=False, responses=None):
        self.probe_ok = probe_ok
        self.verify_ok = verify_ok
        self.responses = list(responses or [])
        self.systems: list[str] = []
        self.probes: list[str] = []
        self.n_probe = 0

    def llm(self, system, user):
        self.systems.append(system)
        if self.responses:
            return self.responses.pop(0)
        return _bp_json()

    def probe(self, header, body):
        self.n_probe += 1
        self.probes.append(body)
        ok = self.probe_ok(self.n_probe)
        return {"ok": ok, "errors": None if ok else "error: unsolved goals",
                "body_line_offset": 1}

    def verify(self, header, body):
        return {"ok": self.verify_ok, "errors": None if self.verify_ok
                else "error: unsolved goals", "body_line_offset": 1}

    def run(self, **kw):
        return proof_dag.attempt_dag_proof(
            HEADER, sketch_llm_call=self.llm, verify_fn=self.verify,
            probe_fn=self.probe, sketch_attempts=1, repair_rounds=0, **kw)


def test_default_mode_uses_the_legacy_sketch_prompt():
    h = _H(probe_ok=lambda n: True,
           responses=['{"haves": [{"id":"h1","type":"n + 0 = n",'
                      '"tactic":"simp","depends":[]}], "closer":"exact h1"}'])
    h.run()
    assert proof_dag.SKETCH_SYSTEM in h.systems
    assert h.n_probe == 0            # no sufficiency probe in legacy mode


def test_blueprint_mode_uses_the_blueprint_prompt_and_probes():
    h = _H(probe_ok=lambda n: True)
    h.run(sketch_mode="blueprint")
    assert bp.BLUEPRINT_SYSTEM in h.systems
    assert proof_dag.SKETCH_SYSTEM not in h.systems
    assert h.n_probe == 1


def test_probe_stubs_every_claim_with_sorry():
    """The probe must grant the claims for free — that is what makes it a
    test of the PLAN rather than of the tactics."""
    h = _H(probe_ok=lambda n: True)
    h.run(sketch_mode="blueprint")
    assert "sorry" in h.probes[0]
    assert "have h1" in h.probes[0]


def test_insufficient_blueprint_is_regenerated_with_feedback():
    h = _H(probe_ok=lambda n: n >= 2)     # first plan rejected
    res = h.run(sketch_mode="blueprint", blueprint_rounds=2)
    assert h.n_probe == 2
    assert any("blueprint_insufficient" in e for e in res.repair_errors)


def test_regeneration_is_bounded_by_blueprint_rounds():
    h = _H(probe_ok=lambda n: False)      # never satisfied
    h.run(sketch_mode="blueprint", blueprint_rounds=2)
    assert h.n_probe == 3                 # initial + 2 regenerations


def test_blueprint_proceeds_when_probe_unavailable():
    """No probe_fn → cannot check → must not block."""
    res = proof_dag.attempt_dag_proof(
        HEADER, sketch_llm_call=lambda s, u: _bp_json(),
        verify_fn=lambda h, b: {"ok": False, "errors": "e",
                                "body_line_offset": 1},
        probe_fn=None, sketch_attempts=1, repair_rounds=0,
        sketch_mode="blueprint")
    assert res.verified is False          # ran to completion, no crash


def test_blueprint_claims_carry_no_tactics():
    """Tactics come from the ladder, not the planner."""
    captured = {}

    def verify(header, body):
        captured["body"] = body
        return {"ok": True, "errors": None, "body_line_offset": 1}

    res = proof_dag.attempt_dag_proof(
        HEADER, sketch_llm_call=lambda s, u: _bp_json(),
        verify_fn=verify,
        probe_fn=lambda h, b: {"ok": True, "errors": None,
                               "body_line_offset": 1},
        sketch_attempts=1, repair_rounds=0, sketch_mode="blueprint",
        leaf_fallbacks=("norm_num", "omega"))
    assert res.verified
    assert all(h.tactic == "" for h in res.sketch.haves)
    # Each alternative parenthesised — see test_fallback_ladder_parens.py:
    # a bare ladder entry containing `<;>` swallows the next alternative.
    assert "first | (norm_num) | (omega)" in captured["body"]


def test_unknown_sketch_mode_rejected():
    with pytest.raises(ValueError):
        proof_dag.attempt_dag_proof(
            HEADER, sketch_llm_call=lambda s, u: "{}",
            verify_fn=lambda h, b: {"ok": True}, sketch_mode="nonsense")


def test_blueprint_system_prompt_is_domain_agnostic():
    import re
    low = bp.BLUEPRINT_SYSTEM.lower() + bp.SEQUENTIAL_SYSTEM.lower()
    for banned in ("minif2f", "putnam", "amc", "aime", "imo", "2520",
                   "b5", "lrs"):
        assert not re.search(rf"\b{re.escape(banned)}\b", low), banned


def test_blueprint_prompt_forbids_tactics_on_claims():
    assert "DO NOT WRITE TACTICS" in bp.BLUEPRINT_SYSTEM
    assert "DO NOT WRITE A TACTIC" in bp.SEQUENTIAL_SYSTEM
