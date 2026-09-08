"""Theory invention: reframing a problem into a domain where it is routine.

`_abduce_theory` decomposes a problem in its own language; it has no
bridge and no leverage test, so it cannot change the frame. Across this
session it produced a correct decomposition on every hard problem and
never once reframed one.

The properties that make reframing economical, pinned here:

* the LEVERAGE gate costs no compile — a theory whose hardest claim is as
  hard as the target is rejected before Lean is touched;
* the SUFFICIENCY probe costs one compile and runs BEFORE any proving —
  granting every claim, does the derivation close the goal?
* a theory is returned only when that probe passed, so the caller never
  spends proof effort on a frame that could not have worked;
* the probe tolerates `sorry` and never counts as a solve.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.reframe import (                                   # noqa: E402
    Theory, assemble_theory_header, leverage_ok, parse_reframe,
    propose_reframe, REFRAME_SYSTEM,
)

TARGET = "theorem tgt (n : ℕ) : n + 0 = n"
GOAL = "n + 0 = n"


def _reframe(objects=None, bridges=None, theorems=None,
             derivation="exact bridge_one n", name="Frame"):
    return json.dumps({
        "name": name, "rationale": "because",
        "objects": objects if objects is not None else ["def F : ℕ := 0"],
        "bridges": bridges if bridges is not None
        else ["lemma bridge_one (n : ℕ) : n + 0 = n"],
        "theorems": theorems if theorems is not None else [],
        "derivation": derivation})


# ---- parsing ----------------------------------------------------------------

def test_parse_minimal_reframe():
    t, err = parse_reframe(_reframe())
    assert err is None
    assert t.objects and t.bridges and t.derivation == "exact bridge_one n"


def test_empty_reframe_is_an_honest_decline():
    raw = json.dumps({"name": "", "rationale": "no better frame exists",
                      "objects": [], "bridges": [], "theorems": [],
                      "derivation": ""})
    t, err = parse_reframe(raw)
    assert err is None and t.is_empty
    assert "no better frame" in t.rationale


def test_reframe_without_derivation_is_rejected():
    t, err = parse_reframe(_reframe(derivation=""))
    assert t is None and "derivation" in err


def test_unnamed_claim_is_rejected():
    """A claim must be a named lemma — it is proved and cited by name."""
    t, err = parse_reframe(_reframe(bridges=["∀ n : ℕ, n + 0 = n"]))
    assert t is None and "named" in err


def test_forbidden_tactic_is_rejected():
    t, err = parse_reframe(_reframe(derivation="sorry"))
    assert t is None and "forbidden" in err


def test_leading_by_is_stripped():
    t, _ = parse_reframe(_reframe(derivation="by exact bridge_one n"))
    assert t.derivation == "exact bridge_one n"


# ---- leverage (free) --------------------------------------------------------

def _difficulty(s: str) -> float:
    return float(len(s))


def test_leverage_rejects_a_claim_as_hard_as_the_target():
    t = Theory(name="X", bridges=["lemma b : " + "x" * 500],
               derivation="exact b")
    ok, why = leverage_ok(t, "x" * 100, _difficulty)
    assert not ok and "no leverage" in why


def test_leverage_accepts_genuinely_easier_claims():
    t = Theory(name="X", bridges=["lemma b : x"], derivation="exact b")
    ok, _ = leverage_ok(t, "x" * 500, _difficulty)
    assert ok


def test_leverage_never_blocks_when_it_cannot_judge():
    def boom(_s):
        raise RuntimeError("nope")
    t = Theory(name="X", bridges=["lemma b : x"], derivation="exact b")
    ok, _ = leverage_ok(t, "goal", boom)
    assert ok


# ---- assembly ---------------------------------------------------------------

def test_probe_header_stubs_every_claim():
    t = Theory(objects=["def F : ℕ := 0"],
               bridges=["lemma b1 : True"], theorems=["lemma t1 : True"],
               derivation="trivial")
    h = assemble_theory_header("import Mathlib", t, stub=True)
    assert h.count(":= by sorry") == 2
    assert "def F : ℕ := 0" in h
    assert h.index("def F") < h.index("lemma b1")     # objects first


def test_unstubbed_header_has_no_sorry():
    t = Theory(objects=[], bridges=["lemma b1 : True"], derivation="trivial")
    h = assemble_theory_header("", t, stub=False)
    assert "sorry" not in h


# ---- the loop ---------------------------------------------------------------

class _H:
    def __init__(self, replies, probe_ok):
        self.replies = list(replies)
        self.probe_ok = probe_ok
        self.n_probe = 0
        self.prompts: list[str] = []

    def llm(self, system, user):
        self.prompts.append(user)
        return self.replies.pop(0) if self.replies else _reframe()

    def probe(self, header, body):
        self.n_probe += 1
        ok = self.probe_ok(self.n_probe)
        return {"ok": ok, "errors": None if ok else "error: unsolved goals"}


def test_sufficient_theory_is_returned():
    h = _H([_reframe()], probe_ok=lambda n: True)
    r = propose_reframe(TARGET, GOAL, llm_call=h.llm, probe_fn=h.probe)
    assert r.sufficient and r.theory is not None
    assert "SUFFICIENT" in r.log[-1]


def test_insufficient_theory_is_not_returned_as_sufficient():
    h = _H([_reframe(), _reframe()], probe_ok=lambda n: False)
    r = propose_reframe(TARGET, GOAL, llm_call=h.llm, probe_fn=h.probe,
                        rounds=2)
    assert r.sufficient is False
    assert h.n_probe == 2


def test_probe_failure_is_fed_back():
    h = _H([_reframe(), _reframe()], probe_ok=lambda n: n == 2)
    r = propose_reframe(TARGET, GOAL, llm_call=h.llm, probe_fn=h.probe,
                        rounds=2)
    assert r.sufficient
    assert "did NOT close the goal" in h.prompts[1]


def test_no_leverage_costs_zero_compiles():
    """The whole point of the gate: reject before touching Lean."""
    hard = _reframe(bridges=["lemma b : " + "x" * 900])
    h = _H([hard, hard], probe_ok=lambda n: True)
    r = propose_reframe(TARGET, "x" * 100, llm_call=h.llm, probe_fn=h.probe,
                        rounds=2, difficulty_fn=_difficulty)
    assert h.n_probe == 0
    assert r.sufficient is False
    assert any("no leverage" in ln for ln in r.log)


def test_decline_stops_immediately():
    raw = json.dumps({"name": "", "rationale": "none", "objects": [],
                      "bridges": [], "theorems": [], "derivation": ""})
    h = _H([raw, _reframe()], probe_ok=lambda n: True)
    r = propose_reframe(TARGET, GOAL, llm_call=h.llm, probe_fn=h.probe,
                        rounds=3)
    assert r.rounds_used == 1 and h.n_probe == 0
    assert r.theory.is_empty and not r.sufficient


def test_llm_and_probe_failures_are_survivable():
    def boom(system, user):
        raise RuntimeError("down")
    r = propose_reframe(TARGET, GOAL, llm_call=boom,
                        probe_fn=lambda h, b: {"ok": True})
    assert not r.sufficient

    h = _H([_reframe()], probe_ok=lambda n: True)

    def pboom(header, body):
        raise RuntimeError("lake died")
    r2 = propose_reframe(TARGET, GOAL, llm_call=h.llm, probe_fn=pboom)
    assert not r2.sufficient


def test_result_is_json_shaped():
    h = _H([_reframe()], probe_ok=lambda n: True)
    r = propose_reframe(TARGET, GOAL, llm_call=h.llm, probe_fn=h.probe)
    json.dumps(r.to_dict())


# ---- the prompt -------------------------------------------------------------

def test_prompt_demands_a_frame_change_not_a_decomposition():
    assert "CHANGE THE FRAME" in REFRAME_SYSTEM
    assert "decompose" in REFRAME_SYSTEM.lower()


def test_prompt_states_the_leverage_requirement():
    assert "LEVERAGE" in REFRAME_SYSTEM
    assert "moved the difficulty" in REFRAME_SYSTEM


def test_prompt_prefers_existing_mathlib_theory():
    assert "PREFER EXISTING THEORY" in REFRAME_SYSTEM


def test_prompt_permits_declining():
    assert "honest \"no reframe\"" in REFRAME_SYSTEM


def test_prompt_is_domain_agnostic():
    import re
    low = REFRAME_SYSTEM.lower()
    for banned in ("galois", "putnam", "minif2f", "recurrence", "polygon"):
        assert not re.search(rf"\b{banned}\b", low), banned
