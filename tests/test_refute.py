"""Pre-flight refutation search.

The property that matters most is SOUNDNESS: a refutation is believed
only when the verifier accepts a proof of the negation. A model that
merely *claims* to have refuted the goal must change nothing. Everything
else here is parsing and loop control.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search import refute                                   # noqa: E402
from search.refute import (                                  # noqa: E402
    attempt_refutation, build_refutation_header, parse_refutation,
    REFUTATION_DECL, REFUTE_SYSTEM,
)

PROP = "∀ (n : ℕ), n * n ≠ n + 12"


def _resp(verdict="refuted", proof="exact fun h => by simpa using h 4",
          decls=None, witness="n = 4", rationale="4*4 = 16 = 4+12"):
    import json
    return json.dumps({"verdict": verdict, "witness": witness,
                       "decls": decls or [], "proof": proof,
                       "rationale": rationale})


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def test_parse_refuted():
    obj, err = parse_refutation(_resp())
    assert err is None
    assert obj["verdict"] == "refuted" and obj["proof"]


def test_parse_no_counterexample_needs_no_proof():
    obj, err = parse_refutation(_resp(verdict="no_counterexample", proof=""))
    assert err is None and obj["verdict"] == "no_counterexample"


def test_parse_refuted_without_proof_is_rejected():
    obj, err = parse_refutation(_resp(proof=""))
    assert obj is None and "no proof" in err


def test_parse_rejects_unknown_verdict():
    obj, err = parse_refutation(_resp(verdict="maybe"))
    assert obj is None and "verdict" in err


def test_parse_rejects_non_json():
    obj, err = parse_refutation("I think the statement is false because …")
    assert obj is None and "JSON" in err


def test_parse_strips_leading_by():
    """`by` would become `by by` after assembly — the same slip the ARM
    prove step had to guard against."""
    obj, _ = parse_refutation(_resp(proof="by decide"))
    assert obj["proof"] == "decide"


def test_parse_accepts_decls_as_bare_string():
    obj, _ = parse_refutation(_resp(decls="def foo : ℕ := 4"))
    assert obj["decls"] == ["def foo : ℕ := 4"]


def test_parse_tolerates_prose_and_fences_around_json():
    raw = "Sure!\n```json\n" + _resp() + "\n```\nHope that helps."
    obj, err = parse_refutation(raw)
    assert err is None and obj["verdict"] == "refuted"


# --------------------------------------------------------------------------
# header assembly
# --------------------------------------------------------------------------

def test_header_negates_the_proposition():
    h = build_refutation_header("import X", ["def d : ℕ := 4"], PROP)
    assert f"theorem {REFUTATION_DECL} : ¬ (" in h
    assert PROP in h and "def d : ℕ := 4" in h


def test_header_without_prelude_or_decls():
    h = build_refutation_header("", [], PROP)
    assert h == f"theorem {REFUTATION_DECL} : ¬ ({PROP})"


# --------------------------------------------------------------------------
# the loop — soundness first
# --------------------------------------------------------------------------

class _Harness:
    def __init__(self, responses, verify_ok):
        self.responses = list(responses)
        self.verify_ok = verify_ok
        self.prompts: list[str] = []
        self.verifies: list[tuple[str, str]] = []

    def llm(self, system, user):
        self.prompts.append(user)
        return self.responses.pop(0) if self.responses else _resp()

    def verify(self, header, body):
        self.verifies.append((header, body))
        ok = self.verify_ok(len(self.verifies))
        return {"ok": ok, "errors": None if ok else "error: it does hold",
                "body_line_offset": 1}


def test_kernel_accepts_gives_refuted():
    h = _Harness([_resp()], lambda n: True)
    r = attempt_refutation(PROP, llm_call=h.llm, verify_fn=h.verify,
                           attempts=5)
    assert r.refuted and r.proof and r.refutation_source
    assert "REFUTED" in r.log[-1]


def test_model_claim_without_kernel_is_NOT_refuted():
    """The soundness test. The model insists every time; Lean always says
    no; nothing is refuted."""
    h = _Harness([_resp()] * 5, lambda n: False)
    r = attempt_refutation(PROP, llm_call=h.llm, verify_fn=h.verify,
                           attempts=5)
    assert r.refuted is False
    assert r.proof is None and r.refutation_source is None
    assert r.attempts_used == 5 and len(h.verifies) == 5


def test_failures_are_fed_back_into_later_prompts():
    h = _Harness([_resp()] * 3, lambda n: n >= 3)
    r = attempt_refutation(PROP, llm_call=h.llm, verify_fn=h.verify,
                           attempts=3)
    assert r.refuted
    assert "PREVIOUS ATTEMPTS AT REFUTATION FAILED" in h.prompts[1]
    assert "it does hold" in h.prompts[1]


def test_single_decline_does_not_end_the_search():
    """A first decline earns a push-back, not termination.

    Regression on `lrs_refute_v1`: the model declined on attempt 1 after
    104s, ending a 5-attempt budget at 1 — on a statement that is in fact
    false.
    """
    h = _Harness([_resp(verdict="no_counterexample", proof=""), _resp()],
                 lambda n: True)
    r = attempt_refutation(PROP, llm_call=h.llm, verify_fn=h.verify,
                           attempts=5)
    assert r.refuted                     # the retry found one
    assert r.attempts_used == 2 and r.declines == 1


def test_pushback_text_is_sent_after_a_decline():
    h = _Harness([_resp(verdict="no_counterexample", proof=""), _resp()],
                 lambda n: True)
    attempt_refutation(PROP, llm_call=h.llm, verify_fn=h.verify, attempts=5)
    assert "YOU DECLINED" in h.prompts[1]
    assert "CONSTRUCT, don't search" in h.prompts[1]
    assert "YOU DECLINED" not in h.prompts[0]


def test_declining_uses_the_WHOLE_budget_by_default():
    """The caller asked for 5 tries, so a reluctant model gets 5 — no
    short-circuit. Earlier versions used 1, then 2."""
    h = _Harness([_resp(verdict="no_counterexample", proof="")] * 9,
                 lambda n: False)
    r = attempt_refutation(PROP, llm_call=h.llm, verify_fn=h.verify,
                           attempts=5)
    assert r.refuted is False
    assert r.attempts_used == 5 and r.declines == 5
    assert h.verifies == []              # refusals never reach Lean


def test_max_declines_caps_the_search_when_asked():
    h = _Harness([_resp(verdict="no_counterexample", proof="")] * 9,
                 lambda n: False)
    r = attempt_refutation(PROP, llm_call=h.llm, verify_fn=h.verify,
                           attempts=5, max_declines=2)
    assert r.attempts_used == 2 and r.declines == 2


def test_escalation_note_appears_from_the_second_decline():
    h = _Harness([_resp(verdict="no_counterexample", proof="")] * 9,
                 lambda n: False)
    attempt_refutation(PROP, llm_call=h.llm, verify_fn=h.verify, attempts=5)
    assert "THIS IS DECLINE" not in h.prompts[0]
    assert "THIS IS DECLINE" not in h.prompts[1]   # push-back only
    assert "THIS IS DECLINE 2 OF AT MOST 5" in h.prompts[2]
    assert "WRITE IT OUT" in h.prompts[2]


def test_escalation_preserves_the_right_to_decline():
    """Pressure must never become an instruction to fabricate — a
    fabricated witness costs a compile and teaches nothing."""
    from search.refute import ESCALATION_NOTE
    assert "Do not invent a witness you believe is wrong" in ESCALATION_NOTE
    assert "Only decline again if" in ESCALATION_NOTE


def test_declines_accumulate_across_genuine_attempts():
    h = _Harness([_resp(verdict="no_counterexample", proof=""),
                  _resp(),                       # engages, kernel rejects
                  _resp(verdict="no_counterexample", proof=""),
                  _resp()], lambda n: n >= 2)
    r = attempt_refutation(PROP, llm_call=h.llm, verify_fn=h.verify,
                           attempts=5)
    assert r.declines == 2 and r.refuted


def test_forbidden_tactic_is_discarded_and_retried():
    h = _Harness([_resp(proof="sorry"), _resp()], lambda n: True)
    r = attempt_refutation(PROP, llm_call=h.llm, verify_fn=h.verify,
                           attempts=3)
    assert r.refuted
    assert len(h.verifies) == 1      # the sorry attempt never reached Lean
    assert "forbidden" in r.log[0]


def test_forbidden_tactic_in_decls_is_also_caught():
    h = _Harness([_resp(decls=["lemma bad : True := by sorry"]), _resp()],
                 lambda n: True)
    r = attempt_refutation(PROP, llm_call=h.llm, verify_fn=h.verify,
                           attempts=3)
    assert r.refuted and len(h.verifies) == 1


def test_unparseable_response_retries():
    h = _Harness(["not json at all", _resp()], lambda n: True)
    r = attempt_refutation(PROP, llm_call=h.llm, verify_fn=h.verify,
                           attempts=3)
    assert r.refuted and r.attempts_used == 2


def test_llm_exception_is_survivable():
    def boom(system, user):
        raise RuntimeError("transport died")
    r = attempt_refutation(PROP, llm_call=boom,
                           verify_fn=lambda h, b: {"ok": True}, attempts=5)
    assert r.refuted is False and "llm error" in r.log[0]


def test_verifier_exception_is_survivable():
    def boom(header, body):
        raise RuntimeError("lake died")
    r = attempt_refutation(PROP, llm_call=lambda s, u: _resp(),
                           verify_fn=boom, attempts=5)
    assert r.refuted is False and "verifier error" in r.log[0]


def test_zero_attempts_does_nothing():
    h = _Harness([_resp()], lambda n: True)
    r = attempt_refutation(PROP, llm_call=h.llm, verify_fn=h.verify,
                           attempts=0)
    assert r.refuted is False and r.attempts_used == 0
    assert h.prompts == [] and h.verifies == []


def test_attempts_are_bounded():
    h = _Harness([_resp()] * 20, lambda n: False)
    r = attempt_refutation(PROP, llm_call=h.llm, verify_fn=h.verify,
                           attempts=3)
    assert r.attempts_used == 3 and len(h.verifies) == 3


def test_result_dict_is_json_shaped():
    import json
    h = _Harness([_resp()], lambda n: True)
    r = attempt_refutation(PROP, llm_call=h.llm, verify_fn=h.verify)
    json.dumps(r.to_dict())          # must not raise
    assert r.to_dict()["refuted"] is True


def test_default_attempts_is_five():
    import inspect
    assert inspect.signature(attempt_refutation).parameters[
        "attempts"].default == 5


# --------------------------------------------------------------------------
# prompt content — the anti-fabrication guarantees
# --------------------------------------------------------------------------

def test_system_prompt_permits_declining():
    """Without an approved way to say 'this is true', a model asked to
    refute will fabricate."""
    assert "no_counterexample" in REFUTE_SYSTEM
    assert "HONESTY REQUIREMENT" in REFUTE_SYSTEM


def test_system_prompt_states_the_hypothesis_rule():
    assert "EVERY hypothesis" in REFUTE_SYSTEM


def test_system_prompt_bans_escape_hatches():
    for bad in ("sorry", "admit", "native_decide"):
        assert bad in REFUTE_SYSTEM


def test_system_prompt_is_domain_agnostic():
    """No problem-shaped hardcoding (repo rule).

    What is banned is problem/benchmark IDENTITY — dataset names, problem
    ids, the specific constants and objects of a problem we happen to have
    worked on. What is NOT banned is general mathematical vocabulary:
    a counterexample checklist that could not say "equality case", "empty
    set" or "index edge" would be useless, and those heuristics come from
    the standard literature, not from any target.

    Word-boundary matched: an earlier version of this test failed on
    'aime' inside 'claimed'.
    """
    import re
    low = REFUTE_SYSTEM.lower()
    banned = ("minif2f", "putnambench", "putnam", "amc", "aime",
              "imo", "smoke10", "2520", "b5", "lrs", "mathd")
    hits = [b for b in banned if re.search(rf"\b{re.escape(b)}\b", low)]
    assert not hits, f"problem-shaped tokens in the prompt: {hits}"


def test_system_prompt_has_no_problem_specific_constants():
    """No bare numeric literals beyond the small structural ones a
    degenerate-case checklist legitimately names (0, 1, 2, 3, 4 for the
    enumerated steps and boundary values)."""
    import re
    nums = {int(n) for n in re.findall(r"\b\d+\b", REFUTE_SYSTEM)}
    assert nums <= {0, 1, 2, 3, 4, 5, 6, 7}, sorted(nums)


def test_user_prompt_carries_prop_failures_and_premises():
    u = refute.refute_user_prompt(PROP, ["boom"], ["Nat.succ_le"])
    assert PROP in u and "boom" in u and "Nat.succ_le" in u
