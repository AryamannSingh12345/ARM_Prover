"""Recognition wired ahead of the theory loop (flag-gated, default OFF).

Inventing a lemma that Mathlib already carries is the most expensive
possible way to fail: `putnam_2020_a2` spent its whole budget on a
reindexing `Finset.sum_range_succ'` performs in one step. So before any
theory is proposed, each stuck leaf is checked against what has already
been proved.

The ordering the whole chain depends on:

    recognize  -> does this ALREADY EXIST?      (1 compile per leaf)
    abduce     -> decompose in its OWN language (many)
    reframe    -> CHANGE THE FRAME              (escalation, last)

What these tests pin:

* OFF is the legacy path exactly — `_abduce_theory` called with the
  untouched broken list;
* a recognised leaf is CLOSED and never reaches the theory loop;
* recognition is asked BEFORE the theory prompt, not after;
* a recognised leaf stays fixed even when the theory loop abandons the
  rest — it was kernel-accepted on its own standalone statement;
* retrieval searches the GOAL but confirmation uses the WHOLE statement,
  because a goal lifted out of its binders has unbound variables;
* a failing or raising recogniser costs the run nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.proof_dag import (                                  # noqa: E402
    PROVE_LEMMA_SYSTEM, THEORY_SYSTEM, attempt_dag_proof,
)
from search.recognize import (                                  # noqa: E402
    Candidate, StatementIndex, recognize_for_statement,
)

HEADER = "theorem t (a b : ℝ) (ha : 0 < a) : a^2 + b^2 ≥ 2*a*b"
SKETCH = json.dumps({
    "setup": [],
    "haves": [{"id": "h1", "type": "a^2 + b^2 ≥ 2*a*b",
               "tactic": "positivity", "depends": []}],
    "closer": "exact h1",
})
THEORY = json.dumps({
    "defs": [],
    "lemmas": [{"name": "aux_b",
                "statement": "lemma aux_b (x y : ℝ) : x^2 + y^2 ≥ 2*x*y"}],
    "leaf_tactics": {"h1": "nlinarith [aux_b a b]"},
})
TIMEOUT_ERR = "f.lean:5:4: error: timeout: heartbeat exceeded"
RECOGNIZED = "first\n  | (exact two_mul_le_add_sq a b)"


def _fixture(recognizer):
    c = {"theory": 0, "prove": 0, "recognize": 0, "order": []}

    def llm(system, user):
        if system == THEORY_SYSTEM:
            c["theory"] += 1
            c["order"].append("theory")
            return THEORY
        if system == PROVE_LEMMA_SYSTEM:
            c["prove"] += 1
            return json.dumps({"proof": "GOODPROOF"})
        return SKETCH

    def verify(header, body):
        if "theorem t" not in header:
            ok = "GOODPROOF" in body
            return {"ok": ok, "errors": None if ok else "e: nope",
                    "body_line_offset": 1}
        ok = ("aux_b" in header and ":= by sorry" not in header) \
            or "two_mul_le_add_sq" in body
        return {"ok": ok, "errors": None if ok else TIMEOUT_ERR,
                "body_line_offset": 3}

    def probe(header, body):
        return {"ok": "aux_b" in header, "errors": None,
                "body_line_offset": 3}

    def rec(stmt):
        c["recognize"] += 1
        c["order"].append("recognize")
        return recognizer(stmt)

    return llm, verify, probe, rec, c


def _run(llm, verify, probe, rec, *, on: bool):
    return attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify,
        sketch_attempts=1, repair_rounds=3, leaf_fallbacks=(),
        decompose_depth=0, abduce_lemmas=True, abduce_mode="theory",
        probe_fn=probe, abduce_theory_rounds=1,
        abduce_theory_trigger="always",
        recognize_leaf_call=(rec if on else None),
    )


# ---- off is legacy ----------------------------------------------------------

def test_off_never_calls_the_recognizer():
    llm, verify, probe, rec, c = _fixture(lambda _s: RECOGNIZED)
    _run(llm, verify, probe, rec, on=False)
    assert c["recognize"] == 0
    assert c["theory"] >= 1                 # legacy path ran


def test_flag_defaults_to_off():
    import inspect
    sig = inspect.signature(attempt_dag_proof)
    assert sig.parameters["recognize_leaf_call"].default is None


# ---- recognition short-circuits the theory loop -----------------------------

def test_a_recognized_leaf_never_reaches_the_theory_loop():
    """The saving IS the theory loop not running."""
    llm, verify, probe, rec, c = _fixture(lambda _s: RECOGNIZED)
    res = _run(llm, verify, probe, rec, on=True)
    assert c["recognize"] >= 1
    assert c["theory"] == 0, "a leaf Mathlib already proves needs no theory"
    assert c["prove"] == 0
    assert res.verified


def test_recognition_is_asked_before_any_theory():
    llm, verify, probe, rec, c = _fixture(lambda _s: None)
    _run(llm, verify, probe, rec, on=True)
    assert c["order"], "nothing ran"
    assert c["order"][0] == "recognize"


def test_unrecognized_leaf_falls_through_to_theory():
    llm, verify, probe, rec, c = _fixture(lambda _s: None)
    res = _run(llm, verify, probe, rec, on=True)
    assert c["recognize"] >= 1
    assert c["theory"] >= 1
    assert res.verified                     # theory still fixed it


def test_recognized_tactic_lands_on_the_leaf():
    llm, verify, probe, rec, _c = _fixture(lambda _s: RECOGNIZED)
    res = _run(llm, verify, probe, rec, on=True)
    assert "two_mul_le_add_sq" in (res.assembled_proof or "")


# ---- failure is free --------------------------------------------------------

def test_a_raising_recognizer_does_not_break_the_run():
    def boom(_s):
        raise RuntimeError("index died")
    llm, verify, probe, rec, c = _fixture(boom)
    res = _run(llm, verify, probe, rec, on=True)
    assert c["theory"] >= 1                 # fell through cleanly
    assert res.verified


def test_recognizer_returning_empty_is_treated_as_no_answer():
    llm, verify, probe, rec, c = _fixture(lambda _s: "")
    _run(llm, verify, probe, rec, on=True)
    assert c["theory"] >= 1


# ---- the goal/statement split ----------------------------------------------

def _index():
    return StatementIndex([
        {"name": "two_mul_le_add_sq", "kind": "theorem", "module": "M",
         "statement": "theorem two_mul_le_add_sq (a b : ℝ) : "
                      "2 * a * b ≤ a ^ 2 + b ^ 2"},
    ])


class _V:
    def __init__(self, ok):
        self.ok = ok
        self.calls: list[tuple[str, str]] = []

    def __call__(self, header, body):
        self.calls.append((header, body))
        return {"ok": self.ok}


def test_confirmation_uses_the_whole_statement_not_the_bare_goal():
    """A goal lifted out of its binders has unbound variables and would
    fail to elaborate for reasons unrelated to whether the lemma fits."""
    v = _V(ok=True)
    stmt = "theorem leaf (a b : ℝ) : 2 * a * b ≤ a ^ 2 + b ^ 2"
    recognize_for_statement(stmt, "2 * a * b ≤ a ^ 2 + b ^ 2", _index(),
                            verify_fn=v)
    header, _body = v.calls[0]
    assert header == stmt
    assert "(a b : ℝ)" in header


def test_one_compile_per_leaf():
    v = _V(ok=True)
    recognize_for_statement("theorem leaf (a b : ℝ) : 2 * a * b ≤ a ^ 2 + b ^ 2",
                            "2 * a * b ≤ a ^ 2 + b ^ 2", _index(), verify_fn=v)
    assert len(v.calls) == 1


def test_kernel_rejection_yields_no_tactic():
    v = _V(ok=False)
    out = recognize_for_statement(
        "theorem leaf (a b : ℝ) : 2 * a * b ≤ a ^ 2 + b ^ 2",
        "2 * a * b ≤ a ^ 2 + b ^ 2", _index(), verify_fn=v)
    assert out is None


def test_no_candidates_costs_no_compile():
    v = _V(ok=True)
    out = recognize_for_statement("theorem leaf : True", "", _index(),
                                  verify_fn=v)
    assert out is None and v.calls == []
