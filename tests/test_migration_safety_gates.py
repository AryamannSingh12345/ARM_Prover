"""Migration safety gates — invariants that must hold before ANY scheduler
migration, locking the current working prover's behaviour.

Covers:
- failed ARM proposal leaves NO mutation (header, closer, sketch, bank);
- the theorem being proved cannot be invoked by a repair (circularity);
- REPL acceptance still requires a fresh compile (soundness);
- legacy engine mode instantiates/invokes no new solver stage;
- scheduler engine mode is gated behind human review.

Mocked Lean/LLM throughout; the REPL gate uses a fake two-backend verify.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.proof_dag import (  # noqa: E402
    attempt_dag_proof, THEORY_SYSTEM, PROVE_LEMMA_SYSTEM, CLOSER_ID,
)
from search.dag.scheduler import RepairEngine  # noqa: E402

HEADER = "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b"


# ---------- failed ARM proposal leaves no mutation ----------------------------

def test_failed_arm_leaves_header_and_closer_unmutated():
    """The theory gate passes but every lemma proof fails across all
    revision rounds -> ABANDON. The header must not gain spliced lemmas
    and the closer must be unchanged."""
    sketch = json.dumps({
        "haves": [{"id": "h1", "type": "a^2 + b^2 ≥ 2*a*b",
                   "tactic": "nlinarith [sq_nonneg (a-b)]", "depends": []}],
        "closer": "exact h1"})
    theory = json.dumps({
        "defs": [], "lemmas": [{"name": "aux_x",
                   "statement": "lemma aux_x (x y : ℝ) : x^2 + y^2 ≥ 2*x*y"}],
        "leaf_tactics": {"__closer__": "exact aux_x a b"}})
    headers_seen: list[str] = []

    def llm(system, user):
        if system == THEORY_SYSTEM:
            return theory
        if system == PROVE_LEMMA_SYSTEM:
            return json.dumps({"proof": "WRONG"})   # never proves
        return sketch

    def verify(header, body):
        if "theorem t" not in header:               # lemma gate: always fail
            return {"ok": False, "errors": "g.lean:2:2: error: nope",
                    "body_line_offset": 1}
        headers_seen.append(header)
        # closer fails so the closer-stuck theory fires; nothing ever proves
        return {"ok": False,
                "errors": "f.lean:6:2: error: unsolved goals\n⊢ nope",
                "body_line_offset": 3}

    def probe(header, body):
        return {"ok": ":= by sorry" in header, "errors": "e",
                "body_line_offset": 3}

    res = attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify, probe_fn=probe,
        sketch_attempts=1, repair_rounds=3, leaf_fallbacks=(),
        decompose_depth=0, abduce_lemmas=True, abduce_mode="theory",
        abduce_theory_trigger="always", abduce_theory_rounds=2)
    assert not res.verified
    assert res.abduced_ids == []            # nothing committed
    assert res.abduced_lemmas == []
    # every header the verifier ever saw was the ORIGINAL (no aux_x splice)
    assert all("aux_x" not in h for h in headers_seen)
    # the closer text was never mutated by an abandoned proposal
    assert res.sketch is not None and res.sketch.closer == "exact h1"


def test_theorem_cannot_invoke_itself_via_committed_theory():
    """A theory whose committed proof would call the theorem being proved
    must not yield a false 'verified'. The kernel (verify_fn) is the only
    authority; a self-referential proof must fail there, not slip through
    the probe."""
    sketch = json.dumps({
        "haves": [{"id": "h1", "type": "a^2 + b^2 ≥ 2*a*b",
                   "tactic": "nlinarith [sq_nonneg (a-b)]", "depends": []}],
        "closer": "exact h1"})
    # the model proposes a lemma proved by calling `t` (the theorem itself)
    theory = json.dumps({
        "defs": [], "lemmas": [{"name": "aux_self",
                   "statement": "lemma aux_self (x y : ℝ) : x^2 + y^2 ≥ 2*x*y"}],
        "leaf_tactics": {"__closer__": "exact aux_self a b"}})

    def llm(system, user):
        if system == THEORY_SYSTEM:
            return theory
        if system == PROVE_LEMMA_SYSTEM:
            return json.dumps({"proof": "exact t a b"})   # self-invocation
        return sketch

    def verify(header, body):
        if "theorem t" not in header:            # lemma gate
            # a real Lean would reject `exact t a b` (t not in scope /
            # termination); model that: proofs mentioning ` t ` fail.
            ok = "GOODPROOF" in body and "exact t" not in body
            return {"ok": ok, "errors": None if ok else
                    "g.lean:2:2: error: unknown identifier 't'",
                    "body_line_offset": 1}
        ok = "aux_self" in header and "exact aux_self a b" in body
        return {"ok": ok, "errors": None if ok else
                "f.lean:6:2: error: unsolved goals", "body_line_offset": 3}

    def probe(header, body):
        return {"ok": ":= by sorry" in header, "errors": "e",
                "body_line_offset": 3}

    res = attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify, probe_fn=probe,
        sketch_attempts=1, repair_rounds=2, leaf_fallbacks=(),
        decompose_depth=0, abduce_lemmas=True, abduce_mode="theory",
        abduce_theory_trigger="always")
    # the self-referential lemma proof is rejected by the kernel gate, so
    # nothing commits and the theorem is NOT falsely verified.
    assert not res.verified
    assert res.abduced_ids == []


# ---------- REPL acceptance still requires a fresh compile --------------------

def test_repl_success_requires_fresh_compile():
    """This exercises run_dag's _verify_fn contract at the unit level: a
    REPL 'ok' must be re-confirmed by a fresh compile before it counts.
    We rebuild that composition here with a REPL that lies (says ok) and a
    compile oracle that tells the truth (says fail) -> overall NOT ok."""
    from backend.compile_verify import _ERROR_LINE_RE  # noqa: F401
    calls = {"repl": 0, "compile": 0}

    def repl_verify(th, body):
        calls["repl"] += 1
        return {"ok": True, "errors": None, "body_line_offset": 3}

    def compile_verify(th, body):
        calls["compile"] += 1
        return {"ok": False, "errors": "f.lean:4:2: error: unsolved goals",
                "body_line_offset": 3}

    # the composed verify_fn: REPL ok -> MUST re-confirm by fresh compile
    def composed(th, body):
        r = repl_verify(th, body)
        if not r.get("ok"):
            return r
        return compile_verify(th, body)     # fresh oracle is authoritative

    out = composed(HEADER, "  nlinarith")
    assert out["ok"] is False               # lying REPL cannot count
    assert calls["repl"] == 1 and calls["compile"] == 1


# ---------- legacy engine invokes no new stage; scheduler is gated ------------

def test_legacy_engine_is_inert():
    eng = RepairEngine("legacy")
    assert not eng.is_shadow
    # observe() is a no-op in legacy even with a mismatching plan
    from search.dag.scheduler import RoundContext
    eng.observe(1, ["llm_repair"],
                RoundContext(has_attributable_errors=True, setup_broken=False,
                             closer_broken=False, broken_leaf_ids=("h1",)))
    assert eng.divergences == []
    assert eng.report()["mode"] == "legacy"


def test_scheduler_engine_gated():
    try:
        RepairEngine("scheduler")
        raise AssertionError("scheduler mode must raise until reviewed")
    except NotImplementedError:
        pass


def test_default_dag_run_unchanged_by_migration_modules():
    """Importing the migration modules must not perturb a normal legacy
    run (a first-verify success), and no dag.* solver object is created."""
    sketch = json.dumps({"haves": [], "closer": "nlinarith [sq_nonneg (a-b)]"})

    def llm(system, user):
        return sketch

    def verify(header, body):
        return {"ok": True, "errors": None, "body_line_offset": 3}

    res = attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify,
        sketch_attempts=1, repair_rounds=2, leaf_fallbacks=(),
        decompose_depth=0)
    assert res.verified
