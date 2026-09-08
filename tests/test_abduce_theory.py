"""Theory-mode (deferred) abduction tests: statements-only proposal →
sorry-stubbed satisfaction probe → per-lemma proving (with a cross-round
proved bank) → reference-based post-prove pruning → commit; revision on
failure with a full status ledger. Fake LLM / verify / probe throughout
— no Lean, no API.

Includes regressions for every bug the live arm_engel3 campaign found:
named-diagnostic error locations, leading-`by` proofs, prune-before-
prove, bare-timeout trigger starvation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.proof_dag import (  # noqa: E402
    attempt_dag_proof,
    parse_error_locations,
    _parse_theory,
    _strip_leading_by,
    PROVE_LEMMA_SYSTEM,
    THEORY_SYSTEM,
)

HEADER = "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b"

BAD_SKETCH = json.dumps({
    "haves": [{"id": "h1", "type": "a^2 + b^2 ≥ 2*a*b",
               "tactic": "positivity", "depends": []}],
    "closer": "exact h1",
})

THEORY = json.dumps({
    "defs": [],
    "lemmas": [
        # A GENERALIZATION of the stuck goal (fresh variables) — the
        # circularity guard rejects literal restatements, and that is
        # exactly the distinction the engel3 campaign validated.
        {"name": "aux_bridge",
         "statement": "lemma aux_bridge (x y : ℝ) : x^2 + y^2 ≥ 2*x*y"},
        {"name": "aux_extra",
         "statement": "lemma aux_extra (x : ℝ) : x^2 ≥ 0"},
    ],
    # References aux_bridge textually — the post-prove pruner keeps
    # referenced lemmas and drops the rest.
    "leaf_tactics": {"h1": "nlinarith [aux_bridge a b]"},
    "rationale": "bridge the gap",
})

# Error that both line-maps to h1 (body line 2 at offset 3 → file line 5)
# and contains "timeout" so the stuck trigger fires immediately.
TIMEOUT_ERR = "f.lean:5:4: error: timeout: heartbeat exceeded"


def _fixture(theory_responses, proof_responses, main_fail_err=TIMEOUT_ERR):
    """Build (llm, verify, probe, counters) fakes.

    - main verify: ok iff the committed lemma is in the header (no
      sorry stubs) AND the new leaf tactic (referencing aux_bridge) is
      in the body; else `main_fail_err`.
    - lemma-proof gate: any header without `theorem t` is a standalone
      lemma header; ok iff body contains GOODPROOF.
    - probe: ok iff the aux_bridge sorry-stub is present.
    """
    counters = {"theory_calls": 0, "prove_calls": 0, "probe_calls": 0,
                "revise_seen": False}

    def llm(system, user):
        if system == THEORY_SYSTEM:
            if "did not fully succeed" in user:
                counters["revise_seen"] = True
            resp = theory_responses[
                min(counters["theory_calls"], len(theory_responses) - 1)]
            counters["theory_calls"] += 1
            return resp
        if system == PROVE_LEMMA_SYSTEM:
            resp = proof_responses[
                min(counters["prove_calls"], len(proof_responses) - 1)]
            counters["prove_calls"] += 1
            return resp
        return BAD_SKETCH  # sketch call

    def verify(header, body):
        if "theorem t" not in header:  # standalone lemma-proof gate
            ok = "GOODPROOF" in body
            return {"ok": ok,
                    "errors": None if ok else "g.lean:2:2: error: nope",
                    "body_line_offset": 1}
        ok = ("aux_bridge" in header and ":= by sorry" not in header
              and "aux_bridge" in body)
        return {"ok": ok, "errors": None if ok else main_fail_err,
                "body_line_offset": 3}

    def probe(header, body):
        counters["probe_calls"] += 1
        ok = "aux_bridge" in header and ":= by sorry" in header
        return {"ok": ok,
                "errors": None if ok else "f.lean:1:0: error: unknown "
                                          "identifier aux_bridge",
                "body_line_offset": 3}

    return llm, verify, probe, counters


def _run(llm, verify, probe, rounds=2, trigger="always"):
    return attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify,
        sketch_attempts=1, repair_rounds=3,
        leaf_fallbacks=(), decompose_depth=0,
        abduce_lemmas=True, abduce_mode="theory",
        probe_fn=probe, abduce_theory_rounds=rounds,
        abduce_theory_trigger=trigger,
    )


def test_theory_happy_path_prove_then_prune():
    llm, verify, probe, c = _fixture(
        theory_responses=[THEORY],
        proof_responses=[json.dumps({"proof": "GOODPROOF"})])
    res = _run(llm, verify, probe)
    assert res.verified
    assert "h1" in res.abduced_ids
    joined = "\n".join(res.abduced_lemmas)
    assert "aux_bridge" in joined and "GOODPROOF" in joined
    assert ":= by sorry" not in joined  # committed lemmas carry proofs
    # Prune-AFTER-prove: both lemmas get proved (no satisfaction-driven
    # elimination), then the unreferenced one is dropped from commit.
    assert c["prove_calls"] == 2
    assert "aux_extra" not in joined
    assert c["probe_calls"] == 1  # gate only — no elimination probes


def test_theory_gate_failure_triggers_revision():
    bad_theory = json.dumps({
        "defs": [],
        "lemmas": [{"name": "aux_bad",
                    "statement": "lemma aux_bad (x : ℝ) : x = x + 1"}],
        "leaf_tactics": {"h1": "nlinarith [aux_bad 0]"},
    })
    # aux_bad's probe fails (no aux_bridge stub) → gate failure →
    # revision; the revised theory is the good one.
    llm, verify, probe, c = _fixture(
        theory_responses=[bad_theory, THEORY],
        proof_responses=[json.dumps({"proof": "GOODPROOF"})])
    res = _run(llm, verify, probe)
    assert res.verified
    assert c["theory_calls"] == 2
    assert c["revise_seen"]
    assert any("theory_gate_failed" in e for e in res.repair_errors)


def test_theory_unproved_lemma_bank_survives_revision():
    # Round 0: aux_bridge fails twice, aux_extra proves → ledger goes
    # back, revision proposes the same theory → aux_extra comes from
    # the BANK (no re-prove), aux_bridge proves on the fresh call.
    llm, verify, probe, c = _fixture(
        theory_responses=[THEORY, THEORY],
        proof_responses=[
            json.dumps({"proof": "WRONG"}),      # bridge, attempt 1
            json.dumps({"proof": "WRONG"}),      # bridge, attempt 2
            json.dumps({"proof": "GOODPROOF"}),  # extra, round 0
            json.dumps({"proof": "GOODPROOF"}),  # bridge, round 1
        ])
    res = _run(llm, verify, probe)
    assert res.verified
    assert c["theory_calls"] == 2
    # 3 fresh calls in round 0 + 1 in round 1; aux_extra NOT re-proved.
    assert c["prove_calls"] == 4
    assert any("theory_lemma_unproved" in e for e in res.repair_errors)


def test_theory_exhaustion_is_soft():
    llm, verify, probe, c = _fixture(
        theory_responses=[json.dumps({
            "defs": [], "lemmas": [{"name": "aux_bad",
                                    "statement": "lemma aux_bad : True"}],
            "leaf_tactics": {"h1": "nlinarith [aux_bridge 0 0]"}})],
        proof_responses=[json.dumps({"proof": "WRONG"})])
    res = _run(llm, verify, probe, rounds=1)
    assert not res.verified
    assert res.failure_stage == "verify"
    assert any("theory" in e for e in res.repair_errors)


def test_bare_timeout_triggers_theory():
    # Regression (engel3 pathology): a verify failure with NO parseable
    # error locations must still hand all leaves to the theory loop —
    # even under the default "stuck" trigger.
    llm, verify, probe, c = _fixture(
        theory_responses=[THEORY],
        proof_responses=[json.dumps({"proof": "GOODPROOF"})],
        main_fail_err="timeout after 600s")
    res = _run(llm, verify, probe, trigger="stuck")
    assert res.verified
    assert "h1" in res.abduced_ids


def test_proof_with_leading_by_still_proves():
    # Regression: model proofs prefixed with `by` used to become
    # `by by` parse errors; they are now normalized.
    llm, verify, probe, c = _fixture(
        theory_responses=[THEORY],
        proof_responses=[json.dumps({"proof": "by\n  GOODPROOF"})])
    res = _run(llm, verify, probe)
    assert res.verified
    assert "GOODPROOF" in "\n".join(res.abduced_lemmas)


def test_normalize_proof_indent():
    from search.proof_dag import _normalize_proof_indent as norm
    # The arm_engel3_final1 killer: inline-complete `by linarith` with
    # deeper continuation lines → continuations dedented to top level.
    bad = "have hd : P := by linarith\n rw [foo]\n nlinarith [bar]"
    assert norm(bad) == "have hd : P := by linarith\nrw [foo]\nnlinarith [bar]"
    # Genuine sub-block (first line ends in bare `by`) is untouched.
    sub = "have hd : P := by\n  linarith\nrw [foo]"
    assert norm(sub) == sub
    # Mixed depths reaching column 0 are untouched.
    mixed = "have hd : P := by linarith\nrw [foo]\n  · simp"
    assert norm(mixed) == mixed
    # Single-line proofs untouched.
    assert norm("nlinarith") == "nlinarith"


def test_strip_leading_by():
    assert _strip_leading_by("by\n  rw [foo]\n  ring") == "rw [foo]\nring"
    assert _strip_leading_by("by nlinarith") == "nlinarith"
    assert _strip_leading_by("nlinarith") == "nlinarith"
    assert _strip_leading_by("by") == ""
    # `by_contra` must NOT be mangled
    assert _strip_leading_by("by_contra h") == "by_contra h"


def test_error_locations_match_named_diagnostics():
    # Regression: `error(lean.unknownIdentifier):` was invisible to
    # attribution (and to the probe's error detection).
    errs = (r"C:\x\Try_1.lean:18:6: error(lean.unknownIdentifier): "
            "Unknown identifier `div_le_div_iff` "
            "\nC:\\x\\Try_1.lean:7:65: error: unsolved goals")
    locs = parse_error_locations(errs)
    assert [line for line, _ in locs] == [18, 7]
    assert "div_le_div_iff" in locs[0][1]


def test_parse_theory_contract():
    ok, err = _parse_theory(THEORY)
    assert err is None
    defs, lemmas, tactics = ok
    assert len(lemmas) == 2
    assert tactics["h1"] == "nlinarith [aux_bridge a b]"
    # proof tails are truncated from statements
    got, err2 = _parse_theory(json.dumps({
        "lemmas": [{"name": "a",
                    "statement": "lemma a : True := by trivial"}],
        "leaf_tactics": {"h1": "exact a"}}))
    assert err2 is None
    assert got[1][0][1] == "lemma a : True"
    # leading `by` in a leaf tactic is normalized
    got3, err3 = _parse_theory(json.dumps({
        "lemmas": [{"name": "a", "statement": "lemma a : True"}],
        "leaf_tactics": {"h1": "by\n  exact a"}}))
    assert err3 is None and got3[2]["h1"] == "exact a"
    # sorry in a leaf tactic is rejected
    _, err4 = _parse_theory(json.dumps({
        "lemmas": [{"name": "a", "statement": "lemma a : True"}],
        "leaf_tactics": {"h1": "sorry"}}))
    assert err4 == "theory_tactic_forbidden"


# ---- a timed-out satisfaction probe must not read as "satisfied" -----------

def test_probe_timeout_is_not_a_passing_gate():
    """A timeout carries no `file:line:col:` markers, so error
    attribution blames no segment and the "errors are outside our
    leaves" branch used to return True — reading "we learned nothing" as
    "the theory is satisfied".

    Measured on p2020a2_recognize_0805: a 600s probe timeout was gated
    OK, and the run then spent a 103s LLM call and a further 600s
    compile proving a theory that had never been validated.
    """
    llm, verify, probe, c = _fixture(
        theory_responses=[THEORY],
        proof_responses=[json.dumps({"proof": "GOODPROOF"})])

    def timing_out_probe(header, body):
        c["probe_calls"] += 1
        return {"ok": False, "errors": "timeout after 600s",
                "body_line_offset": 3}

    res = _run(llm, verify, timing_out_probe)
    assert c["probe_calls"] >= 1
    # The theory was never validated, so nothing may be committed on it.
    assert not res.abduced_ids
    assert c["prove_calls"] == 0, (
        "proving a theory whose probe timed out spends the most "
        "expensive calls in the loop on an unvalidated theory")


def test_timeout_detector_ignores_ordinary_errors():
    from search.proof_dag import _is_timeout_error
    assert _is_timeout_error("timeout after 600s")
    assert _is_timeout_error("  Timeout after 900s ")
    assert not _is_timeout_error("f.lean:5:4: error: unsolved goals")
    assert not _is_timeout_error("")


# ---- a resubmitted proof must not be recompiled -----------------------------

def _run_traced(llm, verify, probe, events, rounds=0):
    return attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify,
        sketch_attempts=1, repair_rounds=3,
        leaf_fallbacks=(), decompose_depth=0,
        abduce_lemmas=True, abduce_mode="theory",
        probe_fn=probe, abduce_theory_rounds=rounds,
        abduce_theory_trigger="always",
        trace=lambda kind, **kw: events.append((kind, kw)))


def _lemma_gate_counter(verify):
    """Count only STANDALONE lemma compiles (the expensive ones)."""
    n = {"v": 0}

    def counting(header, body):
        if "theorem t" not in header:
            n["v"] += 1
        return verify(header, body)
    return counting, n


def test_identical_proof_is_not_compiled_twice():
    """An identical proof cannot earn a different verdict, so the compile
    is pure waste. Measured on p2020a2_recognize_0805:
    `aux_weightedChooseSum_succ` received the SAME proof three times and
    paid three compiles (229s + 283s + 331s) for one bit of information.

    THEORY carries two lemmas, so the floor is one compile EACH — the
    guard keys on the proof text per lemma, not globally.
    """
    same = json.dumps({"proof": "BADPROOF"})
    llm, verify, probe, c = _fixture(
        theory_responses=[THEORY], proof_responses=[same] * 6)
    counting, n = _lemma_gate_counter(verify)
    events = []
    _run_traced(llm, counting, probe, events)
    assert c["prove_calls"] > n["v"], "repeats must be asked but not compiled"
    assert n["v"] == 2, f"expected one compile per lemma, got {n['v']}"
    assert any(kw.get("stage") == "duplicate proof skipped"
               for _k, kw in events)


def test_repeat_is_told_it_is_a_repeat():
    """The previous error alone evidently read as 'try again' rather than
    'try something else', so the feedback must say so explicitly."""
    same = json.dumps({"proof": "BADPROOF"})
    llm, verify, probe, _c = _fixture(
        theory_responses=[THEORY], proof_responses=[same] * 6)
    events = []
    _run_traced(llm, verify, probe, events)
    dups = [kw for _k, kw in events
            if kw.get("stage") == "duplicate proof skipped"]
    assert dups, "the repeat was never detected"
    assert dups[0].get("detail")          # names the lemma it applied to


def test_a_different_proof_is_still_compiled():
    """The guard must key on the PROOF TEXT, never on the lemma."""
    llm, verify, probe, _c = _fixture(
        theory_responses=[THEORY],
        proof_responses=[json.dumps({"proof": "FIRST"}),
                         json.dumps({"proof": "SECOND"}),
                         json.dumps({"proof": "THIRD"}),
                         json.dumps({"proof": "FOURTH"})])
    counting, n = _lemma_gate_counter(verify)
    events = []
    _run_traced(llm, counting, probe, events)
    assert not any(kw.get("stage") == "duplicate proof skipped"
                   for _k, kw in events)
    assert n["v"] == 4, f"every distinct proof must be compiled, got {n['v']}"


def test_import_refresh_clears_the_duplicate_guard():
    """An identical proof CAN earn a different verdict — if the
    environment moved under it.

    `test_refresh_imports_fires_on_unknown_identifier_in_proof` is the
    real case: the model resubmits `exact needs_module` unchanged and it
    compiles the second time because the import hook supplied the missing
    module in between. That is the entire point of refreshing imports, so
    the guard must forget what it saw whenever imports move — otherwise
    this optimisation silently disables that recovery path.
    """
    seen: list[str] = []
    state = {"fixed": False}

    def refresh(text):
        seen.append(text)
        if "Unknown identifier" in text:
            state["fixed"] = True
            return "+1: import Mathlib.Fake"
        return None

    def llm(system, user):
        if system == THEORY_SYSTEM:
            return THEORY
        if system == PROVE_LEMMA_SYSTEM:
            return json.dumps({"proof": "exact needs_module"})
        return BAD_SKETCH

    def verify(header, body):
        if "theorem t" not in header:
            if state["fixed"]:
                return {"ok": True, "errors": None, "body_line_offset": 1}
            return {"ok": False,
                    "errors": ("g.lean:2:8: error(lean.unknownIdentifier"
                               "): Unknown identifier `needs_module`"),
                    "body_line_offset": 1}
        ok = "aux_bridge" in header and "aux_bridge" in body
        return {"ok": ok,
                "errors": None if ok else "f.lean:5:4: error: timeout",
                "body_line_offset": 3}

    def probe(header, body):
        return {"ok": ":= by sorry" in header, "errors": "e",
                "body_line_offset": 3}

    res = attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify,
        sketch_attempts=1, repair_rounds=3, leaf_fallbacks=(),
        decompose_depth=0, abduce_lemmas=True, abduce_mode="theory",
        abduce_theory_trigger="always", probe_fn=probe,
        refresh_imports_call=refresh)
    assert res.verified, "the identical retry must still be allowed to run"
