"""Regressions from the b5_bare_v1 autopsy (2026-07-26):

1. `strip_warning_blocks` — deprecation-warning spam is dropped from
   LLM-bound Lean output; error blocks and leading non-diagnostic text
   (bare timeouts) survive.
2. Setup-segment repairs — the repair contract's new "setup" field is
   applied when (and only when) the setup block was flagged broken.
3. `refresh_imports_call` — the ARM loop invokes the import-refresh
   hook on every theory proposal with the proposed declarations, AND
   (error-driven) when a lemma proof fails on an unknown identifier —
   the generic mechanism that replaced the (removed, hardcoded)
   geometry import rule.

Fake LLM / verify throughout — no Lean, no API.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.proof_dag import (  # noqa: E402
    attempt_dag_proof,
    strip_warning_blocks,
    _apply_repairs,
    Sketch,
    HaveNode,
    SETUP_ID,
    PROVE_LEMMA_SYSTEM,
    THEORY_SYSTEM,
)

HEADER = "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b"


# ---------- strip_warning_blocks ---------------------------------------------

WARN = (r"C:\x\Try_1.lean:19:43: warning: `Finset.toSet` has been "
        "deprecated: Use `SetLike.coe` instead\n"
        "Note: The updated constant has a different type:\n"
        "  {A : Type u_1} -> ...\n")
ERR1 = (r"C:\x\Try_1.lean:12:4: error: unsolved goals" "\n⊢ True\n")
ERR2 = (r"C:\x\Try_1.lean:15:2: error(lean.unknownIdentifier): "
        "Unknown identifier `foo`\n")


def test_strip_warnings_keeps_errors():
    out = strip_warning_blocks(WARN + ERR1 + WARN + ERR2)
    assert "deprecated" not in out
    assert "unsolved goals" in out
    assert "Unknown identifier" in out


def test_strip_warnings_keeps_leading_timeout_text():
    out = strip_warning_blocks("timeout after 600s\n" + WARN)
    assert out.startswith("timeout after 600s")
    assert "deprecated" not in out


def test_strip_warnings_passthrough_no_diagnostics():
    assert strip_warning_blocks("timeout after 600s") == "timeout after 600s"
    assert strip_warning_blocks("") == ""


def test_strip_warnings_warning_only_becomes_empty():
    assert strip_warning_blocks(WARN) == ""


# ---------- setup repairs -----------------------------------------------------

def test_apply_repairs_setup_field():
    sk = Sketch(haves=[HaveNode("h1", "True", "trivial", [])],
                closer="exact h1", setup=["bad_line"])
    raw = json.dumps({"repairs": [], "setup": ["classical", "intro x"]})
    applied, err = _apply_repairs(sk, raw, set(), False, True)
    assert err is None and SETUP_ID in applied
    assert sk.setup == ["classical", "intro x"]


def test_apply_repairs_setup_ignored_when_not_broken():
    sk = Sketch(haves=[HaveNode("h1", "True", "trivial", [])],
                closer="exact h1", setup=["good_line"])
    raw = json.dumps({"repairs": [], "setup": ["evil"]})
    applied, err = _apply_repairs(sk, raw, set(), False, False)
    assert sk.setup == ["good_line"]
    assert err == "repair_applied_nothing"


def test_apply_repairs_setup_rejects_forbidden():
    sk = Sketch(haves=[HaveNode("h1", "True", "trivial", [])],
                closer="exact h1", setup=["x"])
    raw = json.dumps({"repairs": [], "setup": ["sorry"]})
    applied, err = _apply_repairs(sk, raw, set(), False, True)
    assert sk.setup == ["x"] and err == "repair_applied_nothing"


def test_setup_error_repaired_end_to_end():
    """Verify fails with an error line-mapped to the setup block; the
    repair supplies a new setup; the reverify (which sees the new
    setup) passes."""
    sketch = json.dumps({
        "setup": ["exact absurd"],
        "haves": [{"id": "h1", "type": "a^2 + b^2 ≥ 2*a*b",
                   "tactic": "nlinarith [sq_nonneg (a-b)]",
                   "depends": []}],
        "closer": "exact h1",
    })
    repair = json.dumps({"repairs": [], "setup": ["classical"]})
    calls = {"n": 0}

    def llm(system, user):
        return repair if "BROKEN setup block" in user else sketch

    def verify(header, body):
        calls["n"] += 1
        if "classical" in body:
            return {"ok": True, "errors": None, "body_line_offset": 3}
        # offset 3 → file line 4 = body line 1 = the setup line
        return {"ok": False,
                "errors": "f.lean:4:2: error: unknown identifier absurd",
                "body_line_offset": 3}

    res = attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify,
        sketch_attempts=1, repair_rounds=2,
        leaf_fallbacks=(), decompose_depth=0)
    assert res.verified
    assert SETUP_ID in res.repaired_ids
    assert "classical" in (res.assembled_proof or "")


# ---------- ARM import-refresh hook ------------------------------------------

BAD_SKETCH = json.dumps({
    "haves": [{"id": "h1", "type": "a^2 + b^2 ≥ 2*a*b",
               "tactic": "positivity", "depends": []}],
    "closer": "exact h1",
})

THEORY = json.dumps({
    "defs": [],
    # Generalized statement (fresh variables) — literal restatements
    # of a stuck goal are circular-rejected.
    "lemmas": [{"name": "aux_bridge",
                "statement": "lemma aux_bridge (x y : ℝ) : "
                             "x^2 + y^2 ≥ 2*x*y"}],
    "leaf_tactics": {"h1": "nlinarith [aux_bridge a b]"},
})


def test_refresh_imports_called_per_theory_proposal():
    seen: list[str] = []

    def llm(system, user):
        if system == THEORY_SYSTEM:
            return THEORY
        if system == PROVE_LEMMA_SYSTEM:
            return json.dumps({"proof": "GOODPROOF"})
        return BAD_SKETCH

    def verify(header, body):
        if "theorem t" not in header:  # lemma gate
            return {"ok": "GOODPROOF" in body, "errors": "e",
                    "body_line_offset": 1}
        ok = "aux_bridge" in header and "aux_bridge" in body
        return {"ok": ok,
                "errors": None if ok else
                "f.lean:5:4: error: timeout: heartbeats",
                "body_line_offset": 3}

    def probe(header, body):
        return {"ok": ":= by sorry" in header, "errors": "e",
                "body_line_offset": 3}

    res = attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify,
        sketch_attempts=1, repair_rounds=3,
        leaf_fallbacks=(), decompose_depth=0,
        abduce_lemmas=True, abduce_mode="theory",
        abduce_theory_trigger="always", probe_fn=probe,
        refresh_imports_call=lambda decls: seen.append(decls) or "+1")
    assert res.verified
    assert len(seen) == 1  # one proposal → one refresh ask
    assert "aux_bridge" in seen[0]


def test_refresh_imports_exception_is_soft():
    def llm(system, user):
        if system == THEORY_SYSTEM:
            return THEORY
        if system == PROVE_LEMMA_SYSTEM:
            return json.dumps({"proof": "GOODPROOF"})
        return BAD_SKETCH

    def verify(header, body):
        if "theorem t" not in header:
            return {"ok": "GOODPROOF" in body, "errors": "e",
                    "body_line_offset": 1}
        ok = "aux_bridge" in header and "aux_bridge" in body
        return {"ok": ok,
                "errors": None if ok else
                "f.lean:5:4: error: timeout: heartbeats",
                "body_line_offset": 3}

    def probe(header, body):
        return {"ok": ":= by sorry" in header, "errors": "e",
                "body_line_offset": 3}

    def boom(_decls):
        raise RuntimeError("refresh exploded")

    res = attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify,
        sketch_attempts=1, repair_rounds=3,
        leaf_fallbacks=(), decompose_depth=0,
        abduce_lemmas=True, abduce_mode="theory",
        abduce_theory_trigger="always", probe_fn=probe,
        refresh_imports_call=boom)
    assert res.verified  # the hook must never kill the loop


# ---------- error-driven import refresh --------------------------------------

def test_refresh_imports_fires_on_unknown_identifier_in_proof():
    """A lemma proof failing on `Unknown identifier` must hand the
    statement + error to the import hook (once), and the retry can
    then succeed — the generic replacement for hardcoded import
    rules. The fake verify accepts the proof only after the hook has
    been consulted, simulating a missing-module fix."""
    seen: list[str] = []
    state = {"imports_fixed": False}

    def refresh(text):
        seen.append(text)
        # Only the error-driven ask (statement + Lean error) finds the
        # missing module; the proposal-time ask changes nothing.
        if "Unknown identifier" in text:
            state["imports_fixed"] = True
            return "+1: import Mathlib.Fake"
        return None

    def llm(system, user):
        if system == THEORY_SYSTEM:
            return THEORY
        if system == PROVE_LEMMA_SYSTEM:
            return json.dumps({"proof": "exact needs_module"})
        return BAD_SKETCH

    def verify(header, body):
        if "theorem t" not in header:  # lemma gate
            if state["imports_fixed"]:
                return {"ok": True, "errors": None,
                        "body_line_offset": 1}
            return {"ok": False,
                    "errors": ("g.lean:2:8: error(lean.unknownIdentifier"
                               "): Unknown identifier `needs_module`"),
                    "body_line_offset": 1}
        ok = "aux_bridge" in header and "aux_bridge" in body
        return {"ok": ok,
                "errors": None if ok else
                "f.lean:5:4: error: timeout: heartbeats",
                "body_line_offset": 3}

    def probe(header, body):
        return {"ok": ":= by sorry" in header, "errors": "e",
                "body_line_offset": 3}

    res = attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify,
        sketch_attempts=1, repair_rounds=3,
        leaf_fallbacks=(), decompose_depth=0,
        abduce_lemmas=True, abduce_mode="theory",
        abduce_theory_trigger="always", probe_fn=probe,
        refresh_imports_call=refresh)
    assert res.verified
    # Called once for the proposal, once error-driven for the failed
    # proof; the error-driven text carries statement AND Lean error.
    assert len(seen) == 2
    assert "Unknown identifier" in seen[1]
    assert "aux_bridge" in seen[1]


def test_refresh_imports_fires_from_main_repair_loop():
    """b5_bare_v2 regression: an unknown identifier in the CLOSER (not
    a prove-lemma gate) must also reach the import hook. The fake
    verify passes once the hook has 'fixed' imports and the repair has
    supplied a new closer."""
    seen: list[str] = []
    state = {"imports_fixed": False}

    def refresh(text):
        seen.append(text)
        if "Unknown identifier" in text:
            state["imports_fixed"] = True
            return "+1: import Mathlib.Fake"
        return None

    sketch = json.dumps({
        "haves": [{"id": "h1", "type": "a^2 + b^2 ≥ 2*a*b",
                   "tactic": "nlinarith [sq_nonneg (a-b)]",
                   "depends": []}],
        "closer": "exact h1",
    })
    repair = json.dumps({"repairs": [], "closer": "exact fixed_h1"})

    def llm(system, user):
        return repair if "BROKEN closer" in user else sketch

    def verify(header, body):
        if state["imports_fixed"] and "fixed_h1" in body:
            return {"ok": True, "errors": None, "body_line_offset": 3}
        # the have takes body lines 1-2, closer is line 3 → file line 6
        return {"ok": False,
                "errors": ("f.lean:6:2: error(lean.unknownIdentifier): "
                           "Unknown identifier `dist_add_dist_eq_iff`"),
                "body_line_offset": 3}

    res = attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify,
        sketch_attempts=1, repair_rounds=2,
        leaf_fallbacks=(), decompose_depth=0,
        refresh_imports_call=refresh)
    assert res.verified
    assert len(seen) == 1  # dedup: same error set asked once
    assert "Unknown identifier" in seen[0]


def test_theory_fires_on_closer_stuck():
    """b5_bare_v2 regression: trivial leaves all verify, the closer
    carries the mathematics and keeps failing — the theory loop must
    fire on the closer posed as a pseudo-leaf, and its committed
    CLOSER_ID tactic must replace the closer."""
    sketch = json.dumps({
        "haves": [{"id": "h1", "type": "a^2 + b^2 ≥ 2*a*b",
                   "tactic": "nlinarith [sq_nonneg (a-b)]",
                   "depends": []}],
        "closer": "exact h1",
    })
    closer_theory = json.dumps({
        "defs": [],
        "lemmas": [{"name": "aux_bridge",
                    "statement": "lemma aux_bridge (x y : ℝ) : "
                                 "x^2 + y^2 ≥ 2*x*y"}],
        "leaf_tactics": {"__closer__": "exact aux_bridge a b"},
    })

    def llm(system, user):
        if system == THEORY_SYSTEM:
            # The stuck obligation must be the closer's pseudo-leaf.
            assert "leaf `__closer__`" in user
            assert "leaf___closer__" in user  # pseudo-leaf statement
            return closer_theory
        if system == PROVE_LEMMA_SYSTEM:
            return json.dumps({"proof": "GOODPROOF"})
        return sketch  # sketch + any repair call

    def verify(header, body):
        if "theorem t" not in header:  # lemma gate
            return {"ok": "GOODPROOF" in body, "errors": "e",
                    "body_line_offset": 1}
        ok = "aux_bridge" in header and "exact aux_bridge a b" in body
        # the have takes body lines 1-2, closer is line 3 → file line 6
        return {"ok": ok,
                "errors": None if ok else
                "f.lean:6:2: error: unsolved goals\n⊢ nope",
                "body_line_offset": 3}

    def probe(header, body):
        return {"ok": ":= by sorry" in header
                and "exact aux_bridge a b" in body,
                "errors": "e", "body_line_offset": 3}

    res = attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify,
        sketch_attempts=1, repair_rounds=3,
        leaf_fallbacks=(), decompose_depth=0,
        abduce_lemmas=True, abduce_mode="theory",
        abduce_theory_trigger="always", probe_fn=probe)
    assert res.verified
    assert "__closer__" in res.abduced_ids
    assert "aux_bridge" in "\n".join(res.abduced_lemmas)
    assert "exact aux_bridge a b" in (res.assembled_proof or "")


def test_error_messages_not_glued_with_warnings():
    """b5_bare_v3 regression: parse_error_locations sliced each error
    message up to the next ERROR marker, gluing intervening warning
    blocks (deprecation spam) into attributed messages."""
    from search.proof_dag import parse_error_locations
    locs = parse_error_locations(ERR1 + WARN + ERR2)
    assert [line for line, _ in locs] == [12, 15]
    assert "deprecated" not in locs[0][1]
    assert "unsolved goals" in locs[0][1]


def test_theory_circular_proposal_rejected_then_revised():
    """b5_bare_v3 regression: all six theory proposals RESTATED the
    theorem as a lemma — gate passes trivially, prove faces the
    original problem. Restatements are now rejected before any
    compile, with the reason in the revision ledger; a genuine
    generalization on revision goes through."""
    sketch = json.dumps({
        "haves": [{"id": "h1", "type": "a^2 + b^2 ≥ 2*a*b",
                   "tactic": "nlinarith [sq_nonneg (a-b)]",
                   "depends": []}],
        "closer": "exact h1",
    })
    circular_theory = json.dumps({
        "defs": [],
        "lemmas": [{"name": "aux_restate",
                    "statement": "lemma aux_restate (a b : ℝ) "
                                 "(h_extra : 0 ≤ a) : "
                                 "a^2 + b^2 ≥ 2*a*b"}],
        "leaf_tactics": {"__closer__": "exact aux_restate a b h"},
    })
    good_theory = json.dumps({
        "defs": [],
        "lemmas": [{"name": "aux_bridge",
                    "statement": "lemma aux_bridge (x y : ℝ) : "
                                 "x^2 + y^2 ≥ 2*x*y"}],
        "leaf_tactics": {"__closer__": "exact aux_bridge a b"},
    })
    calls = {"theory": 0, "probes": 0}

    def llm(system, user):
        if system == THEORY_SYSTEM:
            calls["theory"] += 1
            if calls["theory"] == 1:
                return circular_theory
            # The revision prompt must carry the circularity verdict.
            assert "circular" in user.lower()
            return good_theory
        if system == PROVE_LEMMA_SYSTEM:
            return json.dumps({"proof": "GOODPROOF"})
        return sketch

    def verify(header, body):
        if "theorem t" not in header:
            return {"ok": "GOODPROOF" in body, "errors": "e",
                    "body_line_offset": 1}
        ok = "aux_bridge" in header and "exact aux_bridge a b" in body
        return {"ok": ok,
                "errors": None if ok else
                "f.lean:6:2: error: unsolved goals\n⊢ nope",
                "body_line_offset": 3}

    def probe(header, body):
        calls["probes"] += 1
        return {"ok": ":= by sorry" in header
                and "exact aux_bridge a b" in body,
                "errors": "e", "body_line_offset": 3}

    res = attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify,
        sketch_attempts=1, repair_rounds=3,
        leaf_fallbacks=(), decompose_depth=0,
        abduce_lemmas=True, abduce_mode="theory",
        abduce_theory_trigger="always", probe_fn=probe)
    assert res.verified
    assert calls["theory"] == 2
    assert any(e.startswith("theory_circular") for e in res.repair_errors)
    # The circular round must burn ZERO compiles (no gate probe).
    assert calls["probes"] == 1


def test_no_hardcoded_geometry_import_rule():
    """Guard: the benchmark-tuned plane_geometry rule stays deleted —
    proof-layer import gaps go through the generic LLM refresh, never
    a module list keyed to problem vocabulary."""
    from search.import_inference import infer_imports_from_header
    header = ("theorem putnam_1966_b5 (S : Finset (EuclideanSpace ℝ "
              "(Fin 2))) (hS : ∀ s ⊆ S, s.card = 3 → ¬Collinear ℝ "
              "s.toSet) : ∃ L, ∀ I ∈ segment ℝ (L 0) (L 1), I = L 0 "
              ":= by sorry")
    r = infer_imports_from_header(header)
    assert "plane_geometry" not in r.matched_rules
