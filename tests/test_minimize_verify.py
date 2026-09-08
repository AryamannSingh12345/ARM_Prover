"""Point 7 — opt-in compile check after reference-based lemma minimisation.

When --abduce-minimize-verify is on and the textual prune drops a lemma,
the pruned set is probed; if the leaves no longer close, the full proved
set is restored instead of wasting the round. Off by default = unchanged.

Setup: the theory proposes two lemmas; the leaf tactic references only
`aux_use`, so the textual prune drops `aux_base`. `res.abduced_lemmas`
then contains aux_base ONLY when the run reverted to the full set.
"""
from __future__ import annotations

import json

from search import proof_dag


def _drive(*, minimize_verify: bool, probe_pruned_ok: bool):
    header = "theorem tgt (x : ℝ) (h : 0 < x) : x + x = 2 * x := by sorry"

    def fake_sketch(system, user):
        if system == proof_dag.SKETCH_SYSTEM:
            return json.dumps({
                "setup": [],
                "haves": [{"id": "h0", "type": "x + x = 2 * x",
                           "tactic": "nlinarith", "depends": []}],
                "closer": "exact h0"})
        if system == proof_dag.THEORY_SYSTEM:
            return json.dumps({
                "defs": [],
                "lemmas": [
                    {"name": "aux_base",
                     "statement": "lemma aux_base (y : ℝ) : y + y = 2 * y"},
                    {"name": "aux_use",
                     "statement": "lemma aux_use (z : ℝ) : z + z = 2 * z"}],
                "leaf_tactics": {"h0": "exact aux_use x"}})
        return json.dumps({"proof": "ring"})

    def fake_verify(hdr, body):
        if "aux_" in hdr and "sorry" not in body:
            return {"ok": True, "errors": None}   # a lemma proof compile
        return {"ok": False, "errors": "error: unsolved goals",
                "body_line_offset": 1}

    import re as _re

    def fake_probe(hdr, body):
        # Discriminate the two probe kinds by header CONTENT, not by the
        # theorem's own ':= by sorry': the satisfaction gate STUBS the
        # invented lemmas (`lemma aux_… := by sorry`); the post-minimise
        # probe splices their REAL proofs (`:= by ring`).
        gate = _re.search(r"lemma aux_\w+[^\n]*:= by sorry", hdr) is not None
        if gate:
            return {"ok": True, "errors": None}   # satisfaction gate passes
        if probe_pruned_ok:
            return {"ok": True, "errors": None}
        return {"ok": False, "errors": "error: unknown identifier",
                "body_line_offset": 1}

    return proof_dag.attempt_dag_proof(
        header, sketch_llm_call=fake_sketch, verify_fn=fake_verify,
        probe_fn=fake_probe, sketch_attempts=1, repair_rounds=1,
        abduce_lemmas=True, abduce_mode="theory", abduce_theory_rounds=0,
        abduce_theory_trigger="always", abduce_minimize=True,
        abduce_minimize_verify=minimize_verify)


def _has(res, name):
    return any(name in d for d in res.abduced_lemmas)


def test_default_off_keeps_pruned_set():
    # verify off → aux_base pruned and NOT restored (legacy behaviour).
    res = _drive(minimize_verify=False, probe_pruned_ok=False)
    assert _has(res, "aux_use")
    assert not _has(res, "aux_base")


def test_verify_reverts_when_pruned_set_fails():
    # verify on + pruned probe FAILS → revert to full set (aux_base back).
    res = _drive(minimize_verify=True, probe_pruned_ok=False)
    assert _has(res, "aux_use")
    assert _has(res, "aux_base")


def test_verify_keeps_prune_when_pruned_set_ok():
    # verify on + pruned probe OK → keep the smaller set (no needless revert).
    res = _drive(minimize_verify=True, probe_pruned_ok=True)
    assert _has(res, "aux_use")
    assert not _has(res, "aux_base")
