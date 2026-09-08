"""Diagnostic-trace tests: Tracer file format, unicode safety, and the
trace events emitted by attempt_dag_proof (fake LLM + verifier, no
Lean). Also the Sample.thinking plumbing contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diag.tracer import Tracer, render_event  # noqa: E402
from search.proof_dag import attempt_dag_proof  # noqa: E402
from policy.vllm_policy import Sample  # noqa: E402


GOOD_SKETCH = json.dumps({
    "haves": [
        {"id": "h1", "type": "0 ≤ (a - b)^2", "tactic": "positivity",
         "depends": []},
    ],
    "closer": "nlinarith [h1]",
})


def _events(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in
            path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_tracer_writes_valid_jsonl(tmp_path):
    tr = Tracer(tmp_path / "t.trace.jsonl", problem_id="p", echo="quiet")
    tr.event("problem_start", header="theorem foo : 1 ≤ 2")
    tr.event("llm_call", role="sketch", response="{}", thinking="ℝ ≥ 0 …")
    evs = _events(tmp_path / "t.trace.jsonl")
    assert [e["seq"] for e in evs] == [1, 2]
    assert evs[0]["kind"] == "problem_start"
    assert evs[1]["thinking"] == "ℝ ≥ 0 …"
    assert all("ts" in e and "t" in e for e in evs)


def test_tracer_survives_unwritable_path(capsys):
    # A bad path must not raise — tracing is never load-bearing.
    tr = Tracer(Path("Z:/definitely/not/a/dir/x.jsonl"),
                problem_id="p", echo="quiet")
    tr.event("outcome", verified=True, wall_s=1.0)


def test_render_event_handles_all_known_kinds():
    kinds = [
        {"kind": "problem_start", "id": "p", "header": "theorem x : True"},
        {"kind": "imports_resolved", "source": "cli", "imports": "import Mathlib"},
        {"kind": "premises", "strategy": "bm25", "names": ["sq_nonneg"]},
        {"kind": "fallback_ladder", "source": "static", "tactics": ["ring"]},
        {"kind": "llm_call", "role": "sketch", "model": "m",
         "duration_s": 1.0, "response": "{}", "thinking": "hmm"},
        {"kind": "sketch_attempt", "attempt": 1, "of": 3, "mode": "fresh"},
        {"kind": "sketch_parsed", "n_haves": 1, "ids": ["h1"],
         "haves": [{"id": "h1", "type": "T", "tactic": "ring"}],
         "setup": [], "closer": "ring"},
        {"kind": "sketch_parse_error", "error": "no_json_object_found"},
        {"kind": "topo_error", "error": "cycle_in_have_graph"},
        {"kind": "assembled", "round": 0, "body_lines": 3,
         "segments": [["h1", 1, 2]]},
        {"kind": "verify_call", "ok": False, "backend": "compile",
         "duration_s": 100.0, "errors": "x.lean:5:2: error: nope"},
        {"kind": "error_attribution", "by_segment": {"h1": ["nope"]}},
        {"kind": "repair_round", "round": 1, "broken": ["h1"],
         "closer_broken": False},
        {"kind": "leaf_closer", "id": "h1", "fixed": True, "tactic": "omega"},
        {"kind": "abduce", "id": "h1", "status": "ACCEPTED (kernel-gated)",
         "lemmas": ["lemma aux_x : True := trivial"]},
        {"kind": "decompose", "id": "h1", "ok": True},
        {"kind": "repair_applied", "ids": ["h1"]},
        {"kind": "repair_break", "reason": "empty_repair_response"},
        {"kind": "outcome", "verified": False, "stage": "verify",
         "detail": "err", "wall_s": 9.0},
        {"kind": "brand_new_kind", "whatever": 1},  # forward compat
    ]
    for ev in kinds:
        ev["t"] = 1.0
        lines = render_event(ev, "verbose")
        assert lines, f"no rendering for {ev['kind']}"
    # quiet renders nothing
    assert render_event({"kind": "outcome", "t": 1.0}, "quiet") == []


def test_attempt_dag_proof_emits_trace_events():
    events: list[tuple[str, dict]] = []

    def trace(kind, **payload):
        events.append((kind, payload))

    # First verify fails at the leaf, repair fixes it, second verifies.
    calls = {"n": 0}

    def verify_fn(header, body):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"ok": False,
                    "errors": "f.lean:4:4: error: positivity failed",
                    "body_line_offset": 3}
        return {"ok": True, "errors": None, "body_line_offset": 3}

    def llm(system, user):
        if "repairing FAILED steps" in system:
            return json.dumps({"repairs": [
                {"id": "h1", "type": "0 ≤ (a - b)^2",
                 "tactic": "nlinarith [sq_nonneg (a - b)]"}]})
        return GOOD_SKETCH

    res = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=llm, verify_fn=verify_fn,
        sketch_attempts=1, repair_rounds=2, trace=trace,
    )
    assert res.verified
    kinds = [k for k, _ in events]
    for expected in ("sketch_attempt", "sketch_parsed", "assembled",
                     "error_attribution", "repair_round", "repair_applied"):
        assert expected in kinds, f"missing {expected}: {kinds}"
    # The repair round targeted the broken leaf.
    rr = dict(events)[
        "repair_round"]
    assert rr["broken"] == ["h1"]


def test_attempt_dag_proof_trace_none_is_noop():
    def verify_fn(header, body):
        return {"ok": True, "errors": None, "body_line_offset": 3}

    res = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=lambda s, u: GOOD_SKETCH,
        verify_fn=verify_fn, sketch_attempts=1, repair_rounds=0,
    )
    assert res.verified


def test_sample_thinking_field_defaults_none():
    s = Sample(text="x", score=0.5, source="rank-fallback")
    assert s.thinking is None
    s2 = Sample(text="x", score=0.5, source="rank-fallback",
                thinking="chain of thought")
    assert s2.thinking == "chain of thought"
