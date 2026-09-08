"""Unit tests for the DAG targeted-repair path (no Lean, no LLM —
fake callables throughout)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.proof_dag import (  # noqa: E402
    CLOSER_ID,
    HEADER_ID,
    Sketch,
    HaveNode,
    assemble_proof_with_map,
    attribute_errors,
    attempt_dag_proof,
    combine_candidates,
    has_header_level_error,
    make_leaf_theorem,
    parse_error_locations,
    split_theorem_header,
)


def _sketch_json(haves, closer):
    return json.dumps({"haves": haves, "closer": closer})


GOOD_SKETCH = _sketch_json(
    [
        {"id": "h1", "type": "0 ≤ (a - b)^2", "tactic": "positivity",
         "depends": []},
        {"id": "h2", "type": "a^2 + b^2 ≥ 2*a*b", "tactic": "nlinarith [h1]",
         "depends": ["h1"]},
    ],
    "linarith [h2]",
)


def test_assemble_map_segments_cover_body():
    sketch_raw = json.loads(GOOD_SKETCH)
    sketch = Sketch(
        haves=[HaveNode(h["id"], h["type"], h["tactic"], h["depends"])
               for h in sketch_raw["haves"]],
        closer=sketch_raw["closer"],
    )
    body, segments = assemble_proof_with_map(sketch)
    n_lines = len(body.splitlines())
    assert segments[-1][0] == CLOSER_ID
    assert segments[-1][2] == n_lines
    # Segments are contiguous and start at line 1.
    assert segments[0][1] == 1
    for (_, _, prev_end), (_, start, _) in zip(segments, segments[1:]):
        assert start == prev_end + 1


def test_parse_error_locations_windows_paths():
    errors = (
        r"path\to\prover\lean\Generated\Try_ab.lean:5:2: error: "
        "linarith failed\nsome detail\n"
        r"path\to\prover\lean\Generated\Try_ab.lean:9:4: error: "
        "unsolved goals"
    )
    locs = parse_error_locations(errors)
    assert [line for line, _ in locs] == [5, 9]
    assert "linarith failed" in locs[0][1]
    assert "unsolved goals" in locs[1][1]


def test_attribute_errors_maps_to_leaf_and_closer():
    segments = [("h1", 1, 2), ("h2", 3, 4), (CLOSER_ID, 5, 5)]
    errors = (
        "Try_x.lean:6:2: error: broke in h2\n"
        "Try_x.lean:8:0: error: closer broke\n"
        "Try_x.lean:1:0: error: header-level error"
    )
    by_seg = attribute_errors(errors, segments, body_line_offset=3)
    assert "broke in h2" in by_seg["h2"][0]
    assert "closer broke" in by_seg[CLOSER_ID][0]
    # Pre-body errors that aren't env/unsolved-goals go to HEADER_ID —
    # the statement itself is broken, no sketch repair can fix it.
    assert any("header-level" in m for m in by_seg[HEADER_ID])
    assert not any("header-level" in m for m in by_seg[CLOSER_ID])


def test_repair_loop_fixes_single_leaf_keeps_rest():
    calls = {"llm": 0, "verify": 0}

    def fake_llm(system, user):
        calls["llm"] += 1
        if calls["llm"] == 1:
            return GOOD_SKETCH
        # Repair call: fix h1 only.
        assert "BROKEN steps" in user
        assert "h1" in user
        return json.dumps(
            {"repairs": [{"id": "h1", "type": "0 ≤ (a - b)^2",
                          "tactic": "nlinarith [sq_nonneg (a - b)]"}]}
        )

    def fake_verify(header, body):
        calls["verify"] += 1
        if "nlinarith [sq_nonneg (a - b)]" in body:
            return {"ok": True, "errors": None, "body_line_offset": 3}
        # First verify: error on h1's tactic line. Body line 2 (= file
        # line 5 with offset 3) is inside h1's segment.
        return {"ok": False,
                "errors": "Try_x.lean:5:4: error: positivity failed",
                "body_line_offset": 3}

    result = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=fake_llm,
        verify_fn=fake_verify,
        sketch_attempts=1,
        repair_rounds=2,
    )
    assert result.verified
    assert result.repair_rounds_used == 1
    assert result.repaired_ids == ["h1"]
    # The untouched leaf survived the repair.
    assert any(h.id == "h2" and h.tactic == "nlinarith [h1]"
               for h in result.sketch.haves)
    assert calls["verify"] == 2


def test_repair_rejects_forbidden_and_falls_back_to_sketch_retry():
    calls = {"llm": 0}

    def fake_llm(system, user):
        calls["llm"] += 1
        if calls["llm"] == 1:
            return GOOD_SKETCH
        if calls["llm"] == 2:
            # Repair tries to cheat with sorry — must be rejected.
            return json.dumps(
                {"repairs": [{"id": "h1", "tactic": "sorry"}]}
            )
        # Outer sketch retry succeeds.
        return _sketch_json(
            [{"id": "h1", "type": "True", "tactic": "trivial",
              "depends": []}],
            "exact h1.elim",
        )

    def fake_verify(header, body):
        if "trivial" in body:
            return {"ok": True, "errors": None, "body_line_offset": 3}
        return {"ok": False,
                "errors": "Try_x.lean:5:4: error: nope",
                "body_line_offset": 3}

    result = attempt_dag_proof(
        "theorem t : True",
        sketch_llm_call=fake_llm,
        verify_fn=fake_verify,
        sketch_attempts=2,
        repair_rounds=2,
    )
    assert result.verified
    assert result.sketch_attempts == 2
    assert result.repaired_ids == []


def test_split_header_ignores_binder_colons():
    header = ("theorem mathd_x (a b : ℝ) (h₀ : 0 < a) "
              "(h₁ : a * b = 1) : a + b ≥ 2")
    prefix, goal = split_theorem_header(header)
    assert prefix.endswith("(h₁ : a * b = 1)")
    assert goal == "a + b ≥ 2"


def test_make_leaf_theorem_adds_dep_binders_and_renames():
    sketch = Sketch(
        haves=[
            HaveNode("h_pos", "0 < a * b", "positivity", []),
            HaveNode("h_key", "a + b ≥ 2", "nlinarith [h_pos]", ["h_pos"]),
        ],
        closer="exact h_key",
    )
    header = "theorem orig_name (a b : ℝ) (h₀ : 0 < a) : a + b ≥ 2"
    stmt = make_leaf_theorem(header, sketch, sketch.haves[1])
    assert stmt.startswith("theorem leaf_h_key ")
    assert "(a b : ℝ)" in stmt and "(h₀ : 0 < a)" in stmt
    assert "(h_pos : 0 < a * b)" in stmt
    assert stmt.endswith(": a + b ≥ 2")


def test_combine_candidates_dedup_and_first():
    assert combine_candidates([]) == ""
    # Single candidate passes through verbatim (multi-line preserved).
    single = combine_candidates(["intro h\nlinarith"])
    assert single == "intro h\nlinarith"
    combined = combine_candidates(["omega", "omega", "nlinarith\nring_nf"])
    assert combined == "first | (omega) | (nlinarith; ring_nf)"


def test_leaf_closer_fixes_leaf_without_sketch_llm_repair():
    calls = {"llm": 0, "closer": 0}

    def fake_llm(system, user):
        calls["llm"] += 1
        assert calls["llm"] == 1, (
            "sketch LLM must not be called for repair when the leaf "
            "closer fixes every broken segment"
        )
        return GOOD_SKETCH

    def fake_closer(leaf_stmt):
        calls["closer"] += 1
        assert leaf_stmt.startswith("theorem leaf_h1")
        return "nlinarith [sq_nonneg (a - b)]"

    def fake_verify(header, body):
        if "nlinarith [sq_nonneg (a - b)]" in body:
            return {"ok": True, "errors": None, "body_line_offset": 3}
        return {"ok": False,
                "errors": "Try_x.lean:5:4: error: positivity failed",
                "body_line_offset": 3}

    result = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=fake_llm,
        verify_fn=fake_verify,
        sketch_attempts=1,
        repair_rounds=2,
        leaf_closer_call=fake_closer,
    )
    assert result.verified
    assert result.leaf_closer_fixed_ids == ["h1"]
    assert result.repaired_ids == ["h1"]
    assert calls == {"llm": 1, "closer": 1}


def test_leaf_closer_sorry_rejected_falls_to_sketch_llm():
    calls = {"llm": 0}

    def fake_llm(system, user):
        calls["llm"] += 1
        if calls["llm"] == 1:
            return GOOD_SKETCH
        # Repair call — closer's sorry was rejected, so the broken leaf
        # must appear here.
        assert "BROKEN steps" in user and "h1" in user
        return json.dumps(
            {"repairs": [{"id": "h1",
                          "tactic": "nlinarith [sq_nonneg (a - b)]"}]}
        )

    def fake_verify(header, body):
        if "nlinarith [sq_nonneg (a - b)]" in body:
            return {"ok": True, "errors": None, "body_line_offset": 3}
        return {"ok": False,
                "errors": "Try_x.lean:5:4: error: positivity failed",
                "body_line_offset": 3}

    result = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=fake_llm,
        verify_fn=fake_verify,
        sketch_attempts=1,
        repair_rounds=2,
        leaf_closer_call=lambda stmt: "sorry",
    )
    assert result.verified
    assert result.leaf_closer_fixed_ids == []
    assert result.repaired_ids == ["h1"]


def test_attribute_errors_drops_prebody_env_errors():
    """Missing-olean / unknown-module errors at the import lines must be
    dropped (repair can't fix the environment), while genuine header-
    line errors still fall through to the closer bucket."""
    segments = [("h1", 1, 2), (CLOSER_ID, 3, 3)]
    errors = (
        "Try_x.lean:1:0: error: object file 'C:\\x\\Basic.olean' of "
        "module Mathlib.Algebra.BigOperators.Basic does not exist\n"
        "Try_x.lean:3:0: error: unsolved goals"
    )
    by_seg = attribute_errors(errors, segments, body_line_offset=3)
    assert CLOSER_ID in by_seg and len(by_seg) == 1
    assert "unsolved goals" in by_seg[CLOSER_ID][0]

    only_env = "Try_x.lean:1:0: error: unknown module prefix 'Mathlib.X'"
    assert attribute_errors(only_env, segments, body_line_offset=3) == {}


def test_topo_drops_hypothesis_deps_keeps_have_deps():
    """`depends` on theorem hypotheses (h₀ …) must be dropped, not
    rejected — they're always in scope and carry no ordering info."""
    from search.proof_dag import topo_sort

    haves = [
        HaveNode("h_b", "B", "tac_b", ["h_a", "h₀"]),  # h₀ not a have
        HaveNode("h_a", "A", "tac_a", ["h₁"]),          # h₁ not a have
    ]
    ordered, err = topo_sort(haves)
    assert err is None
    assert [h.id for h in ordered] == ["h_a", "h_b"]  # have-dep respected
    assert ordered[0].depends == [] and ordered[1].depends == ["h_a"]

    # Real cycles among haves are still rejected.
    cyc = [HaveNode("x", "X", "t", ["y"]), HaveNode("y", "Y", "t", ["x"])]
    ordered2, err2 = topo_sort(cyc)
    assert ordered2 is None and err2 == "cycle_in_have_graph"


def test_structure_sensitive_tactics_not_inlined():
    from search.proof_dag import _wrap_with_fallback, DEFAULT_LEAF_FALLBACKS

    calc_tac = "calc a = b := by ring\n  _ = c := by ring"
    # Fallback wrap must leave calc blocks verbatim — `;`-joining them
    # is the `unexpected token ';'` failure from amc12a_2020_p9.
    assert _wrap_with_fallback(calc_tac, DEFAULT_LEAF_FALLBACKS) == calc_tac
    # Plain tactics still get the ladder.
    wrapped = _wrap_with_fallback("omega", DEFAULT_LEAF_FALLBACKS)
    assert wrapped.startswith("first | (omega) | ")
    # combine_candidates skips calc blocks when combining, but a lone
    # structure-sensitive candidate passes through verbatim.
    assert combine_candidates([calc_tac, "omega", "ring"]) == \
        "first | (omega) | (ring)"
    assert combine_candidates([calc_tac]) == calc_tac


def test_premises_appear_in_all_prompts():
    from search.proof_dag import (
        sketch_user_prompt, sketch_retry_prompt, repair_user_prompt,
    )
    prem = ["Nat.gcd_mul_lcm", "sq_nonneg"]
    assert "Nat.gcd_mul_lcm" in sketch_user_prompt("theorem t : True", prem)
    assert "sq_nonneg" in sketch_retry_prompt(
        "theorem t : True", "prev", "err", prem)
    assert "Nat.gcd_mul_lcm" in repair_user_prompt(
        "theorem t : True", [], [], "omega", True, "err", premises=prem)
    # And absent when not supplied.
    assert "retrieved" not in sketch_user_prompt("theorem t : True")


def test_recursive_decomposition_on_stubborn_leaf():
    """A leaf broken twice (surviving one ordinary repair) gets
    decomposed into a sub-DAG; the assembled sub-proof becomes its
    tactic block and the whole proof then verifies."""
    calls = {"llm": 0}
    SUB_SKETCH = _sketch_json(
        [{"id": "h_sub", "type": "0 ≤ (a - b)^2",
          "tactic": "positivity", "depends": []}],
        "nlinarith [h_sub]",
    )

    def fake_llm(system, user):
        calls["llm"] += 1
        if calls["llm"] == 1:
            return GOOD_SKETCH
        if calls["llm"] == 2:
            # Round-1 ordinary repair: returns a tactic that still fails.
            return json.dumps(
                {"repairs": [{"id": "h1", "tactic": "norm_num"}]})
        # Round-2: h1 broken again (streak 2) → decomposition prompt is
        # the leaf posed standalone.
        assert "theorem leaf_h1" in user
        return SUB_SKETCH

    def fake_verify(header, body):
        if "nlinarith [h_sub]" in body:
            return {"ok": True, "errors": None, "body_line_offset": 3}
        # h1's tactic line keeps failing (body line 2 → file line 5).
        return {"ok": False,
                "errors": "Try_x.lean:5:4: error: tactic failed",
                "body_line_offset": 3}

    result = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=fake_llm,
        verify_fn=fake_verify,
        sketch_attempts=1,
        repair_rounds=3,
        decompose_depth=1,
    )
    assert result.verified
    assert result.decomposed_ids == ["h1"]
    # The decomposed leaf's tactic is a have-block, not a single tactic.
    h1 = next(h for h in result.sketch.haves if h.id == "h1")
    assert "have h_sub" in h1.tactic and "nlinarith [h_sub]" in h1.tactic


def test_repl_messages_to_lake_errors_roundtrip():
    """REPL message dicts → lake-style text → parse_error_locations must
    recover the same line numbers, so the REPL backend plugs into leaf
    attribution unchanged."""
    from backend.repl_step import repl_messages_to_lake_errors

    msgs = [
        {"severity": "error", "pos": {"line": 5, "column": 4},
         "data": "linarith failed to find a contradiction"},
        {"severity": "warning", "pos": {"line": 1, "column": 0},
         "data": "declaration uses 'sorry'"},
        {"severity": "error", "pos": {"line": 9, "column": 2},
         "data": "unsolved goals\n⊢ False"},
    ]
    text = repl_messages_to_lake_errors(msgs)
    locs = parse_error_locations(text)
    assert [line for line, _ in locs] == [5, 9]  # warning excluded
    assert "linarith failed" in locs[0][1]
    assert "unsolved goals" in locs[1][1]


def test_prebody_unsolved_goals_still_goes_to_closer():
    """`unsolved goals` reported at the theorem line means the closer
    didn't finish the job — that stays repairable via CLOSER_ID."""
    segments = [("h1", 1, 2), (CLOSER_ID, 3, 3)]
    errors = "repl.lean:1:0: error: unsolved goals\n⊢ False"
    by_seg = attribute_errors(errors, segments, body_line_offset=3)
    assert CLOSER_ID in by_seg and HEADER_ID not in by_seg


def test_has_header_level_error():
    # Pre-body parse error → header-level (the amc12_2001_p5 shape:
    # scoped `!` notation without the file's `open Nat`).
    assert has_header_level_error(
        "repl.lean:2:37: error: expected token", body_line_offset=3)
    # Pre-body unsolved goals → NOT header-level (closer's job).
    assert not has_header_level_error(
        "repl.lean:1:0: error: unsolved goals", body_line_offset=3)
    # In-body error → not header-level.
    assert not has_header_level_error(
        "x.lean:5:0: error: expected token", body_line_offset=3)
    # Pre-body env error → not header-level (dropped, not repaired).
    assert not has_header_level_error(
        "x.lean:1:0: error: unknown module prefix 'Mathlib.X'",
        body_line_offset=3)


def test_header_level_error_stops_repair_loop():
    """A header that doesn't elaborate must not burn repair rounds —
    smoke10_v3 spent 7 closer repairs chasing amc12_2001_p5's header
    parse error."""
    calls = {"llm": 0}

    def fake_llm(system, user):
        calls["llm"] += 1
        return GOOD_SKETCH

    def fake_verify(header, body):
        return {"ok": False,
                "errors": "repl.lean:2:37: error: expected token",
                "body_line_offset": 3}

    result = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=fake_llm,
        verify_fn=fake_verify,
        sketch_attempts=2,
        repair_rounds=4,
    )
    assert not result.verified
    assert result.repair_rounds_used == 0
    assert calls["llm"] == 2  # the two sketch calls, zero repair calls
    assert any(e.startswith("header_level_error")
               for e in result.repair_errors)


def test_empty_repair_response_recorded_and_breaks():
    """An empty repair response must be visible in telemetry, not a
    silent break — smoke10_v3's algebra_abpbcpcageq3 showed repairs=0
    with no trace of why."""
    calls = {"llm": 0}

    def fake_llm(system, user):
        calls["llm"] += 1
        if calls["llm"] == 1:
            return GOOD_SKETCH
        return ""  # repair call comes back empty

    def fake_verify(header, body):
        return {"ok": False,
                "errors": "Try_x.lean:5:4: error: positivity failed",
                "body_line_offset": 3}

    result = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=fake_llm,
        verify_fn=fake_verify,
        sketch_attempts=1,
        repair_rounds=3,
    )
    assert not result.verified
    assert result.repair_rounds_used == 0
    assert "empty_repair_response" in result.repair_errors


def test_unapplied_repair_recorded():
    """A repair response that parses but applies nothing (e.g. only
    forbidden tactics) must leave a repair_not_applied trace."""
    calls = {"llm": 0}

    def fake_llm(system, user):
        calls["llm"] += 1
        if calls["llm"] == 1:
            return GOOD_SKETCH
        return json.dumps({"repairs": [{"id": "h1", "tactic": "sorry"}]})

    def fake_verify(header, body):
        return {"ok": False,
                "errors": "Try_x.lean:5:4: error: positivity failed",
                "body_line_offset": 3}

    result = attempt_dag_proof(
        "theorem t (a b : ℝ) : a^2 + b^2 ≥ 2*a*b",
        sketch_llm_call=fake_llm,
        verify_fn=fake_verify,
        sketch_attempts=1,
        repair_rounds=3,
    )
    assert not result.verified
    assert any(e.startswith("repair_not_applied")
               for e in result.repair_errors)


def test_prepend_open_scopes():
    from eval.run_minif2f import prepend_open_scopes

    src = ("import Mathlib\n\nset_option maxHeartbeats 0\n\n"
           "open BigOperators Real Nat Topology Rat\n\n"
           "theorem foo :\n  (10000!) = 1 := by sorry")
    hdr = "theorem foo :\n  (10000!) = 1"
    out = prepend_open_scopes(src, hdr)
    # set_option and open both carried, in file order, as `… in` lines —
    # dropping set_option maxHeartbeats reintroduces the whnf-timeout
    # failure from smoke10_v4 (algebra_abpbcpcageq3).
    assert out == ("set_option maxHeartbeats 0 in\n"
                   "open BigOperators Real Nat Topology Rat in\n" + hdr)
    # No opens in the file → header unchanged.
    assert prepend_open_scopes("import Mathlib\n", hdr) == hdr
    # Indented statement lines must never be scooped as opens.
    src2 = "theorem bar :\n  open_interval x := by sorry"
    assert prepend_open_scopes(src2, hdr) == hdr


def test_open_scoped_header_splits_and_makes_leaves():
    """split_theorem_header / make_leaf_theorem must survive an
    `open … in`-prefixed header (the post-fix header shape)."""
    header = ("open BigOperators Real Nat Topology Rat in\n"
              "theorem amc_x (a b : ℝ) (h₀ : 0 < a) : a + b ≥ 2")
    prefix, goal = split_theorem_header(header)
    assert goal == "a + b ≥ 2"
    assert prefix.startswith("open BigOperators")
    sketch = Sketch(
        haves=[HaveNode("h_key", "0 < a * b", "positivity", [])],
        closer="nlinarith [h_key]",
    )
    stmt = make_leaf_theorem(header, sketch, sketch.haves[0])
    assert stmt is not None
    assert "theorem leaf_h_key" in stmt
    assert stmt.startswith("open BigOperators Real Nat Topology Rat in")
    assert stmt.endswith(": 0 < a * b")


def test_no_offset_means_no_repair_calls():
    calls = {"llm": 0}

    def fake_llm(system, user):
        calls["llm"] += 1
        return GOOD_SKETCH

    def fake_verify(header, body):
        return {"ok": False, "errors": "Try_x.lean:5:4: error: nope"}

    result = attempt_dag_proof(
        "theorem t : True",
        sketch_llm_call=fake_llm,
        verify_fn=fake_verify,
        sketch_attempts=2,
        repair_rounds=3,
    )
    assert not result.verified
    assert result.repair_rounds_used == 0
    # Only the two sketch calls — no repair calls without line offsets.
    assert calls["llm"] == 2
