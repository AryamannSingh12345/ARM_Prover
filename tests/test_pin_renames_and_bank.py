"""Regressions from the 2026-07-24 arm_engel3_final1/final2 triage:

1. `apply_pin_renames` — stale pre-`₀` Mathlib names in ANY model
   response are rewritten to their current pinned names before parsing
   (the model emits `div_le_div_iff` no matter what the prompt says).
2. Header-level errors BREAK the repair loop — final1 rewrote its
   correct `exact goal` closer chasing the `unsolved goals` cascade of
   a broken header.
3. `_append_lemma_library` dedups by declared name — blind appends
   accumulated duplicate `aux_titu3` declarations across runs.

Fake LLM / verify throughout — no Lean, no API.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.proof_dag import (  # noqa: E402
    apply_pin_renames,
    attempt_dag_proof,
    _append_lemma_library,
)

HEADER = "theorem t (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : a / b ≤ a / b"


# ---------- apply_pin_renames -------------------------------------------------

def test_pin_renames_basic():
    assert apply_pin_renames("rw [div_le_div_iff hb hd]") == \
        "rw [div_le_div_iff₀ hb hd]"
    assert apply_pin_renames("le_div_iff hc, div_le_iff hc") == \
        "le_div_iff₀ hc, div_le_iff₀ hc"
    assert apply_pin_renames("lt_div_iff h, div_lt_iff h, div_lt_div_iff a b") \
        == "lt_div_iff₀ h, div_lt_iff₀ h, div_lt_div_iff₀ a b"


def test_pin_renames_no_double_suffix():
    assert apply_pin_renames("div_le_div_iff₀ hb hd") == \
        "div_le_div_iff₀ hb hd"


def test_pin_renames_word_boundaries():
    # Derived names with word-char continuations are untouched.
    assert apply_pin_renames("div_le_div_iff_left h") == \
        "div_le_div_iff_left h"
    # Substring inside a longer identifier is untouched
    # (`le_div_iff` inside `div_le_div_iff` must not double-fire).
    assert apply_pin_renames("div_le_div_iff hb hd") == \
        "div_le_div_iff₀ hb hd"


def test_pin_renames_primed_forms():
    # `div_le_iff'` → `div_le_iff₀'` (the primed ₀ variant exists).
    assert apply_pin_renames("div_le_iff' hc") == "div_le_iff₀' hc"


def test_pin_renames_inside_json():
    raw = json.dumps({"proof": "rw [div_le_div_iff hb hd]\nnlinarith"})
    out = apply_pin_renames(raw)
    assert json.loads(out)["proof"].startswith("rw [div_le_div_iff₀")


def test_llm_responses_are_renamed_before_verify():
    """A sketch whose tactic uses a stale name must reach verify_fn
    already renamed — the rewrite sits between the model and the
    parser, so verifier and committed proof always agree."""
    sketch = json.dumps({
        "haves": [{"id": "h1", "type": "a / b ≤ a / b",
                   "tactic": "rw [div_le_div_iff h2 h2]", "depends": []}],
        "closer": "exact h1",
    })
    seen_bodies: list[str] = []

    def llm(system, user):
        return sketch

    def verify(header, body):
        seen_bodies.append(body)
        return {"ok": True, "errors": None, "body_line_offset": 3}

    res = attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify,
        sketch_attempts=1, repair_rounds=0,
        leaf_fallbacks=(), decompose_depth=0)
    assert res.verified
    assert all("div_le_div_iff₀" in b for b in seen_bodies)
    assert not any("div_le_div_iff " in b for b in seen_bodies)
    assert "div_le_div_iff₀" in (res.assembled_proof or "")


# ---------- header-level error breaks the repair loop -------------------------

def test_header_error_breaks_repair_without_touching_closer():
    """Regression (arm_engel3_final1): a compile with a header-level
    error AND an `unsolved goals` cascade routed to the closer used to
    trigger a closer repair that replaced the correct `exact goal`.
    Now the round records header_level_error and breaks — no repair
    LLM call, closer untouched."""
    sketch = json.dumps({
        "haves": [{"id": "h1", "type": "a / b ≤ a / b",
                   "tactic": "norm_num", "depends": []}],
        "closer": "exact goal_marker",
    })
    # offset 3: line 2 is pre-body (header), so both errors are
    # pre-body → one 'header', one 'goals' (→ closer).
    fail = ("f.lean:2:4: error: unexpected identifier; expected command\n"
            "f.lean:3:10: error: unsolved goals\n⊢ True")
    calls = {"llm": 0}

    def llm(system, user):
        calls["llm"] += 1
        return sketch  # any repair call would also land here

    def verify(header, body):
        return {"ok": False, "errors": fail, "body_line_offset": 3}

    res = attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify,
        sketch_attempts=1, repair_rounds=3,
        leaf_fallbacks=(), decompose_depth=0)
    assert not res.verified
    assert any(e.startswith("header_level_error") for e in res.repair_errors)
    # exactly ONE llm call (the sketch) — no repair calls chased the cascade
    assert calls["llm"] == 1
    # the closer survived untouched
    assert "exact goal_marker" in (res.assembled_proof or "")


# ---------- lemma library dedup ----------------------------------------------

def test_lemma_library_dedup_by_name(tmp_path):
    lib = tmp_path / "lib.lean"
    a1 = "lemma aux_titu3 (x : ℝ) : x = x := by rfl"
    a2 = "lemma aux_titu3 (y : ℝ) : y = y := by rfl"  # same name, new proof
    b = "lemma aux_engel3 : True := by trivial"
    _append_lemma_library(str(lib), "run one", [a1])
    _append_lemma_library(str(lib), "run two", [a2, b])
    text = lib.read_text(encoding="utf-8")
    assert text.count("lemma aux_titu3") == 1
    assert "aux_engel3" in text


def test_lemma_library_dedup_within_one_call(tmp_path):
    lib = tmp_path / "lib.lean"
    d = "lemma aux_dup : True := by trivial"
    _append_lemma_library(str(lib), "tag", [d, d])
    assert lib.read_text(encoding="utf-8").count("lemma aux_dup") == 1


def test_lemma_library_never_raises(tmp_path):
    # unwritable path (a directory) — must swallow, not raise
    _append_lemma_library(str(tmp_path), "tag", ["lemma x : True := trivial"])
