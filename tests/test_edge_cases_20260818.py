"""Edge-case sweep over every change made on 2026-08-18.

Written after a regression shipped the same day: an unconditional
short-circuit in `compile_lean` silently broke FIVE sorry-tolerant
callers, and the fast suite stayed green because none of them had a test
exercising the branch. These tests target the boundaries rather than the
happy path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backend import compile_verify as CV                       # noqa: E402
from backend.axiom_check import (evaluate_report,              # noqa: E402
                                 extract_theorem_names,
                                 parse_axiom_report)
from search.decl_module_index import modules_for_premises      # noqa: E402
from search.proof_dag import harden_tactic_block               # noqa: E402


# =====================================================================
# 1. compile_lean short-circuit
# =====================================================================

def test_sorry_inside_a_comment_still_rejects_and_that_is_PRE_EXISTING():
    """`\\bsorry\\b` matches inside comments — a false rejection.

    NOT introduced today: `ok` has always required `not used_sorry`, and
    `used_sorry` has always come from this same source scan, so a proof
    carrying `-- no sorry here` was ALREADY rejected. The short-circuit
    only makes the (already wrong) verdict arrive faster and with a
    clearer message. Pinned so the behaviour is deliberate and visible
    rather than folklore.
    """
    src = "theorem t : True := by\n  -- no sorry here\n  trivial\n"
    res = CV.compile_lean(src)
    assert res.used_sorry is True
    assert res.ok is False


def test_identifier_containing_sorry_is_not_matched():
    """`not_sorry` must NOT trip the scan — `_` is a word character."""
    assert not CV._SORRY_RE.search("exact not_sorry")
    assert not CV._SORRY_RE.search("exact sorrymongering")


def test_sorryax_variants():
    assert CV._SORRY_RE.search("exact sorryAx _ true")
    assert CV._SORRY_RE.search("  sorryAx")
    # sorryAx as a suffix of a longer identifier must not match
    assert not CV._SORRY_RE.search("exact mysorryAxioms")


def test_empty_and_whitespace_sources_do_not_short_circuit():
    for src in ("", "   \n\n  "):
        # no sorry, no native_decide -> must proceed to the normal path
        assert not CV._SORRY_RE.search(src)


def test_short_circuit_result_is_shaped_like_a_real_result():
    """Callers read these fields; none may be missing or wrong-typed."""
    res = CV.compile_lean("theorem t : True := by native_decide\n")
    assert res.ok is False
    assert res.used_native_decide is True
    assert res.used_sorry is False
    assert isinstance(res.errors, str) and res.errors
    assert isinstance(res.argv, list)
    assert isinstance(res.cwd, str) and res.cwd
    assert res.elapsed_s == 0.0
    assert res.axioms_ok is None       # gate never ran
    assert res.axiom_detail == ""


def test_short_circuit_still_writes_the_debug_dump(tmp_path):
    """--debug-dump-lean must produce an artifact even when no compile ran.

    A source rejected for `sorry` is precisely what someone turns dumping
    on to inspect; returning early without writing it would leave that
    case with no artifact at all.
    """
    stem = tmp_path / "d"
    src = "theorem t : True := by sorry\n"
    res = CV.compile_lean(src, debug_dump_path=stem)
    assert res.ok is False
    assert stem.with_suffix(".lean").exists()
    assert stem.with_suffix(".lean").read_text(encoding="utf-8") == src
    assert res.debug_dump_path == str(stem.with_suffix(".lean"))
    # the documented contract is BOTH files; the sidecar must say why no
    # subprocess ran rather than simply be absent
    import json
    side = stem.with_suffix(".json")
    assert side.exists()
    meta = json.loads(side.read_text(encoding="utf-8"))
    assert meta["status"] == "rejected_precompile"
    assert meta["argv"] == [] and meta["used_sorry"] is True


def test_reject_sorry_false_reaches_the_axiom_gate_path_safely(monkeypatch):
    """opt-out + a clean compile must not crash the axiom-gate branch."""
    from types import SimpleNamespace
    tmp = Path(__file__).parent / "_edge_tmp.lean"
    tmp.write_text("x", encoding="utf-8")
    monkeypatch.setattr(CV, "_write_temp", lambda s: tmp)
    monkeypatch.setattr(
        CV.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""))
    try:
        res = CV.compile_lean("theorem t : True := by sorry\n",
                              reject_sorry=False, check_axioms=False)
        # lake "succeeded" but the source has sorry, so ok stays False
        assert res.ok is False and res.used_sorry is True
    finally:
        tmp.unlink(missing_ok=True)


# =====================================================================
# 2. harden_tactic_block
# =====================================================================

def test_whitespace_only_block():
    out = harden_tactic_block("   \n  \n", sweep=True)
    assert "all_goals try omega" in out


def test_already_hardened_line_is_not_double_wrapped():
    assert harden_tactic_block("try simp", sweep=False) == "try simp"
    assert harden_tactic_block("all_goals simp", sweep=False) == "all_goals simp"


def test_focus_dot_and_case_arrow_are_left_alone():
    for line in ("· simp", "case foo => simp", "next => simp"):
        assert harden_tactic_block(line, sweep=False) == line


def test_line_after_bare_first_is_treated_as_alternation():
    src = "first\nsimp\nring"
    out = harden_tactic_block(src, sweep=False)
    # the line directly after `first` must not be wrapped
    assert out.splitlines()[1] == "simp"


def test_sweep_indent_follows_first_nonempty_line():
    out = harden_tactic_block("\n\n      exact foo").splitlines()
    assert out[-1] == "      all_goals try ac_rfl"


def test_multiline_mixed_indentation_preserved():
    src = "intro i\n  simp\n    norm_num"
    out = harden_tactic_block(src, sweep=False).splitlines()
    assert out[1] == "  try simp"
    assert out[2] == "    try norm_num"


def test_hardening_never_reorders_or_drops_lines():
    src = "intro i\nsimp\nexact foo\nring_nf"
    out = harden_tactic_block(src, sweep=False).splitlines()
    assert len(out) == 4
    assert out[0] == "intro i" and out[2] == "exact foo"


def test_semicolon_chain_is_not_wrapped():
    """`simp; exact foo` — wrapping would change sequencing semantics."""
    src = "constructor <;> simp <;> ring"
    assert harden_tactic_block(src, sweep=False) == src


# =====================================================================
# 3. axiom gate
# =====================================================================

def test_theorem_name_extraction_edge_cases():
    src = ("private theorem a' : True := trivial\n"
           "protected lemma B_2 : True := trivial\n"
           "noncomputable theorem c₀ : True := trivial\n"
           "@[simp] theorem d : True := trivial\n")
    names = extract_theorem_names(src)
    assert "a'" in names and "B_2" in names and "d" in names


def test_example_declares_nothing():
    assert extract_theorem_names("example : True := trivial\n") == []


def test_report_with_empty_axiom_list_is_accepted():
    rep = parse_axiom_report("'foo' depends on axioms: []")
    assert rep["foo"] == frozenset()
    assert evaluate_report(["foo"], rep)[0] is True


def test_namespaced_report_matches_short_name_but_not_a_suffix_collision():
    ok, _, _, inconc, _ = evaluate_report(["foo"], {"Ns.foo": frozenset()})
    assert ok and not inconc
    # `barfoo` must NOT satisfy a request for `foo`
    ok2, _, _, inconc2, _ = evaluate_report(["foo"], {"Ns.barfoo": frozenset()})
    assert not ok2 and inconc2


def test_partial_report_fails_closed():
    """One theorem reported, one missing -> inconclusive, not ok."""
    ok, _, _, inconc, detail = evaluate_report(
        ["a", "b"], {"a": frozenset()})
    assert not ok and inconc and "b" in detail


# =====================================================================
# 4. premise seeding
# =====================================================================

def test_limit_zero_and_negative_are_safe(monkeypatch):
    """Degenerate limits must terminate and never raise.

    The loop appends then checks `len(out) >= limit`, so limit<=0 yields
    exactly one module rather than none. That is harmless — one import
    is not a correctness problem — but it IS the behaviour, so pin it
    rather than assume zero.
    """
    from search import decl_module_index as DMI
    monkeypatch.setattr(DMI, "resolve_decl_module", lambda n: f"M.{n}")
    assert modules_for_premises(["a", "b"], limit=0) == ["M.a"]
    assert modules_for_premises(["a", "b"], limit=-1) == ["M.a"]


def test_duplicate_names_collapse(monkeypatch):
    from search import decl_module_index as DMI
    monkeypatch.setattr(DMI, "resolve_decl_module", lambda n: "SameMod")
    assert modules_for_premises(["a", "b", "c"]) == ["SameMod"]


# =====================================================================
# 5. bonus repair round — boundary budgets
# =====================================================================

def _drive(commit: bool, repair_rounds: int):
    import itertools

    from search import proof_dag
    from search.dag.lemma_store import LemmaStore

    header = "import Mathlib\n\ntheorem tgt (n : ℕ) : n + 0 = n"
    sketch = ('{"haves": [{"id": "h1", "type": "n + 0 = n", '
              '"tactic": "simp", "depends": []}], "closer": "exact h1"}')
    theory = ('{"defs": [], "lemmas": [{"name": "aux_good", "statement": '
              '"lemma aux_good : (1:ℕ) + 0 = 1"}], '
              '"leaf_tactics": {"h1": "exact aux_good"}, "rationale": "r"}')
    counter = itertools.count()

    def llm(system, user):
        if system == proof_dag.SKETCH_SYSTEM:
            return sketch
        if system == proof_dag.THEORY_SYSTEM:
            return theory
        if system == proof_dag.PROVE_LEMMA_SYSTEM:
            return '{"proof": "rfl"}'
        return '{"repairs": [{"id": "h1", "tactic": "simp [x%d]"}]}' % next(counter)

    def verify(hdr, body):
        if hdr.rstrip().endswith("lemma aux_good : (1:ℕ) + 0 = 1"):
            return {"ok": bool(commit),
                    "errors": None if commit else "x.lean:1:1: error: nope",
                    "body_line_offset": 1}
        return {"ok": False,
                "errors": "x.lean:4:2: error: unsolved goals\n  ⊢ n + 0 = n",
                "body_line_offset": 3}

    events: list = []
    proof_dag.attempt_dag_proof(
        header, sketch_llm_call=llm, verify_fn=verify,
        probe_fn=lambda h, b: {"ok": True, "errors": None,
                               "body_line_offset": 3},
        sketch_attempts=1, repair_rounds=repair_rounds,
        abduce_lemmas=True, abduce_mode="theory",
        abduce_theory_trigger="always", abduce_theory_rounds=1,
        lemma_store=LemmaStore(),
        trace=lambda kind, **kw: events.append((kind, kw)))
    return [e[1].get("round") for e in events if e[0] == "assembled"]


def test_repair_rounds_zero_never_repairs_even_with_a_commit():
    """The boundary: 0 rounds must mean exactly one assembly, always."""
    assert _drive(commit=False, repair_rounds=0) == [0]
    assert _drive(commit=True, repair_rounds=0) == [0]


def test_repair_rounds_one_gains_exactly_one_on_commit():
    assert _drive(commit=False, repair_rounds=1) == [0, 1]
    assert _drive(commit=True, repair_rounds=1) == [0, 1, 2]


def test_bonus_is_capped_at_one_per_sketch():
    """Even with a commit available every round, only ONE bonus is given."""
    assert _drive(commit=True, repair_rounds=2) == [0, 1, 2, 3]
