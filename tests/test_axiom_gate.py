"""Regression tests for the axiom gate (2026-08-16).

Origin: `putnam_1981_a1` was scored `outcome="solved"` on a bare-baseline
run whose whole proof was `exact sorryAx _ true`. Both sorry guards
missed it and the campaign's `--audit` could not have caught it, since
`sorryAx` is not a declaration head.

These tests are all Lean-free: `check_axioms` takes its compile function
by injection, so every branch is exercised with fakes. The live
end-to-end check against a real `lake env lean` is marked `live`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backend.axiom_check import (  # noqa: E402
    ALLOWED_AXIOMS,
    build_probe_source,
    check_axioms,
    evaluate_report,
    extract_theorem_names,
    parse_axiom_report,
)

STD = "[propext, Classical.choice, Quot.sound]"


# --------------------------------------------------------------- parsing

def test_extracts_theorem_and_lemma_names():
    src = (
        "import Mathlib\n"
        "theorem foo : True := trivial\n"
        "lemma bar' : True := trivial\n"
        "private theorem baz : True := trivial\n"
        "noncomputable theorem qux : True := trivial\n"
    )
    assert extract_theorem_names(src) == ["foo", "bar'", "baz", "qux"]


def test_ignores_defs_and_abbrevs():
    src = ("abbrev putnam_x_solution : Nat := 8\n"
           "def helper : Nat := 1\n"
           "theorem real_one : True := trivial\n")
    assert extract_theorem_names(src) == ["real_one"]


def test_parses_both_report_forms():
    out = (f"'foo' depends on axioms: {STD}\n"
           "'bar' does not depend on any axioms\n")
    rep = parse_axiom_report(out)
    assert rep["foo"] == frozenset(ALLOWED_AXIOMS)
    assert rep["bar"] == frozenset()


def test_parses_sorryax_report():
    out = "'foo' depends on axioms: [sorryAx]\n"
    assert parse_axiom_report(out)["foo"] == frozenset({"sorryAx"})


def test_probe_source_appends_one_command_per_theorem():
    src = "theorem a : True := trivial\ntheorem b : True := trivial\n"
    probe = build_probe_source(src, ["a", "b"])
    assert probe.count("#print axioms") == 2
    # the original source must be preserved verbatim at the front
    assert probe.startswith(src.rstrip("\n"))


# -------------------------------------------------------------- decision

def test_standard_axioms_pass():
    ok, checked, offending, inconc, _ = evaluate_report(
        ["foo"], {"foo": frozenset(ALLOWED_AXIOMS)})
    assert ok and not offending and not inconc
    assert checked["foo"] == frozenset(ALLOWED_AXIOMS)


def test_no_axioms_passes():
    ok, _, _, _, _ = evaluate_report(["foo"], {"foo": frozenset()})
    assert ok


def test_sorryax_is_rejected():
    ok, _, offending, inconc, detail = evaluate_report(
        ["foo"], {"foo": frozenset({"sorryAx"})})
    assert not ok
    assert offending["foo"] == frozenset({"sorryAx"})
    assert not inconc          # a definite rejection, not a hiccup
    assert "sorryAx" in detail


def test_user_declared_axiom_is_rejected():
    """The OTHER open hole: `axiom foo_aux : <goal>` then `exact foo_aux`."""
    ok, _, offending, _, _ = evaluate_report(
        ["foo"], {"foo": frozenset(ALLOWED_AXIOMS | {"putnam_aux"})})
    assert not ok
    assert offending["foo"] == frozenset({"putnam_aux"})


def test_one_bad_theorem_fails_the_file():
    ok, _, offending, _, _ = evaluate_report(
        ["good", "bad"],
        {"good": frozenset(ALLOWED_AXIOMS), "bad": frozenset({"sorryAx"})})
    assert not ok and set(offending) == {"bad"}


def test_qualified_name_in_report_matches_short_name():
    ok, _, _, inconc, _ = evaluate_report(
        ["foo"], {"Ns.foo": frozenset(ALLOWED_AXIOMS)})
    assert ok and not inconc


# ------------------------------------------------- fails CLOSED, always

def test_missing_report_is_inconclusive_and_not_ok():
    ok, _, _, inconc, detail = evaluate_report(["foo"], {})
    assert not ok and inconc
    assert "foo" in detail


def test_source_with_no_named_theorem_is_not_applicable():
    """`example : … := rfl` and probe sources declare nothing to check.

    These PASS (rejecting them would break every diagnostic compile) but
    are flagged inconclusive so the pass is visible. Safe because a real
    solve always carries the benchmark's named theorem, and because the
    `sorryAx` source prefilter catches the unnamed case one layer up.
    """
    ok, _, _, inconc, detail = evaluate_report([], {})
    assert ok and inconc
    assert "not applicable" in detail


def test_unnamed_sorryax_still_caught_by_prefilter():
    """The hole the previous test could open, closed one layer up."""
    from backend.compile_verify import _SORRY_RE
    assert _SORRY_RE.search("example : True := by exact sorryAx _ true")


def test_probe_compile_failure_is_not_ok():
    res = check_axioms("theorem foo : True := trivial",
                       lambda src: (False, "error: boom"))
    assert not res.ok and res.inconclusive


def test_unparseable_report_is_not_ok():
    res = check_axioms("theorem foo : True := trivial",
                       lambda src: (True, "totally unrelated output"))
    assert not res.ok and res.inconclusive


def test_gate_passes_on_a_clean_report():
    res = check_axioms(
        "theorem foo : True := trivial",
        lambda src: (True, f"'foo' depends on axioms: {STD}"))
    assert res.ok and not res.offending and not res.inconclusive


def test_gate_rejects_the_verbatim_p1981a1_exploit():
    """The exact proof that was scored solved on 2026-08-16."""
    src = (
        "import Mathlib\n"
        "theorem putnam_1981_a1 : (1 : Nat) = 2 := by\n"
        "  exact sorryAx _ true\n"
    )
    res = check_axioms(
        src, lambda s: (True, "'putnam_1981_a1' depends on axioms: [sorryAx]"))
    assert not res.ok
    assert res.offending["putnam_1981_a1"] == frozenset({"sorryAx"})


def test_sorry_regex_prefilter_catches_sorryax():
    """Defence in depth: the cheap source scan should also see it."""
    from backend.compile_verify import _SORRY_RE
    assert _SORRY_RE.search("exact sorryAx _ true")
    assert _SORRY_RE.search("  sorry")
    assert _SORRY_RE.search("admit")
    # and must not fire on innocent text
    assert not _SORRY_RE.search("theorem no_sorries_here : True := trivial")


def test_sorry_warning_regex_accepts_lean_backtick_quoting():
    """MEASURED on toolchain v4.30.0-rc2, not assumed.

    Lean emits ``declaration uses `sorry` `` with backticks. The original
    pattern required straight quotes and so never matched — the implicit
    -sorry layer was dead code on this toolchain. Both quotings must work.
    """
    from backend.compile_verify import _SORRY_WARNING_RE
    backtick = ("C:\\x\\Try_96df8fe795b5.lean:3:8: "
                "warning: declaration uses `sorry`")
    straight = "foo.lean:3:8: warning: declaration uses 'sorry'"
    assert _SORRY_WARNING_RE.search(backtick)
    assert _SORRY_WARNING_RE.search(straight)


def test_print_axioms_output_from_the_real_probe_parses():
    """Verbatim stdout captured from the live probe on 2026-08-16."""
    out = ("'probe_synthetic_true' depends on axioms: [sorryAx]\n"
           "'probe_synthetic_false' depends on axioms: [sorryAx]\n")
    rep = parse_axiom_report(out)
    assert rep["probe_synthetic_true"] == frozenset({"sorryAx"})
    assert rep["probe_synthetic_false"] == frozenset({"sorryAx"})
    ok, _, offending, _, _ = evaluate_report(
        ["probe_synthetic_true", "probe_synthetic_false"], rep)
    assert not ok and len(offending) == 2


@pytest.mark.live
def test_live_sorryax_is_rejected_end_to_end():
    """Real `lake env lean`. Slow (minutes); excluded from the fast suite."""
    from backend.compile_verify import compile_lean

    root = Path(__file__).resolve().parents[1] / "lean"
    src = ("import Mathlib\n\n"
           "theorem probe_gate_sorryax : (1 : Nat) = 2 := by\n"
           "  exact sorryAx _ true\n")
    res = compile_lean(src, timeout_s=2400, lean_project_root=root)
    assert not res.ok, "sorryAx must never be scored ok"
