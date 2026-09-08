"""Regression tests for premise-seeded imports (2026-08-18).

Origin: `p1981a1_dag_v1`. Import selection ran BEFORE premise retrieval
and from the STATEMENT alone, so the names BM25 had just surfaced never
informed the import set. The LLM guessed `Polynomial.Basic` +
`Finset.Interval` + `Deriv.Basic` for a 5-adic valuation problem,
`Nat.Prime` was never imported at all, and `Nat.Factorization.Defs`
arrived at t=13522s of a 14172s run — via reactive refresh, after the
round that needed it had already failed.

`modules_for_premises` closes the loop with the declaration graph that
already existed. Properties pinned here:

  * deterministic — graph lookup only, no LLM, no compiles;
  * environment-derived — no module names in code, so the no-hardcoding
    rule is untouched;
  * safe on failure — unresolvable names contribute NOTHING rather than
    a guess, and a missing graph yields [] (caller behaviour unchanged).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search import decl_module_index as DMI  # noqa: E402
from search.decl_module_index import modules_for_premises  # noqa: E402


def test_empty_and_none_are_safe():
    assert modules_for_premises(None) == []
    assert modules_for_premises([]) == []


def test_unresolvable_names_contribute_nothing(monkeypatch):
    """A hallucinated premise must add NO import, not a guessed one."""
    monkeypatch.setattr(DMI, "resolve_decl_module", lambda n: None)
    assert modules_for_premises(["Totally.Made.Up", "Nope"]) == []


def test_resolves_and_dedupes_preserving_rank(monkeypatch):
    table = {
        "Nat.Prime": "Mathlib.Data.Nat.Prime.Defs",
        "Nat.factorization": "Mathlib.Data.Nat.Factorization.Defs",
        "Nat.Prime.two_le": "Mathlib.Data.Nat.Prime.Defs",   # dup module
    }
    monkeypatch.setattr(DMI, "resolve_decl_module", table.get)
    out = modules_for_premises(
        ["Nat.Prime", "Nat.factorization", "Nat.Prime.two_le"])
    assert out == ["Mathlib.Data.Nat.Prime.Defs",
                   "Mathlib.Data.Nat.Factorization.Defs"]


def test_limit_caps_and_keeps_highest_ranked(monkeypatch):
    monkeypatch.setattr(DMI, "resolve_decl_module", lambda n: f"Mod.{n}")
    out = modules_for_premises(["a", "b", "c", "d"], limit=2)
    assert out == ["Mod.a", "Mod.b"]


def test_missing_graph_yields_empty(monkeypatch):
    """No graph archive => [] , so the caller's behaviour is unchanged."""
    monkeypatch.setattr(DMI, "_load", lambda: False)
    monkeypatch.setattr(DMI, "_INDEX", None)
    assert modules_for_premises(["Nat.Prime"]) == []


def test_resolution_is_faithful_to_its_input(monkeypatch):
    """GIVEN the right names, the right modules come back.

    This is the mechanism's actual guarantee, and it is narrower than it
    first appears. Seeding is only as good as retrieval, and
    `premise_retrieval` is BM25 over declaration NAMES, not statements.
    Measured against the real graph on 2026-08-18:

      putnam_1967_b5 -> Nat.Choose.Sum, Nat.Choose.Vandermonde,
                        BigOperators.Group.Finset.Basic     (right)
      putnam_1981_a1 -> Coxeter.Matrix, SpectralSequence,
                        ModularForms, Lindemann             (noise;
                        Nat.Prime never surfaced at all)

    Hence the flag defaults OFF — see test_flag_defaults_off_with_reason.
    """
    table = {
        "Nat.Prime": "Mathlib.Data.Nat.Prime.Defs",
        "Nat.factorization": "Mathlib.Data.Nat.Factorization.Defs",
        "Nat.factorization_le_iff_dvd": "Mathlib.Data.Nat.Factorization.Defs",
    }
    monkeypatch.setattr(DMI, "resolve_decl_module", table.get)
    out = modules_for_premises(list(table))
    assert "Mathlib.Data.Nat.Prime.Defs" in out
    assert "Mathlib.Data.Nat.Factorization.Defs" in out


def test_flag_defaults_off_with_reason():
    """OFF by default: measured noise on 2 of 3 problems checked.

    Shipping this ON would have added twelve irrelevant imports to
    putnam_1981_a1 and putnam_1963_a2 — elaboration cost for nothing —
    while helping putnam_1967_b5. The mechanism is sound; the retrieval
    feeding it is not yet good enough to enable unconditionally.
    """
    src = (Path(__file__).resolve().parents[1]
           / "src" / "eval" / "run_dag.py").read_text(encoding="utf-8")
    i = src.index('"--premise-seeded-imports"')
    decl = src[i:i + 200]
    assert "BooleanOptionalAction, default=False" in decl
    assert '"--premise-import-limit"' in src
    # the reason must travel with the flag, not just this test
    assert "OFF BY DEFAULT because" in src[i:i + 2000]


def test_seeding_is_gated_and_skipped_when_imports_pinned():
    """Two safety properties of the wiring itself."""
    src = (Path(__file__).resolve().parents[1]
           / "src" / "eval" / "run_dag.py").read_text(encoding="utf-8")
    i = src.index("PREMISE-SEEDED IMPORTS")
    tail = src[i:i + 3000]
    # explicit --lean-imports wins
    assert "not args.lean_imports" in tail
    # the merged set is statement-gated before adoption
    assert "compile_lean(f\"{merged}" in tail
    assert "_ERROR_LINE_RE.search(gate.errors" in tail
    # a rejected seed leaves verify_imports untouched
    assert "rejected=added" in tail
