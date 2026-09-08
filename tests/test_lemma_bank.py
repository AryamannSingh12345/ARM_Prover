"""Point 9 — provenance-carrying lemma bank + eval-split/chronology filter.

Covers `search.lemma_bank` (statement hashing, provenance records,
semantic dedup, the eval filter) and the `_append_lemma_library` wiring
that writes the JSONL sidecar without changing the .lean append.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from search import lemma_bank as lb
from search.proof_dag import _append_lemma_library


# ---------------- statement normalization / hashing -------------------------

def test_split_decl_extracts_name_statement_proof():
    name, stmt, proof = lb.split_decl(
        "lemma aux_foo (x : ℝ) : x = x := by rfl")
    assert name == "aux_foo"
    assert stmt == "lemma aux_foo (x : ℝ) : x = x"
    assert proof == "by rfl"


def test_split_decl_no_proof():
    name, stmt, proof = lb.split_decl("theorem bar : True")
    assert name == "bar"
    assert proof == ""


def test_normalize_strips_name_and_collapses_ws():
    n1 = lb.normalize_statement("lemma aux_foo (x : ℝ)  :  x = x")
    n2 = lb.normalize_statement("lemma renamed_foo (x : ℝ) : x = x")
    # Name dropped, whitespace collapsed → identical normal forms.
    assert n1 == n2 == "(x : ℝ) : x = x"


def test_normalize_strips_comments():
    n = lb.normalize_statement(
        "theorem t /- block -/ : True -- trailing")
    assert "block" not in n and "trailing" not in n


def test_hash_same_shape_different_name_collapses():
    """The core Point-9 improvement over name-only dedup: two lemmas with
    the same shape under different names share a statement hash."""
    a = "lemma aux_foo (x : ℝ) : x = x := by rfl"
    b = "lemma totally_different (x : ℝ) : x = x := by simp"
    ha = lb.statement_hash(lb.normalize_statement(lb.split_decl(a)[1]))
    hb = lb.statement_hash(lb.normalize_statement(lb.split_decl(b)[1]))
    assert ha == hb


def test_hash_differs_on_different_statements():
    a = lb.statement_hash(lb.normalize_statement("lemma f : True"))
    b = lb.statement_hash(lb.normalize_statement("lemma f : False"))
    assert a != b


# ---------------- records + provenance --------------------------------------

def test_build_records_carries_provenance_and_defaults():
    recs = lb.build_records(
        ["lemma aux_foo : True := by trivial"],
        tag="theory-abduced",
        provenance={
            "source_problem": "putnam_1966_b5",
            "source_split": "putnam",
            "run_id": "run42",
            "model": "claude-opus-4-8",
            "verified_mathlib_commit": "abc123",
            "imports": ["import Mathlib"],
        })
    assert len(recs) == 1
    r = recs[0]
    assert r.source_problem == "putnam_1966_b5"
    assert r.source_split == "putnam"
    assert r.run_id == "run42"
    assert r.model == "claude-opus-4-8"
    assert r.verified_mathlib_commit == "abc123"
    assert r.imports == ["import Mathlib"]
    assert r.tag == "theory-abduced"
    assert r.ts  # timestamp present
    # Conservative default: never eval-eligible unless promoted.
    assert r.allowed_for_eval is False


def test_build_records_none_provenance_is_safe():
    recs = lb.build_records(["lemma x : True := by trivial"])
    assert len(recs) == 1
    assert recs[0].source_split == "unknown"
    assert recs[0].source_problem is None
    assert recs[0].allowed_for_eval is False


def test_build_records_skips_blank():
    assert lb.build_records(["", "   "]) == []


# ---------------- append_records: semantic dedup ----------------------------

def test_append_records_dedups_by_statement_hash(tmp_path):
    jl = tmp_path / "bank.jsonl"
    a = lb.build_records(["lemma aux_foo (x : ℝ) : x = x := by rfl"])
    # Same shape, different name + proof → same hash → skipped.
    b = lb.build_records(["lemma renamed (x : ℝ) : x = x := by simp"])
    assert lb.append_records(jl, a) == 1
    assert lb.append_records(jl, b) == 0
    lines = jl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_append_records_keeps_distinct(tmp_path):
    jl = tmp_path / "bank.jsonl"
    lb.append_records(jl, lb.build_records(["lemma a : True := by trivial"]))
    lb.append_records(jl, lb.build_records(["lemma b : False → True := "
                                            "fun _ => trivial"]))
    assert len(jl.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_append_records_never_raises_on_bad_path(tmp_path):
    # A directory is not a writable file — must swallow, return 0.
    assert lb.append_records(tmp_path, lb.build_records(
        ["lemma z : True := by trivial"])) == 0


# ---------------- load_bank: the eval guards --------------------------------

def _seed(tmp_path) -> Path:
    jl = tmp_path / "bank.jsonl"
    recs = []
    recs += lb.build_records(["lemma test_lemma : True := by trivial"],
                             provenance={"source_split": "test",
                                         "source_problem": "p_test"})
    recs += lb.build_records(["lemma valid_lemma : 1 = 1 := by rfl"],
                             provenance={"source_split": "valid",
                                         "source_problem": "p_valid"})
    # Manually vary timestamps for chronology tests.
    recs[0].ts = "2026-01-01T00:00:00+00:00"
    recs[1].ts = "2026-06-01T00:00:00+00:00"
    lb.append_records(jl, recs)
    return jl


def test_load_bank_excludes_split(tmp_path):
    jl = _seed(tmp_path)
    kept = lb.load_bank(jl, exclude_splits=("test",))
    assert {r.source_split for r in kept} == {"valid"}


def test_load_bank_allow_splits(tmp_path):
    jl = _seed(tmp_path)
    kept = lb.load_bank(jl, allow_splits=("valid",))
    assert {r.source_split for r in kept} == {"valid"}


def test_load_bank_chronology(tmp_path):
    jl = _seed(tmp_path)
    # Cutoff between the two records → only the earlier survives.
    kept = lb.load_bank(jl, before_ts="2026-03-01T00:00:00+00:00")
    assert {r.source_problem for r in kept} == {"p_test"}


def test_load_bank_require_allowed(tmp_path):
    jl = _seed(tmp_path)
    assert lb.load_bank(jl, require_allowed=True) == []


# ---------------- mathlib commit from manifest ------------------------------

def test_mathlib_commit_reads_pin():
    root = Path(__file__).resolve().parents[1]
    commit = lb.mathlib_commit_from_manifest(root / "lean")
    # The repo pins Mathlib; the manifest must yield a 40-char sha.
    assert commit is not None and len(commit) == 40


def test_mathlib_commit_missing_manifest_is_none(tmp_path):
    assert lb.mathlib_commit_from_manifest(tmp_path) is None


# ---------------- wiring: _append_lemma_library writes the sidecar ----------

def test_append_lemma_library_writes_sidecar(tmp_path):
    lib = tmp_path / "invented_lemmas.lean"
    _append_lemma_library(
        str(lib), "theory-abduced for leaves ['h0']",
        ["lemma aux_h0 : True := by trivial"],
        provenance={"source_problem": "pid1", "source_split": "test",
                    "run_id": "r1", "model": "m1"})
    # .lean append unchanged in spirit.
    assert "aux_h0" in lib.read_text(encoding="utf-8")
    # sidecar exists with the provenance record.
    side = lib.with_suffix(".jsonl")
    assert side.exists()
    rec = json.loads(side.read_text(encoding="utf-8").strip().splitlines()[0])
    assert rec["source_problem"] == "pid1"
    assert rec["source_split"] == "test"
    assert rec["allowed_for_eval"] is False
    assert rec["statement_hash"]


def test_append_lemma_library_sidecar_without_provenance(tmp_path):
    lib = tmp_path / "invented_lemmas.lean"
    _append_lemma_library(str(lib), "tag", ["lemma q : True := by trivial"])
    side = lib.with_suffix(".jsonl")
    assert side.exists()
    rec = json.loads(side.read_text(encoding="utf-8").strip().splitlines()[0])
    assert rec["source_split"] == "unknown"
    assert rec["source_problem"] is None
