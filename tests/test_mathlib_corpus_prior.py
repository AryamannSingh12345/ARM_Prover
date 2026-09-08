"""Tests for the Mathlib corpus-prior pipeline.

Covers:
  - extract_lean_declarations on a tiny synthetic file (PART 12.1)
  - split_tactic_script on a 3-line proof (PART 12.2)
  - classify + premise extraction for `have h := Nat.gcd_mul_lcm n k`
    (PART 12.3, also exercised in test_proof_prior_pipeline.py — kept
    here so the corpus suite stands alone)
  - build_mathlib_proof_prior on a fake 2-file Mathlib tree (PART 12.4)
  - Output JSONL round-trips through ProofPriorIndex.load (PART 12.5)
  - merge_proof_priors combines counts and recomputes probabilities
    (PART 12.6)
  - Exclusion removes a named theorem (PART 12.7)
  - Theorem names are never feature-key components (PART 12.8)
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from search.corpus_features import extract_theorem_features
from search.proof_prior import ProofPriorIndex, feature_key
from search.proof_script_extract import (
    extract_lean_declarations, split_tactic_script,
)
from search.tactic_classify import classify_tactic, extract_used_premises


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


# ---------------------------------------------------------------------------
# Declaration + tactic split (PART 12.1, 12.2)
# ---------------------------------------------------------------------------

def test_extract_simple_theorem():
    src = "theorem foo : True := by trivial\n"
    decls = extract_lean_declarations(src, "fake.lean")
    assert len(decls) == 1
    assert decls[0].declaration_name == "foo"
    assert decls[0].declaration_kind == "theorem"
    assert decls[0].proof_kind == "tactic"
    assert "trivial" in decls[0].proof_text


def test_extract_multiple_declarations_with_attrs():
    src = (
        "@[simp]\n"
        "theorem a : 1 = 1 := by rfl\n"
        "\n"
        "lemma b (n : ℕ) : n + 0 = n := by\n"
        "  simp\n"
        "\n"
        "example : True := trivial\n"   # term-mode
    )
    decls = extract_lean_declarations(src, "fake.lean")
    names = [d.declaration_name for d in decls]
    assert names == ["a", "b", "example"] or set(names) >= {"a", "b"}
    kinds = {d.declaration_name: d.proof_kind for d in decls}
    assert kinds["a"] == "tactic"
    assert kinds["b"] == "tactic"


def test_split_tactic_script_three_lines():
    body = (
        "  intro x\n"
        "  simp\n"
        "  omega\n"
    )
    tactics = split_tactic_script(body)
    assert tactics == ["intro x", "simp", "omega"]


def test_split_tactic_script_skips_bullets_and_comments():
    body = (
        "  -- a comment\n"
        "  · simp [foo]\n"
        "  intro x\n"
        "\n"
        "  exact rfl\n"
    )
    tactics = split_tactic_script(body)
    # The bullet's inner tactic survives (with `·` stripped).
    assert "simp [foo]" in tactics
    assert "intro x" in tactics
    assert "exact rfl" in tactics
    # No bare bullet character.
    for t in tactics:
        assert not t.startswith("·")
        assert not t.startswith("--")


# ---------------------------------------------------------------------------
# Classify + premise extraction (PART 12.3 — corpus-suite copy)
# ---------------------------------------------------------------------------

def test_classify_have_nat_gcd_mul_lcm():
    t = "have h := Nat.gcd_mul_lcm n k"
    assert classify_tactic(t) == "have_premise"
    assert extract_used_premises(t)[0] == "Nat.gcd_mul_lcm"


# ---------------------------------------------------------------------------
# Build miner end-to-end on a fake Mathlib tree (PART 12.4 + 12.5)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_mathlib(tmp_path: Path) -> Path:
    """Two .lean files under a fake Mathlib tree."""
    root = tmp_path / "Mathlib"
    (root / "Data" / "Nat").mkdir(parents=True)
    (root / "Algebra").mkdir(parents=True)
    (root / "Analysis").mkdir(parents=True)
    (root / "Topology").mkdir(parents=True)
    (root / "NumberTheory").mkdir(parents=True)
    (root / "Data" / "Nat" / "Gcd.lean").write_text(
        "import Mathlib\n"
        "namespace Nat\n"
        "theorem gcd_mul_lcm_demo (n k : ℕ) : n * k = Nat.gcd n k * Nat.lcm n k := by\n"
        "  have h := Nat.gcd_mul_lcm n k\n"
        "  omega\n"
        "\n"
        "theorem simple_demo : True := by trivial\n"
        "end Nat\n",
        encoding="utf-8",
    )
    (root / "Algebra" / "Ring.lean").write_text(
        "import Mathlib\n"
        "theorem norm_num_demo : 2 + 2 = 4 := by norm_num\n"
        "\n"
        "theorem decide_demo : 1 < 5 := by decide\n",
        encoding="utf-8",
    )
    return root


def test_mathlib_miner_end_to_end(fake_mathlib: Path, tmp_path: Path):
    miner = _load_module("_miner", SCRIPTS / "build_mathlib_proof_prior.py")
    out = tmp_path / "prior.jsonl"
    stats = miner.build(
        mathlib_root=fake_mathlib,
        out_path=out,
        max_files=None, max_theorems=None,
        include_dirs=[], exclude_dirs=[],
        min_proof_lines=1, max_proof_lines=80,
        source_name="mathlib_corpus",
        excluded_ids=set(), excluded_regexes=[],
        progress_every=0,
        smoothing_alpha=1e-3,
        write_sidecar=True,
    )
    assert stats["total_files_seen"] >= 2
    assert stats["tactic_proofs_seen"] >= 4
    assert stats["tactics_extracted"] >= 4
    assert stats["moves_recorded"] > 0
    # The synthesised premise-prior move for Nat.gcd_mul_lcm must exist.
    body = out.read_text(encoding="utf-8")
    assert "Nat.gcd_mul_lcm" in body
    assert "mathlib_corpus" in body

    # Output is reloadable.
    idx = ProofPriorIndex.load(out)
    assert idx._by_key, "reloaded index must have at least one feature key"
    # `decide_demo` classified as `decide`, not `unknown`.
    found_decide = any(
        any(m.move.tactic_class == "decide" for m in bucket.values())
        for bucket in idx._by_key.values()
    )
    assert found_decide, "decide should be classified as decide"


def test_mathlib_miner_exclusion_drops_named_theorem(
    fake_mathlib: Path, tmp_path: Path,
):
    miner = _load_module("_miner_excl", SCRIPTS / "build_mathlib_proof_prior.py")
    out = tmp_path / "prior_excl.jsonl"
    stats = miner.build(
        mathlib_root=fake_mathlib,
        out_path=out,
        max_files=None, max_theorems=None,
        include_dirs=[], exclude_dirs=[],
        min_proof_lines=1, max_proof_lines=80,
        source_name="mathlib_corpus",
        excluded_ids={"gcd_mul_lcm_demo"},
        excluded_regexes=[],
        progress_every=0,
        smoothing_alpha=1e-3,
        write_sidecar=True,
    )
    assert stats["rows_excluded"] == 1
    body = out.read_text(encoding="utf-8")
    # The Nat.gcd_mul_lcm citation only appears inside gcd_mul_lcm_demo's
    # proof, so excluding the demo also removes the premise from the
    # output. (Other tests use the real Nat.gcd_mul_lcm so this checks
    # the holdout is doing real work.)
    assert "Nat.gcd_mul_lcm" not in body


def test_mathlib_miner_name_regex_exclusion(fake_mathlib: Path, tmp_path: Path):
    miner = _load_module("_miner_re", SCRIPTS / "build_mathlib_proof_prior.py")
    out = tmp_path / "prior_re.jsonl"
    stats = miner.build(
        mathlib_root=fake_mathlib,
        out_path=out,
        max_files=None, max_theorems=None,
        include_dirs=[], exclude_dirs=[],
        min_proof_lines=1, max_proof_lines=80,
        source_name="mathlib_corpus",
        excluded_ids=set(),
        excluded_regexes=[re.compile(r"_demo$")],
        progress_every=0,
        smoothing_alpha=1e-3,
        write_sidecar=True,
    )
    # All four demos end in `_demo` → all excluded.
    assert stats["rows_excluded"] >= 4
    assert stats["tactic_proofs_seen"] == 0


# ---------------------------------------------------------------------------
# Merge (PART 12.6)
# ---------------------------------------------------------------------------

def test_merge_proof_priors_combines_counts(tmp_path: Path):
    miner = _load_module("_miner_m", SCRIPTS / "build_mathlib_proof_prior.py")
    merger = _load_module("_merger", SCRIPTS / "merge_proof_priors.py")
    # Two tiny inputs whose rows share at least one (feature_key, template).
    a_root = tmp_path / "A_Mathlib"
    b_root = tmp_path / "B_Mathlib"
    for root in (a_root, b_root):
        for sub in ("Data", "Algebra", "Analysis", "Topology", "NumberTheory"):
            (root / sub).mkdir(parents=True)
        (root / "Data" / "x.lean").write_text(
            "theorem t : True := by trivial\n", encoding="utf-8")

    a_out = tmp_path / "a.jsonl"
    b_out = tmp_path / "b.jsonl"
    miner.build(
        mathlib_root=a_root, out_path=a_out,
        max_files=None, max_theorems=None, include_dirs=[], exclude_dirs=[],
        min_proof_lines=1, max_proof_lines=80,
        source_name="mathlib_corpus", excluded_ids=set(), excluded_regexes=[],
        progress_every=0, smoothing_alpha=1e-3, write_sidecar=False,
    )
    miner.build(
        mathlib_root=b_root, out_path=b_out,
        max_files=None, max_theorems=None, include_dirs=[], exclude_dirs=[],
        min_proof_lines=1, max_proof_lines=80,
        source_name="solved_jsonl", excluded_ids=set(), excluded_regexes=[],
        progress_every=0, smoothing_alpha=1e-3, write_sidecar=False,
    )
    merged = tmp_path / "merged.jsonl"
    merge_stats = merger.merge([a_out, b_out], merged)
    assert merge_stats["merged_moves"] >= 1
    # The merged record's `source` is either the surviving single-source
    # value or "merged"; in any case the tag list must mention both
    # contributing sources.
    body = merged.read_text(encoding="utf-8")
    assert "mathlib_corpus" in body
    assert "solved_jsonl" in body
    # Probabilities are renormalised on save — every row's
    # prior_probability is in (0, 1].
    for line in body.splitlines():
        row = json.loads(line)
        p = float(row["prior_probability"])
        assert 0.0 < p <= 1.0
    # Round-trips through ProofPriorIndex.load.
    idx = ProofPriorIndex.load(merged)
    assert idx._by_key


# ---------------------------------------------------------------------------
# Holdout-by-key invariance (PART 12.8)
# ---------------------------------------------------------------------------

def test_mathlib_miner_canonicalises_nat_bare_premises(tmp_path: Path):
    """A fake Mathlib Data/Nat file whose proof body contains
    `rw [..., gcd_mul_lcm]` should produce at least one premise-prior
    row whose `premise` is the canonical `Nat.gcd_mul_lcm` and whose
    `tags` include `rw_lemma` (PART A/B/C end-to-end)."""
    miner = _load_module("_miner_canon",
                          SCRIPTS / "build_mathlib_proof_prior.py")
    root = tmp_path / "Mathlib"
    # The miner expects the standard Mathlib subdirectories to exist.
    for sub in ("Data", "Algebra", "Analysis", "Topology", "NumberTheory"):
        (root / sub).mkdir(parents=True)
    (root / "Data" / "Nat").mkdir(parents=True, exist_ok=True)
    (root / "Data" / "Nat" / "Gcd.lean").write_text(
        "import Mathlib\n"
        "namespace Nat\n"
        "theorem demo (m n : ℕ) (h : Nat.gcd m n = 1) :\n"
        "    m * n = Nat.lcm m n := by\n"
        "  rw [← one_mul (lcm m n), ← h.gcd_eq_one, gcd_mul_lcm]\n"
        "end Nat\n",
        encoding="utf-8",
    )
    out = tmp_path / "prior.jsonl"
    miner.build(
        mathlib_root=root, out_path=out,
        max_files=None, max_theorems=None,
        include_dirs=[], exclude_dirs=[],
        min_proof_lines=1, max_proof_lines=80,
        source_name="mathlib_corpus",
        excluded_ids=set(), excluded_regexes=[],
        progress_every=0, smoothing_alpha=1e-3,
        write_sidecar=False,
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    # The canonicalised premise-prior row exists.
    canonicalised = [
        r for r in rows
        if r.get("premise") == "Nat.gcd_mul_lcm"
        and r.get("tactic_template") == "have h := Nat.gcd_mul_lcm"
    ]
    assert canonicalised, (
        "expected a `have h := Nat.gcd_mul_lcm` premise-prior row; "
        f"got rows={[r['tactic_template'] for r in rows][:5]}"
    )
    # And the rw_lemma tag was applied (came from inside `rw [...]`).
    assert any("rw_lemma" in (r.get("tags") or []) for r in canonicalised), (
        "expected the `rw_lemma` tag on the canonicalised row; "
        f"sample tags={[r.get('tags') for r in canonicalised]}"
    )
    # Local projection from the same rw bracket must not appear as a
    # premise anywhere in the output.
    assert all(r.get("premise") != "h.gcd_eq_one" for r in rows)


def test_theorem_name_is_never_part_of_feature_key():
    """`extract_theorem_features` must derive the same key from a
    statement regardless of the theorem name. Renaming `foo` to `bar`
    while keeping the proposition identical must not change the key."""
    s1 = "theorem foo (n : ℕ) : Nat.gcd n n = n := by sorry"
    s2 = "theorem bar (n : ℕ) : Nat.gcd n n = n := by sorry"
    f1 = extract_theorem_features(s1)
    f2 = extract_theorem_features(s2)
    assert feature_key(f1) == feature_key(f2)
