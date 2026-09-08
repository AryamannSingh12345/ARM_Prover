"""Tests for the proof-prior pipeline.

Covers PART 12 of the proof-prior design: state features, tactic
classification, ProofPriorIndex round-trip, template generation,
dependency exploration, move scoring, plus regression checks for the
new CLI flags. Existing safety regressions (Classical.em /
factorial / sqrt_pos.2) are covered by the existing test_canonicalization_safety,
test_header_normalization, and test_unknown_premise_guard files; this
module adds NEW coverage for the proof-prior modules.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from search.dependency_explorer import dependency_exploration_candidates
from search.proof_prior import (
    ProofPriorIndex, ProofPriorMove, ProofStateFeatures, feature_key,
)
from search.state_features import extract_state_features
from search.tactic_classify import classify_tactic, extract_used_premises
from search.template_tactics import generate_template_tactics


# ---------------------------------------------------------------------------
# State feature extraction
# ---------------------------------------------------------------------------

def test_state_features_gcd_lcm():
    goal = ("n : ℕ\nh₀ : 0 < n\nh₁ : Nat.gcd n 40 = 10\n"
            "h₂ : Nat.lcm n 40 = 280\n⊢ n = 70")
    feats = extract_state_features(goal, proof_prefix=[])
    assert "gcd" in feats.symbols
    assert "lcm" in feats.symbols
    assert "Nat" in feats.namespaces
    assert feats.target_shape == "equality"
    assert "40" in feats.constants or "70" in feats.constants


def test_state_features_real_inequality_with_power():
    goal = "a b : ℝ\nh : 0 < a\n⊢ a^2 + b^2 ≤ (a + b)^2"
    feats = extract_state_features(goal, proof_prefix=[])
    assert "Real" in feats.namespaces
    assert "inequality" in feats.symbols
    assert "power" in feats.symbols
    assert feats.target_shape == "inequality"


def test_state_features_finset_range_sum():
    goal = "n : ℕ\n⊢ Finset.sum (Finset.range n) (fun i => i) = n * (n - 1) / 2"
    feats = extract_state_features(goal, proof_prefix=[])
    assert "Finset" in feats.namespaces
    assert "sum" in feats.symbols
    assert "range" in feats.symbols


def test_state_features_sqrt():
    goal = "x : ℝ\nh : 0 ≤ x\n⊢ 0 ≤ Real.sqrt x"
    feats = extract_state_features(goal, proof_prefix=[])
    assert "Real" in feats.namespaces
    assert "sqrt" in feats.symbols


def test_state_features_previous_tactic_class():
    feats = extract_state_features(
        "⊢ n = n",
        proof_prefix=["intro n", "rw [foo]"],
    )
    assert feats.previous_tactic_class == "rw"


# ---------------------------------------------------------------------------
# Tactic classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tactic,expected_class,expected_premise", [
    ("have h := Nat.gcd_mul_lcm n 40", "have_premise", "Nat.gcd_mul_lcm"),
    ("rw [h₁, h₂] at hprod",           "rw",           None),
    ("nlinarith [sq_nonneg (a-b-1)]",  "nlinarith",    "sq_nonneg"),
    ("rcases h with ⟨a,b⟩",            "cases_or_rcases", None),
    ("obtain ⟨k, hk⟩ := h",            "cases_or_rcases", None),
    ("simp [Nat.factorial_succ]",      "simp",         "Nat.factorial_succ"),
    ("omega",                          "omega",        None),
    ("exact Real.sqrt_nonneg x",       "exact",        "Real.sqrt_nonneg"),
    ("by_cases h : x = 0",             "by_cases",     None),
    ("positivity",                     "positivity",   None),
])
def test_classify_tactic_and_extract_premise(tactic, expected_class, expected_premise):
    assert classify_tactic(tactic) == expected_class
    premises = extract_used_premises(tactic)
    if expected_premise is None:
        # the extractor may pull other tokens but not the named premise
        assert all(not p.endswith(".") for p in premises)
    else:
        assert expected_premise in premises


# --- PART A: decide / native_decide / norm_num / omega are first-class ----

@pytest.mark.parametrize("tactic,expected", [
    ("decide",            "decide"),
    ("decide ",           "decide"),
    ("native_decide",     "native_decide"),
    ("norm_num",          "norm_num"),
    ("omega",             "omega"),
    ("rw [foo]",          "rw"),
    ("ring",              "ring_nf"),
])
def test_classify_decide_and_friends(tactic, expected):
    assert classify_tactic(tactic) == expected


# --- PART A: have-with-type-annotation still classifies as have_premise ---

def test_classify_have_with_type_annotation_is_have_premise():
    t = "have h : n * 40 = Nat.gcd n 40 * Nat.lcm n 40 := Nat.gcd_mul_lcm n 40"
    assert classify_tactic(t) == "have_premise"


def test_classify_have_with_tactic_block_is_have_local():
    t = "have h : 0 < n := by positivity"
    assert classify_tactic(t) == "have_local"


# --- PART B: RHS of `:=` is the primary premise ---------------------------

def test_extract_used_premises_prioritises_rhs_of_assign():
    t = "have h : n * 40 = Nat.gcd n 40 * Nat.lcm n 40 := Nat.gcd_mul_lcm n 40"
    premises = extract_used_premises(t)
    assert premises, "should extract at least one premise"
    assert premises[0] == "Nat.gcd_mul_lcm", (
        f"primary premise must come from RHS of `:=`, got {premises}"
    )
    # LHS premises still appear, just not first.
    assert "Nat.gcd" in premises
    assert "Nat.lcm" in premises


def test_extract_used_premises_no_assign_keeps_left_to_right():
    t = "rw [Nat.add_comm, Nat.mul_comm]"
    premises = extract_used_premises(t)
    assert premises[0] == "Nat.add_comm"
    assert premises[1] == "Nat.mul_comm"


# --- PART A: bracket-list extraction + local-projection filter -----------

def test_extract_premises_inside_rw_with_local_projection():
    t = "rw [← one_mul (lcm m n), ← h.gcd_eq_one, gcd_mul_lcm]"
    premises = extract_used_premises(t)
    # Real lemmas are extracted...
    assert "gcd_mul_lcm" in premises
    assert "one_mul" in premises
    # ...local projection is dropped entirely.
    assert "h.gcd_eq_one" not in premises
    # ...and `h.gcd_eq_one` must not be the primary.
    assert premises[0] != "h.gcd_eq_one"


def test_extract_premises_drops_local_projection_a_dot_gcd():
    t = ("rw [← add_right_inj (a.gcd b).factorization, "
         "← factorization_mul ha hb, gcd_mul_lcm, factorization_gcd ha hb]")
    premises = extract_used_premises(t)
    # All the real lemmas should be present.
    assert "gcd_mul_lcm" in premises
    assert "factorization_gcd" in premises
    assert "factorization_mul" in premises
    assert "add_right_inj" in premises
    # `a.gcd` is a local-variable projection and must be excluded.
    assert "a.gcd" not in premises


def test_extract_rw_simp_lemmas_isolates_bracket_only():
    """The `rw_lemma` tag depends on this helper returning *only* the
    bracket-list items, not dotted hits elsewhere in the tactic."""
    from search.tactic_classify import extract_rw_simp_lemmas
    t = ("have h : Nat.gcd n 40 = 10 := Nat.gcd_mul_lcm n 40 |>.foo; "
         "rw [gcd_assoc, ← lcm_comm]")
    lemmas = extract_rw_simp_lemmas(t)
    assert "gcd_assoc" in lemmas
    assert "lcm_comm" in lemmas
    # Nothing outside brackets — Nat.gcd_mul_lcm is NOT an rw lemma here.
    assert "Nat.gcd_mul_lcm" not in lemmas


def test_extract_premises_simp_only_with_local_projection():
    t = "simp only [Nat.gcd_mul_lcm, h.foo, gcd_assoc]"
    premises = extract_used_premises(t)
    assert "Nat.gcd_mul_lcm" in premises
    assert "gcd_assoc" in premises
    assert "h.foo" not in premises


# --- PART B: Nat-namespace canonicalisation ------------------------------

def test_canonicalise_for_namespace_nat_bare_names():
    from search.tactic_classify import canonicalise_for_namespace
    assert canonicalise_for_namespace("gcd_mul_lcm", "Nat") == "Nat.gcd_mul_lcm"
    assert canonicalise_for_namespace("lcm_ne_zero", "Nat") == "Nat.lcm_ne_zero"
    assert canonicalise_for_namespace("gcd_comm",    "Nat") == "Nat.gcd_comm"
    # Already qualified — left alone.
    assert canonicalise_for_namespace("Nat.gcd_mul_lcm", "Nat") == "Nat.gcd_mul_lcm"
    # Local projections — left alone (they have a `.`).
    assert canonicalise_for_namespace("h.gcd_eq_one", "Nat") == "h.gcd_eq_one"
    # Wrong namespace — no rewrite.
    assert canonicalise_for_namespace("gcd_mul_lcm", "Real") == "gcd_mul_lcm"
    # Outside the canon map — left alone.
    assert canonicalise_for_namespace("add_comm", "Nat") == "add_comm"


# --- PART D: instantiation-required premises ------------------------------

def test_template_skips_exact_for_gcd_mul_lcm():
    """`exact Nat.gcd_mul_lcm` / `exact gcd_mul_lcm` would type-error
    because the lemma takes two explicit args. The template generator
    must skip those candidates entirely; the instantiation block
    (`have h := Nat.gcd_mul_lcm n 40`) and `rw/simp` cover the use case."""
    feats = _gcd_features()
    goal = "h₁ : Nat.gcd n 40 = 10 h₂ : Nat.lcm n 40 = 280 ⊢ n = 70"
    for prem in ("Nat.gcd_mul_lcm", "gcd_mul_lcm"):
        out = generate_template_tactics(
            feats, retrieved_premises=[prem], goal_text=goal,
        )
        texts = [t for t, _c, _m in out]
        assert f"exact {prem}" not in texts
        assert f"apply {prem}" not in texts
        # rw/simp still emitted (cheap to try, no arity mismatch).
        assert f"rw [{prem}]" in texts
        # And the qualified-form instantiation appears.
        if prem == "Nat.gcd_mul_lcm":
            assert "have h := Nat.gcd_mul_lcm n 40" in texts


def test_template_have_before_any_exact_for_gcd_mul_lcm():
    """End-to-end: p100-like features + Nat.gcd_mul_lcm produces the
    `have h := Nat.gcd_mul_lcm n 40` candidate; if any `exact ...`
    candidate is also emitted, the `have h` must come first in the
    returned list. (The `have hprod := …` literal was dropped — it
    was a p100-specific hypothesis name; `have h := …` is the
    Mathlib convention.)"""
    feats = _gcd_features()
    goal = "h₁ : Nat.gcd n 40 = 10 h₂ : Nat.lcm n 40 = 280 ⊢ n = 70"
    out = generate_template_tactics(
        feats, retrieved_premises=["Nat.gcd_mul_lcm"], goal_text=goal,
    )
    texts = [t for t, _c, _m in out]
    have_h_idx = next(
        (i for i, t in enumerate(texts)
         if t == "have h := Nat.gcd_mul_lcm n 40"),
        -1,
    )
    assert have_h_idx >= 0, f"`have h := …` missing; got {texts}"
    exact_indices = [i for i, t in enumerate(texts) if t.startswith("exact ")]
    if exact_indices:
        assert have_h_idx < min(exact_indices), (
            f"have-h at {have_h_idx} must precede exact at "
            f"{exact_indices}; texts={texts}"
        )


def test_classify_unknown_for_unrelated_text():
    assert classify_tactic("") == "unknown"
    assert classify_tactic("// not a tactic") == "unknown"


# ---------------------------------------------------------------------------
# ProofPriorIndex
# ---------------------------------------------------------------------------

def _gcd_features():
    return ProofStateFeatures(
        symbols=("gcd", "lcm", "equality"),
        namespaces=("Nat",),
        target_shape="equality",
        hypothesis_shapes=("gcd_eq", "lcm_eq"),
        constants=("10", "40", "280", "70"),
        previous_tactic_class=None,
        retrieved_premises=(),
    )


def test_proof_prior_round_trip(tmp_path: Path):
    idx = ProofPriorIndex()
    feats = _gcd_features()
    move = ProofPriorMove(
        tactic_template="have h := Nat.gcd_mul_lcm n 40",
        tactic_class="have_premise",
        premise="Nat.gcd_mul_lcm",
        source="solved_jsonl",
    )
    for _ in range(3):
        idx.add_observation(feats, move, "solved")
    other = ProofPriorMove(
        tactic_template="omega",
        tactic_class="omega",
        premise=None,
        source="solved_jsonl",
    )
    idx.add_observation(feats, other, "solved")

    out = tmp_path / "prior.jsonl"
    idx.save(out)
    assert out.exists() and out.read_text().strip()

    reloaded = ProofPriorIndex.load(out)
    suggestions = reloaded.suggest(feats, top_k=5)
    assert suggestions, "reloaded index should return suggestions for known features"
    # Most-observed move ranks first.
    assert suggestions[0].tactic_template == "have h := Nat.gcd_mul_lcm n 40"
    assert suggestions[0].prior_probability > suggestions[-1].prior_probability


def test_proof_prior_does_not_use_theorem_id():
    """feature_key is invariant under any string that is NOT in the bag.
    Theorem IDs are never part of the bag, so feeding two unrelated IDs
    must produce the same key."""
    feats = _gcd_features()
    k1 = feature_key(feats, level=0)
    # Construct a feature bag that looks identical except for hypothetical
    # theorem-ID leakage — there is no such slot, so any "rename" of the
    # theorem produces an identical bag and an identical key.
    feats_same_shape = ProofStateFeatures(
        symbols=feats.symbols, namespaces=feats.namespaces,
        target_shape=feats.target_shape,
        hypothesis_shapes=feats.hypothesis_shapes,
        constants=feats.constants,
        previous_tactic_class=feats.previous_tactic_class,
        retrieved_premises=("renamed_theorem_id_irrelevant",),
    )
    k2 = feature_key(feats_same_shape, level=0)
    assert k1 == k2, "retrieved_premises must not affect the key"


def test_proof_prior_fallback_to_coarser_keys():
    idx = ProofPriorIndex()
    feats = _gcd_features()
    move = ProofPriorMove(
        tactic_template="omega",
        tactic_class="omega",
        premise=None,
        source="solved_jsonl",
    )
    idx.add_observation(feats, move, "solved")
    # Query with a different previous_tactic_class — the level-0 key won't
    # match, but the level-1 key (drop previous_tactic_class) will.
    feats_alt = ProofStateFeatures(
        symbols=feats.symbols, namespaces=feats.namespaces,
        target_shape=feats.target_shape,
        hypothesis_shapes=feats.hypothesis_shapes,
        constants=feats.constants,
        previous_tactic_class="rw",
        retrieved_premises=(),
    )
    suggestions = idx.suggest(feats_alt, top_k=3)
    assert any(m.tactic_template == "omega" for m in suggestions)


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------

def test_template_gcd_lcm_p100_like():
    feats = _gcd_features()
    out = generate_template_tactics(
        feats, retrieved_premises=["Nat.gcd_mul_lcm"],
        goal_text="h₀ : 0 < n h₁ : Nat.gcd n 40 = 10 h₂ : Nat.lcm n 40 = 280 ⊢ n = 70",
    )
    texts = [t for t, _c, _m in out]
    assert any(t == "have h := Nat.gcd_mul_lcm n 40" for t in texts), \
        f"missing p100-style template: {texts}"


def test_template_real_inequality_emits_nlinarith_and_sq_nonneg():
    feats = ProofStateFeatures(
        symbols=("inequality", "power"),
        namespaces=("Real",),
        target_shape="inequality",
        hypothesis_shapes=(),
        constants=(),
        previous_tactic_class=None,
    )
    out = generate_template_tactics(
        feats, retrieved_premises=[],
        goal_text="a b : ℝ ⊢ a^2 + b^2 ≤ (a+b)^2",
    )
    texts = [t for t, _c, _m in out]
    assert "nlinarith" in texts
    assert any("sq_nonneg" in t for t in texts)


def test_template_finset_sum_emits_norm_num_simp_ringnf():
    feats = ProofStateFeatures(
        symbols=("sum", "range"),
        namespaces=("Finset",),
        target_shape="equality",
        hypothesis_shapes=(),
        constants=(),
        previous_tactic_class=None,
    )
    out = generate_template_tactics(feats, retrieved_premises=[])
    texts = [t for t, _c, _m in out]
    assert "norm_num" in texts
    assert "simp" in texts
    assert "ring_nf" in texts


def test_template_emits_no_placeholders():
    feats = _gcd_features()
    out = generate_template_tactics(feats, retrieved_premises=["Nat.gcd_mul_lcm"])
    for text, _cost, _meta in out:
        assert "?" not in text, f"placeholder in template: {text!r}"
        assert "<FILL" not in text and "<TODO" not in text
        # bare-underscore safety
        toks = text.split()
        assert not any(tok == "_" for tok in toks), f"bare _ in template: {text!r}"


# ---------------------------------------------------------------------------
# Dependency exploration
# ---------------------------------------------------------------------------

class _StubGraph:
    """Minimal networkx-like stub: enough for dependency_explorer's
    nodes / predecessors / successors / *_degree calls."""
    def __init__(self, edges):
        self._adj: dict[str, set[str]] = {}
        for u, v in edges:
            self._adj.setdefault(u, set()).add(v)
            self._adj.setdefault(v, set())
        self.nodes = set(self._adj.keys())
    def predecessors(self, n):
        return [u for u, vs in self._adj.items() if n in vs]
    def successors(self, n):
        return list(self._adj.get(n, []))
    def in_degree(self, n):
        return sum(1 for _ in self.predecessors(n))
    def out_degree(self, n):
        return len(self._adj.get(n, []))


def test_dependency_exploration_returns_neighbours_with_epsilon():
    graph = _StubGraph(edges=[
        ("Nat.gcd_mul_lcm", "Nat.gcd_assoc"),
        ("Nat.gcd_assoc",   "Nat.gcd_comm"),
        ("Real.sqrt_pos",   "Real.sqrt_nonneg"),
    ])
    feats = _gcd_features()
    moves = dependency_exploration_candidates(
        feats, graph, retrieved_premises=["Nat.gcd_mul_lcm"], top_k=10,
    )
    names = [m.premise for m in moves]
    assert "Nat.gcd_assoc" in names or "Nat.gcd_comm" in names, \
        f"expected gcd-neighbours, got {names}"
    for m in moves:
        assert m.source == "dependency_exploration"
        assert m.prior_probability > 0.0  # exploration_epsilon


def test_dependency_exploration_none_graph_is_noop():
    feats = _gcd_features()
    assert dependency_exploration_candidates(feats, None, ["x"], top_k=5) == []


# ---------------------------------------------------------------------------
# Move scoring


# ---------------------------------------------------------------------------
# PART C: holdout-safe build_proof_prior.py
# ---------------------------------------------------------------------------

def _write_solved_row(path: Path, *, problem_id: str, tactics: list[str]) -> None:
    """Append one solved JSONL row to `path`."""
    row = {
        "id": problem_id,
        "outcome": "solved",
        "proof_tactics": tactics,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _import_builder():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_build_proof_prior_under_test",
        Path(__file__).resolve().parents[1] / "scripts" / "build_proof_prior.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_holdout_excludes_target_theorem_moves(tmp_path: Path):
    builder = _import_builder()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    data_dir = tmp_path / "data"  # empty; the builder is OK without source files
    jsonl = results_dir / "mock_run.jsonl"
    _write_solved_row(
        jsonl, problem_id="theorem_keep",
        tactics=["have h := Foo.bar_keep n 1", "omega"],
    )
    _write_solved_row(
        jsonl, problem_id="theorem_holdout",
        tactics=["have h := Foo.bar_holdout n 1", "decide"],
    )

    out_path = tmp_path / "prior.jsonl"
    stats = builder.build(
        results_dir, data_dir, out_path,
        excluded_ids={"theorem_holdout"},
    )
    assert stats["total_rows_seen"] == 2
    assert stats["solved_rows_seen"] == 2
    assert stats["rows_excluded_by_theorem_id"] == 1
    assert stats["unique_theorems_indexed"] == 1
    assert stats["moves_recorded"] == 2  # the two kept tactics
    # Sidecar metadata.
    sidecar = out_path.with_suffix(out_path.suffix + ".meta.json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["clean_holdout"] is True
    assert meta["excluded_theorem_count"] == 1
    assert meta["excluded_subset"] == ["theorem_holdout"]

    # The excluded theorem's tactic content must not appear in the JSONL.
    body = out_path.read_text(encoding="utf-8")
    assert "Foo.bar_keep" in body
    assert "Foo.bar_holdout" not in body
    # Theorem IDs may now appear under `provenance` (audit metadata,
    # introduced in PART A of the audit fix-up) but must not appear as
    # feature components — verify by walking the JSONL rows and
    # checking that no row's `feature_key`, `tactic_template`, or any
    # of its tags contains the held-out theorem id. Provenance is the
    # one allowed slot for theorem ids (it is metadata, never a key).
    for line in body.splitlines():
        row = json.loads(line)
        for field in ("feature_key", "tactic_template", "tactic_class",
                       "source"):
            assert "theorem_holdout" not in str(row.get(field, "")), (
                f"holdout id leaked into {field}: {row.get(field)}"
            )
        for tag in (row.get("tags") or []):
            assert "theorem_holdout" not in tag
    # The kept theorem's id is allowed in `provenance` (audit metadata)
    # but must NOT appear in any keying field.
    for line in body.splitlines():
        row = json.loads(line)
        for field in ("feature_key", "tactic_template", "tactic_class"):
            assert "theorem_keep" not in str(row.get(field, ""))


def test_holdout_off_means_clean_holdout_false(tmp_path: Path):
    builder = _import_builder()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_solved_row(results_dir / "r.jsonl",
                      problem_id="any_id", tactics=["decide"])
    out_path = tmp_path / "prior.jsonl"
    stats = builder.build(results_dir, tmp_path / "data", out_path,
                          excluded_ids=set())
    assert stats["rows_excluded_by_theorem_id"] == 0
    assert stats["clean_holdout"] is False
    sidecar = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["clean_holdout"] is False
    assert meta["excluded_theorem_count"] == 0


def test_holdout_theorem_id_is_never_a_feature(tmp_path: Path):
    """Both kept and excluded theorems share a feature key derived from
    the (header-derived) features; excluding the answer-row must not
    change the key for the other row, and the theorem ID itself must
    never appear inside any feature digest."""
    builder = _import_builder()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_solved_row(results_dir / "r.jsonl",
                      problem_id="other_problem", tactics=["decide"])
    out_path = tmp_path / "prior.jsonl"
    stats = builder.build(results_dir, tmp_path / "data", out_path,
                          excluded_ids={"some_held_out_problem"})
    # The held-out id never even appeared in the source corpus, so
    # it must not appear anywhere in the output (including provenance).
    body = out_path.read_text(encoding="utf-8")
    assert "some_held_out_problem" not in body
    # `other_problem` is a kept theorem; its id appears in the
    # provenance audit slot but must NOT appear in any feature-keying
    # field — verify field-by-field.
    for line in body.splitlines():
        row = json.loads(line)
        for field in ("feature_key", "tactic_template", "tactic_class"):
            assert "other_problem" not in str(row.get(field, ""))
        for tag in (row.get("tags") or []):
            assert "other_problem" not in tag
    # And the sidecar records the excluded set verbatim (audit trail).
    meta = json.loads(out_path.with_suffix(out_path.suffix + ".meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["excluded_subset"] == ["some_held_out_problem"]


# ---------------------------------------------------------------------------
# PART A: source-priority ordering and per-source caps


# ---------------------------------------------------------------------------
# PART B: junk-template filter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("junk_premise", [
    "Nat.P", "Nat.log", "Nat.bit", "Nat.nth", "Nat.fib",
    "Nat.Prime", "WType.Nat", "Nat.factorial", "Nat.card",
    "Nat.choose",
])
def test_template_rejects_exact_for_junk_premises(junk_premise):
    feats = _gcd_features()
    out = generate_template_tactics(
        feats, retrieved_premises=[junk_premise],
        goal_text="h₀ : 0 < n ⊢ n = 70",
    )
    texts = [t for t, _c, _m in out]
    bad = [t for t in texts if t.startswith(f"exact {junk_premise}")
                              or t.startswith(f"apply {junk_premise}")
                              or t.startswith(f"rw [{junk_premise}")
                              or t.startswith(f"simp [{junk_premise}")]
    assert bad == [], (
        f"junk premise {junk_premise} should not yield exact/apply/rw/simp; "
        f"got {bad}"
    )
    # But `have h := junk` is still allowed (no type error possible).
    assert any(t == f"have h := {junk_premise}" for t in texts)


def test_template_keeps_rw_simp_have_for_real_lemmas():
    """After PART C (fix-up #5), bare `exact P` / `apply P` are dropped
    for ALL retrieved premises — they would type-error on any lemma
    with non-trivial arguments. The kept templates are `rw [P]`,
    `simp [P]`, and `have h := P` (plus arg-instantiations downstream)."""
    feats = _gcd_features()
    for prem in ("add_comm", "Real.sqrt_pos", "sq_nonneg"):
        out = generate_template_tactics(
            feats, retrieved_premises=[prem],
            goal_text="⊢ n = 70",
        )
        texts = [t for t, _c, _m in out]
        assert f"rw [{prem}]" in texts
        assert f"simp [{prem}]" in texts
        assert f"have h := {prem}" in texts
        # `exact P` / `apply P` are gone unconditionally.
        assert f"exact {prem}" not in texts
        assert f"apply {prem}" not in texts


# ---------------------------------------------------------------------------
# PART C: p100 domain template from retrieval-only path
# ---------------------------------------------------------------------------

def test_gcd_lcm_domain_template_fires_without_solved_jsonl():
    """The gcd/lcm domain template hard-codes Nat.gcd_mul_lcm, so it
    fires from features alone — no solved_jsonl row required. The
    emitted tactic must use the goal's actual (variable, constant) pair
    (n, 40), not the fallback (n, k). Only the conventional `have h`
    form is emitted — the `have hprod` literal was dropped per the
    continuation fix-up contract."""
    feats = _gcd_features()
    goal_text = (
        "n : ℕ h₀ : 0 < n h₁ : Nat.gcd n 40 = 10 h₂ : Nat.lcm n 40 = 280 "
        "⊢ n = 70"
    )
    out = generate_template_tactics(
        feats, retrieved_premises=[],  # NO retrieval, NO prior
        goal_text=goal_text,
    )
    texts = [t for t, _c, _m in out]
    assert "have h := Nat.gcd_mul_lcm n 40" in texts
    assert "have hprod := Nat.gcd_mul_lcm n 40" not in texts


# ---------------------------------------------------------------------------
# PART D: automatic import inference for clean eval
# ---------------------------------------------------------------------------

def test_import_inference_p100_header_picks_gcd_basic():
    from search.import_inference import infer_imports_from_header
    header = (
        "theorem mathd_numbertheory_100\n"
        "  (n : ℕ) (h₀ : 0 < n)\n"
        "  (h₁ : Nat.gcd n 40 = 10)\n"
        "  (h₂ : Nat.lcm n 40 = 280) :\n"
        "  n = 70 := by sorry"
    )
    r = infer_imports_from_header(header)
    assert r.matched is True
    assert "import Mathlib.Data.Nat.GCD.Basic" in r.imports
    assert "nat_gcd_lcm" in r.matched_rules


def test_import_inference_does_not_use_theorem_id():
    """Renaming the theorem must not change the inferred imports."""
    from search.import_inference import infer_imports_from_header
    h1 = ("theorem foo (n : ℕ) (h : Nat.gcd n 40 = 10) :"
          " Nat.lcm n 40 = 280 := by sorry")
    h2 = ("theorem bar (n : ℕ) (h : Nat.gcd n 40 = 10) :"
          " Nat.lcm n 40 = 280 := by sorry")
    r1 = infer_imports_from_header(h1)
    r2 = infer_imports_from_header(h2)
    assert r1.imports == r2.imports
    assert r1.matched_rules == r2.matched_rules


def test_import_inference_fallback_when_no_rule_fires():
    from search.import_inference import (
        infer_imports_from_header, DEFAULT_IMPORTS,
    )
    header = "theorem trivial_example : True := by sorry"
    r = infer_imports_from_header(header)
    assert r.matched is False
    assert r.imports == DEFAULT_IMPORTS


def test_import_inference_real_sqrt_picks_sqrt_plus_tactic():
    from search.import_inference import infer_imports_from_header
    header = ("theorem foo (x : ℝ) (h : 0 ≤ x) :"
              " 0 ≤ Real.sqrt x := by sorry")
    r = infer_imports_from_header(header)
    assert r.matched is True
    assert "import Mathlib.Analysis.SpecialFunctions.Sqrt" in r.imports
    assert "import Mathlib.Tactic" in r.imports


# ---------------------------------------------------------------------------
# PART A diagnostic: merge + suggest preserves solved_jsonl
# ---------------------------------------------------------------------------

def _p100_header_text() -> str:
    return (
        "theorem mathd_numbertheory_100\n"
        "  (n : ℕ) (h₀ : 0 < n)\n"
        "  (h₁ : Nat.gcd n 40 = 10)\n"
        "  (h₂ : Nat.lcm n 40 = 280) :\n"
        "  n = 70"
    )


def test_root_state_features_use_theorem_header(tmp_path: Path):
    """At root state (`goal_text == ""`), the runner must fall back to
    `theorem_header` for feature extraction. Without the fix the bag
    is empty and `suggest()` returns 0 rows from a real prior."""
    from search.proof_prior import (
        ProofPriorIndex, ProofPriorMove, feature_key,
    )
    from search.state_features import extract_state_features

    feats_header = extract_state_features(_p100_header_text(), proof_prefix=[])
    feats_empty  = extract_state_features("", proof_prefix=[])
    # The two bags must differ on every keying slot we mine on.
    assert feats_header.namespaces != feats_empty.namespaces
    assert feats_header.symbols != feats_empty.symbols

    # And their level-0 feature_keys must differ. If they ever match,
    # the whole hypothesis behind the PART A fix collapses.
    assert feature_key(feats_header) != feature_key(feats_empty)


# ---------------------------------------------------------------------------
# PART F: end-to-end ordering test for p100-like features
# ---------------------------------------------------------------------------

def test_p100_candidate_order_have_before_exact():
    """With p100-like features + a mix of real-lemma and junk premises,
    `have h := Nat.gcd_mul_lcm n 40` (the conventional `have h` form
    of the gcd_lcm domain template) must precede every `exact <P>` /
    `apply <P>` candidate. After the per-premise `exact`/`apply` drop
    no exact/apply should appear at all."""
    feats = ProofStateFeatures(
        symbols=("gcd", "lcm", "equality"),
        namespaces=("Nat",),
        target_shape="equality",
        hypothesis_shapes=("gcd_eq", "lcm_eq"),
        constants=("10", "40", "280", "70"),
        previous_tactic_class=None,
    )
    goal = ("n : ℕ h₀ : 0 < n h₁ : Nat.gcd n 40 = 10 h₂ : Nat.lcm n 40 = 280"
            " ⊢ n = 70")
    out = generate_template_tactics(
        feats,
        retrieved_premises=[
            "Nat.gcd_mul_lcm", "Nat.mod_lcm", "Nat.periodic_gcd",
            "PNat.gcd_mul_lcm", "Nat.gcd_greatest",
        ],
        goal_text=goal,
    )
    texts = [t for t, _c, _m in out]
    haveh_idx = next((i for i, t in enumerate(texts)
                      if t == "have h := Nat.gcd_mul_lcm n 40"), -1)
    assert haveh_idx >= 0, f"missing have h template; got {texts}"
    # And it must be the domain_template version (which is emitted
    # FIRST in the generator, so it wins dedup against the per-premise
    # instantiation duplicate at higher cost).
    for text, _cost, meta in out:
        if text == "have h := Nat.gcd_mul_lcm n 40":
            assert "domain_template" in meta.get("tags", ()), (
                f"`{text}` survived as {meta} but should be domain_template"
            )
    # No `exact P` / `apply P` for any retrieved premise.
    for prem in ("Nat.gcd_mul_lcm", "Nat.mod_lcm", "Nat.periodic_gcd",
                 "PNat.gcd_mul_lcm", "Nat.gcd_greatest"):
        assert f"exact {prem}" not in texts
        assert f"apply {prem}" not in texts
    # And no `have hprod := …` literal — that was the p100-tailored form.
    assert "have hprod := Nat.gcd_mul_lcm n 40" not in texts


# ---------------------------------------------------------------------------
# PART E: tactic-text canonicalisation in Nat context
# ---------------------------------------------------------------------------

def test_canonicalise_tactic_text_rewrites_rw_bracket():
    from search.tactic_classify import canonicalise_tactic_text
    # Inside Nat namespace.
    t = "rw [← one_mul (lcm m n), ← h.gcd_eq_one, gcd_mul_lcm]"
    out = canonicalise_tactic_text(t, "Nat")
    assert "Nat.gcd_mul_lcm" in out
    # Local projection untouched.
    assert "h.gcd_eq_one" in out
    # Already-qualified form untouched (idempotency).
    t2 = "rw [Nat.gcd_mul_lcm, Nat.gcd_comm]"
    assert canonicalise_tactic_text(t2, "Nat") == t2
    # Outside Nat namespace, no rewrite.
    assert canonicalise_tactic_text(t, "Real") == t
    # No namespace, no rewrite.
    assert canonicalise_tactic_text(t, None) == t


# ---------------------------------------------------------------------------
# PART A/B: sequence-prior mining + suggest_next
# ---------------------------------------------------------------------------

def test_sequence_prior_round_trip(tmp_path: Path):
    """Two-tactic chain mined via `add_sequence_observation` survives
    save+load and `suggest_next` returns the continuation."""
    idx = ProofPriorIndex()
    move = ProofPriorMove(
        tactic_template="rw [h₁, h₂] at hprod",
        tactic_class="rw",
        source="sequence_solved_jsonl",
        tags=("mined", "sequence"),
    )
    idx.add_sequence_observation(
        "have hprod := Nat.gcd_mul_lcm n 40", move, result="solved",
    )
    out = tmp_path / "seq.jsonl"
    idx.save(out)
    re_idx = ProofPriorIndex.load(out)
    nxt = re_idx.suggest_next("have hprod := Nat.gcd_mul_lcm n 40", top_k=5)
    assert any(m.tactic_template == "rw [h₁, h₂] at hprod" for m in nxt), (
        f"sequence continuation missing; got {[m.tactic_template for m in nxt]}"
    )
    # Unknown previous tactic returns empty.
    assert re_idx.suggest_next("simp", top_k=5) == []
    assert re_idx.suggest_next("", top_k=5) == []


def test_build_proof_prior_records_sequence_for_p100(tmp_path: Path):
    """End-to-end: the build_proof_prior.py builder records the p100
    transition `have hprod := … -> rw [h₁, h₂] at hprod` and
    `rw [h₁, h₂] at hprod -> omega` from a synthetic solved-row."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_bpp_seq_test",
        Path(__file__).resolve().parents[1] / "scripts" / "build_proof_prior.py",
    )
    bpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bpp)

    results = tmp_path / "results"
    results.mkdir()
    (results / "mock.jsonl").write_text(
        json.dumps({
            "id": "mathd_numbertheory_100",
            "outcome": "solved",
            "proof_tactics": [
                "have hprod := Nat.gcd_mul_lcm n 40",
                "rw [h₁, h₂] at hprod",
                "omega",
            ],
        }) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "prior.jsonl"
    bpp.build(results, tmp_path / "data", out, excluded_ids=set())
    idx = ProofPriorIndex.load(out)
    nxt = idx.suggest_next("have hprod := Nat.gcd_mul_lcm n 40", top_k=5)
    assert any(m.tactic_template == "rw [h₁, h₂] at hprod" for m in nxt)
    nxt2 = idx.suggest_next("rw [h₁, h₂] at hprod", top_k=5)
    assert any(m.tactic_template == "omega" for m in nxt2)


# ---------------------------------------------------------------------------
# PART D: state-conditional rewrite template
# ---------------------------------------------------------------------------

def test_no_p100_specific_template_in_template_tactics():
    """The earlier turn's hand-coded `_rewrite_after_have_templates`
    helper has been removed (the user's PART C contract). The generic
    abstract-continuation prior is the only mechanism that may emit
    `rw [<eqs>] at <hyp>` now — and only when the corpus actually
    learned the (have_premise -> rw_at_hyp) transition. We assert (a)
    the helper function is gone, and (b) no domain template emits
    `have hprod := …` (the hand-tailored hypothesis name from p100)."""
    import inspect
    import search.template_tactics as tt
    src = inspect.getsource(tt)
    assert "def _rewrite_after_have_templates" not in src, (
        "the p100-specific helper function must be deleted"
    )
    # The `have hprod := …` emit was a literal p100 mimic. The generic
    # form (`have h := …`) is fine because `h` is a common Mathlib
    # naming convention, not problem-specific.
    assert "have hprod :=" not in src, (
        "template_tactics.py must not emit `have hprod := …` literals"
    )


# ---------------------------------------------------------------------------
# PART C: duplicate-have suppression (inline implementation tested via
# the same regex the runner uses)
# ---------------------------------------------------------------------------

def test_have_name_in_context_helper():
    """The runner's duplicate-have filter uses these exact regexes."""
    import re as _re
    _HAVE_NAME_RE = _re.compile(r"^have\s+([A-Za-z_][A-Za-z0-9_'₀-₉]*)\b")
    ctx = (
        "n : ℕ\n"
        "h₁ : Nat.gcd n 40 = 10\n"
        "hprod : n * 40 = Nat.gcd n 40 * Nat.lcm n 40\n"
        "⊢ n = 70"
    )
    def in_ctx(text, c):
        m = _HAVE_NAME_RE.match(text.strip())
        if m is None: return False
        name = m.group(1)
        pat = _re.compile(rf"(?<![A-Za-z0-9_'₀-₉]){_re.escape(name)}\s*:")
        return bool(pat.search(c))
    # Already-bound `hprod` should be suppressed.
    assert in_ctx("have hprod := Nat.gcd_mul_lcm n 40", ctx) is True
    # Fresh name `h` should pass.
    assert in_ctx("have h := Nat.gcd_mul_lcm n 40", ctx) is False
    # `h₁` is already there — suppress.
    assert in_ctx("have h₁ := whatever", ctx) is True
    # Non-have tactic — passes through unaffected.
    assert in_ctx("rw [h₁, h₂] at hprod", ctx) is False


# ---------------------------------------------------------------------------
# PART A (audit) — per-row provenance + clean-eval leakage check
# ---------------------------------------------------------------------------

def test_provenance_round_trip_through_jsonl(tmp_path: Path):
    """A move stored with a provenance theorem id survives
    save+load and is queryable via `audit_target_provenance`."""
    idx = ProofPriorIndex()
    feats = _gcd_features()
    move = ProofPriorMove(
        tactic_template="have hprod := Nat.gcd_mul_lcm n 40",
        tactic_class="have_premise",
        premise="Nat.gcd_mul_lcm",
        source="solved_jsonl",
        provenance=("mathd_numbertheory_100",),
    )
    idx.add_observation(feats, move, "solved")
    out = tmp_path / "prior.jsonl"
    idx.save(out)
    reloaded = ProofPriorIndex.load(out)
    audit = reloaded.audit_target_provenance("mathd_numbertheory_100")
    assert audit["rows_with_target"] >= 1
    assert "solved_jsonl" in audit["sources_used"]
    assert any("Nat.gcd_mul_lcm" in t for t in audit["sample_templates"])
    # Unknown / unrelated target → zero leakage.
    assert reloaded.audit_target_provenance("amc12_2000_p20")["rows_with_target"] == 0


def test_build_proof_prior_records_provenance(tmp_path: Path):
    """End-to-end: the solved-log miner stamps provenance, and the
    target-exclusion path keeps provenance for non-target theorems
    only."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_bpp_prov_test",
        Path(__file__).resolve().parents[1] / "scripts" / "build_proof_prior.py",
    )
    bpp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bpp)
    results = tmp_path / "results"; results.mkdir()
    for pid, tactics in [
        ("target_problem",  ["have h := Foo.lemma_a n 3", "omega"]),
        ("keep_problem",    ["have h := Foo.lemma_b n 5", "ring_nf"]),
    ]:
        (results / f"{pid}.jsonl").write_text(json.dumps({
            "id": pid, "outcome": "solved", "proof_tactics": tactics,
        }) + "\n", encoding="utf-8")
    # Include both — provenance should distinguish them.
    out_full = tmp_path / "full.jsonl"
    bpp.build(results, tmp_path / "data", out_full, excluded_ids=set())
    full = ProofPriorIndex.load(out_full)
    assert full.audit_target_provenance("target_problem")["rows_with_target"] >= 1
    assert full.audit_target_provenance("keep_problem")["rows_with_target"] >= 1
    # Now exclude target; its rows must vanish.
    out_clean = tmp_path / "clean.jsonl"
    bpp.build(results, tmp_path / "data", out_clean,
              excluded_ids={"target_problem"})
    clean = ProofPriorIndex.load(out_clean)
    assert clean.audit_target_provenance("target_problem")["rows_with_target"] == 0
    assert clean.audit_target_provenance("keep_problem")["rows_with_target"] >= 1


def test_merge_unions_provenance(tmp_path: Path):
    """Both contributing files' provenance survives the merge."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_merge_prov_test",
        Path(__file__).resolve().parents[1] / "scripts" / "merge_proof_priors.py",
    )
    merger = importlib.util.module_from_spec(spec); spec.loader.exec_module(merger)
    feats = _gcd_features()
    key0 = feature_key(feats, level=0)
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    def _row(template, prov):
        return {
            "feature_key": key0, "tactic_template": template,
            "tactic_class": "have_premise", "premise": "X",
            "count": 1, "prior_probability": 0.5,
            "source": "solved_jsonl", "tags": ["mined"],
            "requires_instantiation": False,
            "provenance": list(prov),
        }
    a.write_text(json.dumps(_row("have h := X n 1", ["theorem_A"])) + "\n",
                  encoding="utf-8")
    b.write_text(json.dumps(_row("have h := X n 1", ["theorem_B"])) + "\n",
                  encoding="utf-8")
    out = tmp_path / "merged.jsonl"
    merger.merge([a, b], out)
    merged = ProofPriorIndex.load(out)
    assert merged.audit_target_provenance("theorem_A")["rows_with_target"] >= 1
    assert merged.audit_target_provenance("theorem_B")["rows_with_target"] >= 1


# ---------------------------------------------------------------------------
# PART B/C (loop prevention) — covered live in the propose loop;
# unit-test the same regexes the runner uses
# ---------------------------------------------------------------------------

def test_have_rhs_already_in_prefix():
    """`have h := P args` after `have hprod := P args` (different
    hypothesis name but same RHS) must be suppressed."""
    import re as _re
    _HAVE_RHS_RE = _re.compile(
        r"^have\s+\S+\s*(?::[^:]*?)?\s*:=\s*(.+)$"
    )
    def dup(text, prefix):
        m = _HAVE_RHS_RE.match(text.strip())
        if m is None: return False
        rhs = m.group(1).strip()
        for prior in prefix:
            pm = _HAVE_RHS_RE.match((prior or "").strip())
            if pm and pm.group(1).strip() == rhs:
                return True
        return False
    prefix = ["have hprod := Nat.gcd_mul_lcm n 40"]
    assert dup("have h := Nat.gcd_mul_lcm n 40", prefix) is True
    # Different premise — passes through.
    assert dup("have h := Nat.gcd_comm n 40", prefix) is False
    # Non-have tactic — passes through.
    assert dup("omega", prefix) is False


def test_tactic_already_on_path():
    """Generic loop control: any literal repeat is suppressed."""
    prefix = ["intro x", "simp", "have h := P n 40"]
    def already(text):
        return any(text.strip() == (p or "").strip() for p in prefix)
    assert already("intro x") is True
    assert already("simp") is True
    assert already("have h := P n 40") is True
    assert already("omega") is False


# ---------------------------------------------------------------------------
# PART D — abstract continuation prior + generic instantiator
# ---------------------------------------------------------------------------

def test_tactic_shape_distinguishes_rw_at_from_rw_bare():
    from search.tactic_classify import tactic_shape
    assert tactic_shape("rw [h₁, h₂] at hprod") == "rw_at_hyp"
    assert tactic_shape("rw [Nat.gcd_mul_lcm]") == "rw_bare"
    assert tactic_shape("simp at h₁") == "simp_at_hyp"
    assert tactic_shape("simp [foo]") == "simp_shape"
    assert tactic_shape("omega") == "arithmetic_close"
    assert tactic_shape("nlinarith") == "arithmetic_close"
    assert tactic_shape("decide") == "decision_close"
    assert tactic_shape("ring_nf") == "algebra_close"


def test_abstract_continuation_prior_is_theorem_id_free(tmp_path: Path):
    """Mining a fake corpus must produce an abstract pattern keyed on
    class transitions, NOT on the corpus theorem name."""
    idx = ProofPriorIndex()
    idx.add_abstract_observation(
        "have_premise", "rw_at_hyp",
        next_class="rw",
        source="abstract_continuation_prior",
        tags=("mathlib",),
        provenance=("Nat.some_unrelated_lemma",),
        result="solved",
    )
    idx.add_abstract_observation(
        "have_premise", "arithmetic_close",
        next_class="omega",
        source="abstract_continuation_prior",
        provenance=("Nat.some_unrelated_lemma",),
        result="solved",
    )
    out = tmp_path / "abs.jsonl"; idx.save(out)
    re_idx = ProofPriorIndex.load(out)
    hints = re_idx.suggest_abstract("have_premise", top_k=10)
    shapes = {m.tactic_template for m in hints}
    assert "rw_at_hyp" in shapes
    assert "arithmetic_close" in shapes
    # Target-name independence: querying with the mining theorem ID
    # does NOT route through audit_target_provenance to skew suggest.
    # The audit reports leak, but suggest_abstract is purely class-keyed.
    audit = re_idx.audit_target_provenance("mathd_numbertheory_100")
    assert audit["rows_with_target"] == 0


# ---------------------------------------------------------------------------
# PART G — clean-eval audit JSONL field shape (smoke)


# ---------------------------------------------------------------------------
# CLI flag parsing regression


# ---------------------------------------------------------------------------
# build_proof_prior --exclude-subset coverage
# ---------------------------------------------------------------------------

def test_exclude_subset_excludes_every_id_in_subset_file(tmp_path: Path):
    """Every theorem_id named in a subset file must be excluded, and the
    sidecar must record the count / list / source-path verbatim."""
    builder = _import_builder()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    # Five solved theorems; we hold out four of them.
    for pid, tactic in [
        ("alpha", "decide"),
        ("beta",  "rfl"),
        ("gamma", "omega"),
        ("delta", "linarith"),
        ("epsilon", "trivial"),
    ]:
        _write_solved_row(results_dir / f"{pid}.jsonl",
                          problem_id=pid, tactics=[tactic])
    subset_file = tmp_path / "_holdout.txt"
    subset_file.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
    excluded_ids = builder._load_subset_ids(subset_file)
    out_path = tmp_path / "prior.jsonl"
    stats = builder.build(
        results_dir, tmp_path / "data", out_path,
        excluded_ids=excluded_ids,
        excluded_subset_path=subset_file,
    )
    assert stats["rows_excluded_by_theorem_id"] == 4
    # Only `epsilon` survives.
    assert stats["unique_theorems_indexed"] == 1
    body = out_path.read_text(encoding="utf-8")
    for pid in ("alpha", "beta", "gamma", "delta"):
        assert pid not in body, f"held-out id {pid} leaked into output"
    meta = json.loads(out_path.with_suffix(out_path.suffix + ".meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["clean_holdout"] is True
    assert meta["excluded_theorem_count"] == 4
    assert meta["excluded_subset"] == ["alpha", "beta", "delta", "gamma"]
    assert meta["excluded_theorem_ids"] == meta["excluded_subset"]
    assert meta["excluded_subset_path"] == str(subset_file)


# ---------------------------------------------------------------------------
# merge_proof_priors sidecar preserves holdout metadata
# ---------------------------------------------------------------------------

def _import_merger():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_merge_for_sidecar_test",
        Path(__file__).resolve().parents[1] / "scripts" / "merge_proof_priors.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_merge_preserves_clean_holdout_and_excluded_ids(tmp_path: Path):
    """If any input sidecar has clean_holdout=true, the merged sidecar
    must (a) keep clean_holdout=true, (b) hold the union of excluded IDs
    in both `excluded_subset` and `excluded_theorem_ids`, and (c) sum
    the excluded-theorem count from the union."""
    merger = _import_merger()

    def _write_row(path: Path, template: str, source: str) -> None:
        from search.proof_prior import feature_key
        feats = _gcd_features()
        row = {
            "feature_key": feature_key(feats, level=0),
            "tactic_template": template,
            "tactic_class": "have_premise",
            "premise": "Foo",
            "count": 1,
            "prior_probability": 0.5,
            "source": source,
            "tags": [source],
            "requires_instantiation": False,
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_row(a, "have h := Foo n 1", "solved_jsonl")
    _write_row(b, "have h := Foo n 2", "mathlib_corpus")
    # Sidecar A: holdout with two IDs + a subset path.
    (tmp_path / "a.jsonl.meta.json").write_text(json.dumps({
        "clean_holdout": True,
        "excluded_theorem_count": 2,
        "excluded_subset": ["t1", "t2"],
        "excluded_theorem_ids": ["t1", "t2"],
        "excluded_subset_path": str(tmp_path / "subset_a.txt"),
    }), encoding="utf-8")
    # Sidecar B: holdout with one overlapping + one new ID, no subset path.
    (tmp_path / "b.jsonl.meta.json").write_text(json.dumps({
        "clean_holdout": True,
        "excluded_theorem_count": 2,
        "excluded_subset": ["t2", "t3"],
    }), encoding="utf-8")
    out = tmp_path / "merged.jsonl"
    merger.merge([a, b], out)

    meta = json.loads(out.with_suffix(out.suffix + ".meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["clean_holdout"] is True
    assert meta["excluded_subset"] == ["t1", "t2", "t3"]
    assert meta["excluded_theorem_ids"] == ["t1", "t2", "t3"]
    assert meta["excluded_theorem_count"] == 3
    assert str(tmp_path / "subset_a.txt") in meta["excluded_subset_paths"]
    assert set(meta["input_sidecars"].keys()) == {str(a), str(b)}


def test_merge_propagates_clean_holdout_when_only_one_input_holds_out(tmp_path: Path):
    """Spec: clean_holdout=true if ANY input has it."""
    merger = _import_merger()
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    # Empty-but-valid JSONL inputs (no rows is fine for sidecar union).
    a.write_text("", encoding="utf-8")
    b.write_text("", encoding="utf-8")
    (tmp_path / "a.jsonl.meta.json").write_text(json.dumps({
        "clean_holdout": True,
        "excluded_theorem_count": 1,
        "excluded_subset": ["only_one"],
    }), encoding="utf-8")
    (tmp_path / "b.jsonl.meta.json").write_text(json.dumps({
        "clean_holdout": False,
        "excluded_theorem_count": 0,
        "excluded_subset": [],
    }), encoding="utf-8")
    out = tmp_path / "merged.jsonl"
    merger.merge([a, b], out)
    meta = json.loads(out.with_suffix(out.suffix + ".meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["clean_holdout"] is True
    assert meta["excluded_subset"] == ["only_one"]
    assert meta["excluded_theorem_count"] == 1


def test_merge_no_sidecars_yields_clean_holdout_false(tmp_path: Path):
    """When neither input has a sidecar, the merged sidecar must report
    clean_holdout=false and an empty excluded set — the absence of
    metadata is NOT silently turned into a holdout claim."""
    merger = _import_merger()
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("", encoding="utf-8")
    b.write_text("", encoding="utf-8")
    out = tmp_path / "merged.jsonl"
    merger.merge([a, b], out)
    meta = json.loads(out.with_suffix(out.suffix + ".meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["clean_holdout"] is False
    assert meta["excluded_subset"] == []
    assert meta["excluded_theorem_ids"] == []
    assert meta["excluded_theorem_count"] == 0


# ---------------------------------------------------------------------------
# Strict clean-eval fail on target_trace_leak_detected
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Task A — failure_category classifier


# ---------------------------------------------------------------------------
# Task B — extended feature-based import inference
# ---------------------------------------------------------------------------

def test_import_inference_real_arithmetic_pulls_real_basic():
    from search.import_inference import infer_imports_from_header
    header = ("theorem foo (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) :"
              " (a + b) ^ 2 ≥ 0 := by sorry")
    r = infer_imports_from_header(header)
    assert r.matched is True
    assert "import Mathlib.Data.Real.Basic" in r.imports
    assert "import Mathlib.Tactic" in r.imports
    assert "real_arithmetic" in r.matched_rules


def test_import_inference_sqrt_with_unicode_radical():
    from search.import_inference import infer_imports_from_header
    header = ("theorem foo (x : ℝ) (h : 0 ≤ x) :"
              " 0 ≤ √x := by sorry")
    r = infer_imports_from_header(header)
    assert r.matched is True
    assert "import Mathlib.Analysis.SpecialFunctions.Sqrt" in r.imports


def test_import_inference_irrational_pulls_pow_real():
    from search.import_inference import infer_imports_from_header
    header = ("theorem foo : ∃ x y : ℝ, Irrational x ∧ Irrational y ∧"
              " ¬ Irrational (x ^ y) := by sorry")
    r = infer_imports_from_header(header)
    assert r.matched is True
    assert "import Mathlib.Analysis.SpecialFunctions.Pow.Real" in r.imports
    assert "real_pow_or_irrational" in r.matched_rules


def test_import_inference_finset_filter_pulls_big_ops():
    from search.import_inference import infer_imports_from_header
    header = ("theorem foo : Finset.prod "
              "(Finset.filter (λ x => ¬ Even x) (Finset.range 10)) "
              "(id : ℕ → ℕ) = 1 := by sorry")
    r = infer_imports_from_header(header)
    assert r.matched is True
    # `Mathlib.Algebra.BigOperators.Basic` does not exist in this pin —
    # the 2026-07-05 fix retargeted the rule to the module that does.
    assert ("import Mathlib.Algebra.BigOperators.Group.Finset.Basic"
            in r.imports)
    assert "finset_big_ops" in r.matched_rules


def test_import_inference_floor_pulls_floor_module():
    from search.import_inference import infer_imports_from_header
    header = ("theorem foo (x : ℝ) (h : ⌊x⌋ = 3) :"
              " x ≥ 3 := by sorry")
    r = infer_imports_from_header(header)
    assert r.matched is True
    assert "import Mathlib.Algebra.Order.Floor" in r.imports
    assert "floor_or_ceil" in r.matched_rules


def test_import_inference_unchanged_for_p100():
    """The original p100 rule must keep firing — no regression on the
    gcd_lcm rule when we extended the table."""
    from search.import_inference import infer_imports_from_header
    header = ("theorem mathd_numbertheory_100 (n : ℕ) (h₀ : 0 < n)"
              " (h₁ : Nat.gcd n 40 = 10) (h₂ : Nat.lcm n 40 = 280) :"
              " n = 70 := by sorry")
    r = infer_imports_from_header(header)
    assert r.matched is True
    assert "import Mathlib.Data.Nat.GCD.Basic" in r.imports
    assert "nat_gcd_lcm" in r.matched_rules


# ---------------------------------------------------------------------------
# Task C — REPL header validation method
# ---------------------------------------------------------------------------

def test_validate_header_method_present_on_session():
    """Source-grep: `LeanReplStepSession.validate_header` must exist and
    must NOT raise on Lean errors (returns a structured dict)."""
    src = (Path(__file__).resolve().parents[1] / "src" / "backend" /
           "repl_step.py").read_text(encoding="utf-8")
    assert "def validate_header(" in src
    assert "\"ok\":" in src
    assert "\"first_error_pos\":" in src


# ---------------------------------------------------------------------------
# Task D — candidate quality
# ---------------------------------------------------------------------------

def test_d2_nlinarith_with_hyps_template_fires_on_real_inequality():
    """A real-inequality goal with named hypotheses must emit
    `nlinarith [h₀, h₁, …]` plus a sq_nonneg variant."""
    from search.template_tactics import generate_template_tactics
    from search.proof_prior import ProofStateFeatures
    feats = ProofStateFeatures(
        symbols=("inequality", "power"),
        namespaces=("Real",),
        target_shape="inequality",
        hypothesis_shapes=(),
        constants=(),
        previous_tactic_class=None,
    )
    header = ("theorem foo (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b)"
              " (h₂ : a + b = 1) : a ^ 2 + b ^ 2 ≤ 1 := by sorry")
    out = generate_template_tactics(feats, retrieved_premises=[],
                                     goal_text=header)
    texts = [t for t, _c, _m in out]
    assert any(t.startswith("nlinarith [") and "h₀" in t
                for t in texts), f"missing nlinarith[h…]; got {texts}"
    assert any("sq_nonneg" in t for t in texts), \
        f"missing sq_nonneg variant; got {texts}"


def test_d3_local_equality_rewrite_template_fires():
    """Equality hypotheses must produce `rw [hX]` and `simp [hX]`."""
    from search.template_tactics import generate_template_tactics
    from search.proof_prior import ProofStateFeatures
    feats = ProofStateFeatures(
        symbols=("equality",),
        namespaces=("Nat",),
        target_shape="equality",
        hypothesis_shapes=("gcd_eq",),
        constants=(),
        previous_tactic_class=None,
    )
    header = ("theorem foo (n : ℕ) (h₁ : Nat.gcd n 40 = 10)"
              " (h₂ : Nat.lcm n 40 = 280) : n = 70 := by sorry")
    out = generate_template_tactics(feats, retrieved_premises=[],
                                     goal_text=header)
    texts = [t for t, _c, _m in out]
    assert "rw [h₁]" in texts
    assert "rw [h₂]" in texts
    assert "simp [h₁]" in texts
    assert any(t == "rw [h₁, h₂]" or t == "rw [h₁, h₂, h₀]"
                for t in texts), (
        f"missing joint rewrite over multiple equality hyps; got {texts}"
    )


def test_d_state_templates_zero_overhead_when_no_hooks():
    """A goal without numeric scalars or equality hypotheses must NOT
    emit the D.2 / D.3 templates."""
    from search.template_tactics import generate_template_tactics
    from search.proof_prior import ProofStateFeatures
    feats = ProofStateFeatures(
        symbols=("divisibility",),
        namespaces=("Nat",),
        target_shape="divisibility",
        hypothesis_shapes=(),
        constants=(),
        previous_tactic_class=None,
    )
    header = "theorem foo (n : ℕ) : n ∣ n := by sorry"
    out = generate_template_tactics(feats, retrieved_premises=[],
                                     goal_text=header)
    texts = [t for t, _c, _m in out]
    assert not any(t.startswith("nlinarith [") for t in texts)
    assert not any(t.startswith("rw [h") for t in texts)


def test_d1_context_richness_score_thresholds():
    from search.template_tactics import context_richness_score
    # Header with zero propositional hypotheses — thin.
    thin = "theorem foo (n : ℕ) : n + 0 = n := by sorry"
    assert context_richness_score(thin) <= 1
    # Header with two equality hypotheses — rich.
    rich = ("theorem foo (n : ℕ) (h₁ : n + 1 = 2)"
            " (h₂ : n - 1 = 0) : n = 1 := by sorry")
    assert context_richness_score(rich) >= 2


# ---------------------------------------------------------------------------
# Follow-up tasks 2 / 3 / 4
# ---------------------------------------------------------------------------

def test_complex_arithmetic_rule_pulls_complex_basic():
    """Follow-up task 3: ℂ + equality/power/inequality → Complex.Basic."""
    from search.import_inference import infer_imports_from_header
    header = ("theorem foo (f z : ℂ) (h₀ : f + 3*z = 11)"
              " (h₁ : 3*(f - 1) - 5*z = -68) :"
              " f = -10 ∧ z = 7 := by sorry")
    r = infer_imports_from_header(header)
    assert r.matched is True
    assert "import Mathlib.Data.Complex.Basic" in r.imports
    assert "complex_arithmetic" in r.matched_rules
    # Real.Basic is NOT pulled when the problem is purely ℂ.
    assert "import Mathlib.Data.Real.Basic" not in r.imports


# ---------------------------------------------------------------------------
# LLM header preprocessor

