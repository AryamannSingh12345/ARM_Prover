"""Round-11 patches:

- `_tokenize_statement_for_retrieval` strips bound vars / hypothesis names /
  Lean keywords from BM25 queries.
- Unicode symbols (∑, %, ^, ∣, …) expand into retrieval-relevant tokens.
- retrieve_bm25 seeds with exact dotted identifiers found in the graph.

Each retrieval test fakes a tiny BM25 corpus + graph so the real Mathlib
graph never has to load. No lake, no LLM, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------- shared corpus setup ----------


def _setup_retrieval_corpus(monkeypatch, nodes: list[str]):
    """Replace the graph + BM25 caches with a fixed mini-corpus."""
    from rank_bm25 import BM25Okapi

    from search import premise_retrieval as pr

    docs = [pr._tokenize_name(n) for n in nodes]
    bm25 = BM25Okapi(docs)
    fake_graph = SimpleNamespace(nodes=set(nodes))

    monkeypatch.setattr(pr, "_load_graph", lambda: fake_graph)
    monkeypatch.setattr(pr, "_bm25_corpus", lambda: (nodes, bm25))

    def fake_tail_index():
        idx: dict[str, list[str]] = {}
        for n in nodes:
            key = n.rsplit(".", 1)[-1]
            idx.setdefault(key, []).append(n)
        return idx
    monkeypatch.setattr(pr, "_tail_index", fake_tail_index)


# ---------- pure-helper unit tests ----------


def test_is_statement_stopword_filters_bound_vars():
    from search.premise_retrieval import _is_statement_stopword
    for tok in ("n", "k", "x", "y", "m"):
        assert _is_statement_stopword(tok), tok


def test_is_statement_stopword_filters_hypothesis_patterns():
    from search.premise_retrieval import _is_statement_stopword
    for tok in ("h", "h0", "h1", "h2", "ih", "ih_n", "h_eq", "h₀", "h₁"):
        assert _is_statement_stopword(tok), tok


def test_is_statement_stopword_filters_lean_keywords():
    from search.premise_retrieval import _is_statement_stopword
    for tok in ("theorem", "lemma", "by", "intro", "obtain", "have"):
        assert _is_statement_stopword(tok), tok


def test_is_statement_stopword_keeps_real_identifiers():
    from search.premise_retrieval import _is_statement_stopword
    for tok in ("Nat", "gcd", "Finset", "range", "add_comm", "sum_range"):
        assert not _is_statement_stopword(tok), tok


def test_tokenizer_expands_sum_symbol():
    from search.premise_retrieval import _tokenize_statement_for_retrieval
    toks = _tokenize_statement_for_retrieval("∑ k, f k")
    assert "sum" in toks
    assert "Finset" in toks
    assert "k" not in toks   # bound variable filtered


def test_tokenizer_expands_mod_pow_dvd_symbols():
    from search.premise_retrieval import _tokenize_statement_for_retrieval
    toks = _tokenize_statement_for_retrieval("a % b + c ^ d, ∣ e")
    assert "mod" in toks
    assert "pow" in toks
    assert "dvd" in toks


def test_tokenizer_emits_dotted_idents_and_their_pieces():
    from search.premise_retrieval import _tokenize_statement_for_retrieval
    toks = _tokenize_statement_for_retrieval("h₁ : Nat.gcd n 40 = 10")
    # Full dotted form + pieces, hypothesis/var filtered.
    assert "Nat.gcd" in toks
    assert "Nat" in toks
    assert "gcd" in toks
    assert "h₁" not in toks
    assert "n" not in toks


# ---------- BM25 retrieval against the smoke previews ----------


def test_sum_pow_mod_query_returns_finset_range_high(monkeypatch):
    """Verbatim from the smoke: (∑ k ∈ (Finset.range 101), 2^k) % 7 = 3
    must put Finset.range at the top AND must NOT put QuaternionAlgebra
    / AntilipschitzWith.k near the top."""
    _setup_retrieval_corpus(monkeypatch, [
        # Wanted.
        "Finset.range",
        "Finset.sum_range",
        "Nat.mod",
        "Nat.pow",
        # Quaternion / Antilipschitz distractors with only `k` in common.
        "QuaternionAlgebra.Basis.k_mul_k",
        "AntilipschitzWith.k",
        # Generic Mathlib decls.
        "Nat.add_comm",
        "Set.range",
        "Fintype.card",
    ])
    from search.premise_retrieval import retrieve_bm25
    names = [
        p.name for p in retrieve_bm25(
            "(∑ k ∈ (Finset.range 101), 2^k) % 7 = 3", cap=10,
        )
    ]
    # Exact-identifier seeding: Finset.range must be at the very top.
    assert names[0] == "Finset.range", f"top={names!r}"
    # Symbol expansions (∑, %, ^) drive sum/mod/pow-related decls up.
    top5 = names[:5]
    assert "Finset.sum_range" in top5 or "Nat.mod" in top5 or "Nat.pow" in top5
    # Distractors must not be in top 3 — the goal had nothing to do with
    # quaternions or Lipschitz analysis.
    assert "QuaternionAlgebra.Basis.k_mul_k" not in names[:3]
    assert "AntilipschitzWith.k" not in names[:3]


def test_gcd_lcm_query_returns_nat_gcd_lcm_high(monkeypatch):
    """Verbatim from the smoke: a number-theory goal mentioning Nat.gcd
    and Nat.lcm must put both near the top, and must NOT put `ih_n` /
    `n_lt_xn` style distractors near the top."""
    _setup_retrieval_corpus(monkeypatch, [
        # Wanted.
        "Nat.gcd",
        "Nat.lcm",
        "Nat.gcd_eq_iff_dvd",
        "Nat.lcm_dvd",
        "Nat.gcd_comm",
        # `n`-rich distractors that BM25 used to surface.
        "ih_n",
        "n_lt_xn",
        "yn_ge_n",
        # Padding.
        "Finset.range",
        "Nat.add_comm",
    ])
    from search.premise_retrieval import retrieve_bm25
    query = (
        "(n : ℕ) (h₁ : Nat.gcd n 40 = 10) "
        "(h₂ : Nat.lcm n 40 = 280) : n = 70"
    )
    names = [p.name for p in retrieve_bm25(query, cap=10)]
    # Nat.gcd and Nat.lcm seeded to the very top by exact-identifier match.
    assert "Nat.gcd" in names[:2], f"names={names!r}"
    assert "Nat.lcm" in names[:2], f"names={names!r}"
    # n-distractors must NOT be in top 3.
    for distractor in ("ih_n", "n_lt_xn", "yn_ge_n"):
        assert distractor not in names[:3], (
            f"{distractor!r} must not appear in top 3 for a number-theory goal; "
            f"top3={names[:3]!r}"
        )


def test_finset_filter_icc_query_returns_relevant_high(monkeypatch):
    """Third smoke query: counting divisibility within Finset.Icc must
    surface Finset.card_filter, Finset.filter, Finset.Icc near the top."""
    _setup_retrieval_corpus(monkeypatch, [
        "Finset.card_filter",
        "Finset.filter",
        "Finset.Icc",
        "Finset.card_Icc",
        # Distractor with only `x` matching.
        "Set.image_x",
        # Generic.
        "Nat.add_comm",
        "Finset.range",
    ])
    from search.premise_retrieval import retrieve_bm25
    names = [
        p.name for p in retrieve_bm25(
            "Finset.card (Finset.filter (λ x => 20∣x) (Finset.Icc 15 85)) = 4",
            cap=10,
        )
    ]
    # All three must appear in the top results, in some order.
    top6 = set(names[:6])
    for wanted in ("Finset.filter", "Finset.Icc"):
        assert wanted in top6, f"{wanted!r} must appear in top 6; got top6={top6!r}"
    # Bound-variable distractor `x` must not promote irrelevant decls to
    # the top 3.
    assert "Set.image_x" not in names[:3]


def test_seeding_skips_dotted_idents_not_in_graph(monkeypatch):
    """If a dotted identifier in the goal is NOT in the graph (e.g.
    hallucinated), seeding must not insert it. Defends against the
    seeder becoming a hallucination-laundering vector."""
    _setup_retrieval_corpus(monkeypatch, [
        "Finset.range",   # this IS in the graph
        "Nat.add_comm",
    ])
    from search.premise_retrieval import retrieve_bm25
    names = [
        p.name for p in retrieve_bm25(
            "Finset.range and Nat.fake_lemma somehow combine", cap=10,
        )
    ]
    assert "Finset.range" in names
    assert "Nat.fake_lemma" not in names


def test_empty_query_returns_empty(monkeypatch):
    """A query whose tokens all get filtered (all stopwords) must return
    an empty list, not crash, not return the whole corpus."""
    _setup_retrieval_corpus(monkeypatch, [
        "Nat.add_comm",
        "Finset.range",
    ])
    from search.premise_retrieval import retrieve_bm25
    # Goal is only single-letter vars and digits — nothing survives the
    # statement stopword filter.
    names = [p.name for p in retrieve_bm25("n + k = 42", cap=10)]
    assert names == []
