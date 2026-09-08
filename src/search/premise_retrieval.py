"""Candidate-premise retrieval for the hybrid priority.

Strategy A (graph-neighborhood):
  Extract identifier-shaped tokens from the goal text. For each token that hits
  a Mathlib node, take its k-hop closed neighborhood (default k=2) in the use-
  dependency graph. Union, dedup, cap at MAX_CANDIDATES.

Strategy B (BM25-over-names):
  rank_bm25 over tokenized Mathlib decl names. Query with the goal text. Return
  top-MAX_CANDIDATES.

Both return list[Premise] suitable for premise_scorer.PremiseScorer.score().
"""
from __future__ import annotations

import json
import re
from collections import deque
from functools import cache
from pathlib import Path
from typing import Iterable

from policy.premise_scorer import Premise

ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = ROOT / "data" / "mathlib_graph"

MAX_CANDIDATES = 200
NEIGHBORHOOD_HOPS = 2

# Tokens in goal text that look like a (possibly qualified) Lean identifier.
_GOAL_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


# ---- shared graph loader (lazy, cached) ------------------------------------

@cache
def _load_graph():
    import networkx as nx
    G = nx.DiGraph()
    with (GRAPH_DIR / "nodes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            G.add_node(d["name"], **{k: v for k, v in d.items() if k != "name"})
    with (GRAPH_DIR / "edges.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            # We want a *use* graph that goes TARGET <- USER for neighborhood
            # walks ("find lemmas in the call-graph neighborhood of identifiers
            # in the goal"). Adding both directions is cheap and gives 2-hop
            # walks the right behavior.
            G.add_edge(d["from"], d["to"])
    return G


@cache
def _tail_index() -> dict[str, list[str]]:
    """unqualified name -> list of fully-qualified node names"""
    G = _load_graph()
    from collections import defaultdict
    idx: dict[str, list[str]] = defaultdict(list)
    for n in G.nodes:
        idx[n.rsplit(".", 1)[-1]].append(n)
    return idx


# ---- Strategy A — graph neighborhood ---------------------------------------

def goal_identifiers(goal_text: str) -> list[str]:
    return [m.group(0) for m in _GOAL_IDENT_RE.finditer(goal_text)]


def retrieve_graph(goal_text: str, hops: int = NEIGHBORHOOD_HOPS,
                   cap: int = MAX_CANDIDATES) -> list[Premise]:
    G = _load_graph()
    tail = _tail_index()
    seeds: set[str] = set()
    for ident in goal_identifiers(goal_text):
        if ident in G:
            seeds.add(ident)
            continue
        # Unqualified fallback: include all unambiguous tail-matches.
        cands = tail.get(ident)
        if cands and len(cands) <= 4:
            seeds.update(cands)
    if not seeds:
        return []
    # BFS to hops, on the undirected view of edges.
    visited = set(seeds)
    frontier = deque((s, 0) for s in seeds)
    while frontier and len(visited) < cap * 4:  # over-collect, prune later
        n, d = frontier.popleft()
        if d >= hops:
            continue
        # neighbors in either direction
        for nb in list(G.predecessors(n)) + list(G.successors(n)):
            if nb not in visited:
                visited.add(nb)
                frontier.append((nb, d + 1))
    # Rank by degree (popular things first), cap at MAX.
    ranked = sorted(visited, key=lambda n: -(G.in_degree(n) + G.out_degree(n)))
    return [Premise(name=n) for n in ranked[:cap]]


# ---- Strategy B — BM25 over names ------------------------------------------

@cache
def _bm25_corpus():
    """Tokenize each node name (Mathlib.Foo.bar_baz -> [Mathlib, Foo, bar, baz, bar_baz])
    and build a BM25Okapi index over that corpus."""
    from rank_bm25 import BM25Okapi
    G = _load_graph()
    names = list(G.nodes)
    docs = [_tokenize_name(n) for n in names]
    return names, BM25Okapi(docs)


def _tokenize_name(name: str) -> list[str]:
    # Split on dots, then on underscores; keep both whole-and-parts.
    out: list[str] = []
    for seg in name.split("."):
        out.append(seg)
        if "_" in seg:
            out.extend(seg.split("_"))
    return out


def _tokenize_goal(goal_text: str) -> list[str]:
    """Naive goal tokenizer; kept for backwards compat with any caller that
    expects the original behaviour. The smarter
    `_tokenize_statement_for_retrieval` is what `retrieve_bm25` now uses."""
    out: list[str] = []
    for tok in goal_identifiers(goal_text):
        out.extend(_tokenize_name(tok))
    return out


# ---- Smarter theorem-statement tokenization for retrieval -------------------
#
# The naive `_tokenize_goal` emits every identifier-shaped token from the
# goal, including bound variables (`n`, `k`), hypothesis names (`h`, `h₁`,
# `ih_n`), Lean keywords, and pure numbers. On Mathlib-scale BM25 these
# short / common tokens dilute relevance: short tokens have low DF and high
# IDF, so a decl like `QuaternionAlgebra.Basis.k_mul_k` matches a single
# `k` in the goal twice and dominates a goal that has nothing to do with
# quaternions. The fix below is purely on the QUERY side — the corpus
# tokenization is unchanged so legitimate short tokens in real Mathlib
# names still index.

# Tokens that should not drive retrieval on theorem-statement queries.
_STATEMENT_STOPWORDS = frozenset({
    # Single-letter bound variables — Lean theorem convention.
    *"abcdefghijklmnopqrstuvwxyz",
    # Multi-letter hypothesis-name roots.
    "ih", "hp", "hq", "hh",
    # Lean keywords seen in headers / hypotheses.
    "theorem", "lemma", "example", "by", "fun", "have", "show",
    "use", "intro", "obtain", "let", "in", "where", "do", "match",
    "show", "from",
    # Universe / very-generic typing tokens that match too many decls.
    "Type", "Prop",
})

# Hypothesis-name patterns: `h`, `h₁`, `h2`, `h_eq`, `ih`, `ih_n`, etc.
# We rely on the LLM/Lean convention that `h` or `ih` prefixes hypothesis
# variables.
_HYPOTHESIS_RE = re.compile(r"^(?:h|ih)(?:[_a-zA-Z₀-₉0-9]*)$")

# Unicode subscript range for "h₀", "h₁", etc.
_SUBSCRIPT_DIGIT_RANGE = range(0x2080, 0x208A)

# Unicode / symbolic Lean notation → retrieval-relevant token expansions.
# These bias BM25 toward decls that contain `sum`, `mod`, `pow`, `dvd` …
# even when the goal expresses the operation purely in symbols.
_SYMBOL_EXPANSIONS: dict[str, list[str]] = {
    "∑":  ["sum", "Finset", "range"],
    "∏":  ["prod", "Finset"],
    "%":  ["mod"],
    "^":  ["pow"],
    "∣":  ["dvd", "Dvd"],
    "≤":  ["le"],
    "≥":  ["ge"],
    "≠":  ["ne"],
    "∀":  ["forall"],
    "∃":  ["exists"],
    "→":  ["arrow", "implies"],
    "↔":  ["iff"],
}


def _is_statement_stopword(token: str) -> bool:
    """True iff `token` should not contribute to retrieval scoring on a
    theorem-statement query. Defensive: any of (empty | pure number |
    listed stopword | hypothesis-pattern | single subscript-bearing
    identifier) returns True."""
    if not token:
        return True
    if token.isdigit():
        return True
    if token in _STATEMENT_STOPWORDS:
        return True
    if _HYPOTHESIS_RE.match(token):
        return True
    # Short subscript-bearing identifiers like `h₁` / `n₀`.
    if len(token) <= 2 and any(
        c.isdigit() or ord(c) in _SUBSCRIPT_DIGIT_RANGE for c in token
    ):
        return True
    return False


def _tokenize_statement_for_retrieval(goal_text: str) -> list[str]:
    """Tokenizer tuned for theorem-statement BM25 queries.

    Improvements over `_tokenize_goal`:
      - filters bound variables (`n`, `k`, `x`), hypothesis names
        (`h`, `h₁`, `ih_n`), Lean keywords, and pure numbers;
      - emits BOTH the full dotted identifier (`Finset.range`) AND its
        split components — the full form helps the exact-identifier seeder,
        the split components feed BM25;
      - expands Unicode math symbols (∑, %, ^, ∣, …) into keyword tokens
        that drive retrieval toward sum/mod/pow/dvd-like decls.
    """
    out: list[str] = []
    for ident in goal_identifiers(goal_text):
        if "." in ident:
            out.append(ident)  # full dotted form
            for piece in _tokenize_name(ident):
                if not _is_statement_stopword(piece):
                    out.append(piece)
        elif not _is_statement_stopword(ident):
            out.append(ident)
    for sym, expansions in _SYMBOL_EXPANSIONS.items():
        if sym in goal_text:
            out.extend(expansions)
    return out


def retrieve_bm25(goal_text: str, cap: int = MAX_CANDIDATES) -> list[Premise]:
    """BM25 retrieval with two query-side improvements:

      1. Smarter tokenization (`_tokenize_statement_for_retrieval`) drops
         bound-variable / hypothesis-name noise.
      2. Exact-identifier seeding: any dotted identifier present in the
         goal text AND in the dependency graph is prepended to the
         result, guaranteeing it appears at the top regardless of BM25.

    Public signature is unchanged.
    """
    names, bm25 = _bm25_corpus()
    query = _tokenize_statement_for_retrieval(goal_text)
    if not query:
        return []
    import numpy as np
    scores = bm25.get_scores(query)
    # Exact-identifier seeding.
    G_nodes = _load_graph().nodes
    seeds: list[str] = []
    seen: set[str] = set()
    for ident in goal_identifiers(goal_text):
        if "." in ident and ident in G_nodes and ident not in seen:
            seen.add(ident)
            seeds.append(ident)
    # BM25 winners (skip zero score; skip duplicates of seeds).
    top_idx = np.argsort(scores)[::-1]
    ranked: list[str] = list(seeds)
    for i in top_idx:
        if len(ranked) >= cap:
            break
        n = names[i]
        if n in seen or scores[i] <= 0:
            continue
        seen.add(n)
        ranked.append(n)
    return [Premise(name=n) for n in ranked[:cap]]


# ---- dispatcher ------------------------------------------------------------

def retrieve(goal_text: str, strategy: str = "A",
             cap: int = MAX_CANDIDATES) -> list[Premise]:
    if strategy.upper() == "A":
        return retrieve_graph(goal_text, cap=cap)
    if strategy.upper() == "B":
        return retrieve_bm25(goal_text, cap=cap)
    raise ValueError(f"unknown strategy: {strategy}")
