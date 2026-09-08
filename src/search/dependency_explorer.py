"""Dependency-graph exploration.

Spec: PART 7 of the proof-prior series.

Walk the existing Mathlib use-dependency graph (the one loaded by
`search.premise_retrieval._load_graph`) outward from the seeds suggested
by the current proof state, score the neighbours by token overlap with
state symbols, and emit them as low-probability `ProofPriorMove` objects
of source "dependency_exploration".

These candidates are deliberately given a nonzero `exploration_epsilon`
probability even when their empirical prior is zero — they are the
"rare-but-syntactically-near" arm of the search.
"""
from __future__ import annotations

import math
import re
from typing import Iterable

from search.proof_prior import ProofPriorMove, ProofStateFeatures


EXPLORATION_EPSILON: float = 0.05
DEFAULT_HOPS: int = 2


def _tokens(name: str) -> set[str]:
    """Split a dotted Mathlib name into searchable tokens.

    `Real.sqrt_pos` -> {Real, sqrt, pos, sqrt_pos}
    `Nat.gcd_mul_lcm` -> {Nat, gcd, mul, lcm, gcd_mul_lcm}
    """
    out: set[str] = set()
    for seg in name.split("."):
        if not seg:
            continue
        out.add(seg)
        out.add(seg.lower())
        if "_" in seg:
            for piece in seg.split("_"):
                if piece:
                    out.add(piece)
                    out.add(piece.lower())
    return out


def _seed_names(
    features: ProofStateFeatures,
    retrieved_premises: Iterable[str],
    graph_nodes: set[str] | dict | None,
) -> list[str]:
    """Seeds for the walk: retrieved premises that are in the graph, plus
    any dotted identifier (`Foo.bar`) the features happen to mention."""
    seeds: list[str] = []
    seen: set[str] = set()

    def push(name: str) -> None:
        if name and name not in seen and (graph_nodes is None or name in graph_nodes):
            seen.add(name)
            seeds.append(name)

    for p in retrieved_premises:
        push(p)
    # Namespace + symbol pairs can also seed: `Nat` + `gcd` -> `Nat.gcd`.
    for ns in features.namespaces:
        for sym in features.symbols:
            push(f"{ns}.{sym}")
    return seeds


def _score_neighbour(name: str, state_symbol_tokens: set[str]) -> float:
    """Higher = more relevant. Token-overlap fraction in [0,1]."""
    toks = _tokens(name)
    if not toks or not state_symbol_tokens:
        return 0.0
    inter = toks & state_symbol_tokens
    return len(inter) / max(1, len(state_symbol_tokens))


def dependency_exploration_candidates(
    features: ProofStateFeatures,
    mathlib_graph,
    retrieved_premises: list[str] | None = None,
    top_k: int = 20,
    *,
    hops: int = DEFAULT_HOPS,
    exploration_epsilon: float = EXPLORATION_EPSILON,
) -> list[ProofPriorMove]:
    """Return up to `top_k` exploration moves.

    `mathlib_graph` is a `networkx.DiGraph` (the result of
    `search.premise_retrieval._load_graph`). Pass `None` to short-circuit
    when no graph is loaded; the result is then an empty list.
    """
    if mathlib_graph is None or top_k <= 0:
        return []
    try:
        nodes = mathlib_graph.nodes  # networkx duck-typing
    except Exception:
        return []

    retrieved = list(retrieved_premises or [])
    seeds = _seed_names(features, retrieved, nodes)
    if not seeds:
        return []

    state_tokens: set[str] = set()
    for sym in features.symbols:
        state_tokens.add(sym)
        state_tokens.add(sym.lower())
    for ns in features.namespaces:
        state_tokens.add(ns)
        state_tokens.add(ns.lower())

    # BFS to `hops`, undirected. Cap visited to top_k * 6 to bound cost.
    visited: dict[str, int] = {s: 0 for s in seeds}
    frontier: list[tuple[str, int]] = [(s, 0) for s in seeds]
    cap = max(50, top_k * 6)
    while frontier and len(visited) < cap:
        n, d = frontier.pop(0)
        if d >= hops:
            continue
        try:
            preds = list(mathlib_graph.predecessors(n))
            succs = list(mathlib_graph.successors(n))
        except Exception:
            continue
        for nb in preds + succs:
            if nb in visited:
                continue
            visited[nb] = d + 1
            frontier.append((nb, d + 1))

    # Score & rank: prefer non-seeds; tie-break by graph degree.
    candidates: list[tuple[float, int, str]] = []
    for name, dist in visited.items():
        if name in retrieved:
            continue  # already covered by template_tactics
        score = _score_neighbour(name, state_tokens)
        if score <= 0.0 and dist == 0:
            continue
        try:
            deg = mathlib_graph.in_degree(name) + mathlib_graph.out_degree(name)
        except Exception:
            deg = 0
        # Distance penalty: 1-hop neighbours over 2-hop.
        priority = score - 0.1 * dist
        candidates.append((priority, deg, name))
    candidates.sort(key=lambda t: (-t[0], -t[1], t[2]))

    out: list[ProofPriorMove] = []
    for _score, _deg, name in candidates[:top_k]:
        out.append(ProofPriorMove(
            tactic_template=f"have h := {name}",
            tactic_class="have_premise",
            premise=name,
            prior_probability=exploration_epsilon,
            prior_cost=-math.log(exploration_epsilon),
            source="dependency_exploration",
            tags=("dependency",),
            requires_instantiation=False,
        ))
    return out
