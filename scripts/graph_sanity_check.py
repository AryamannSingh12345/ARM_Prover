"""Sanity-check the built Mathlib graph.

Loads nodes.jsonl + edges.jsonl into NetworkX. Reports:
- node count, edge count, density
- in/out degree distributions (percentiles)
- top-50 highest-in-degree nodes (should look like core decls)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = ROOT / "data" / "mathlib_graph"


def main() -> int:
    G = nx.DiGraph()
    with (GRAPH_DIR / "nodes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            G.add_node(d["name"], **{k: v for k, v in d.items() if k != "name"})
    with (GRAPH_DIR / "edges.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            G.add_edge(d["from"], d["to"])

    print(f"nodes = {G.number_of_nodes():,}")
    print(f"edges = {G.number_of_edges():,}")
    print(f"density = {nx.density(G):.6f}")

    in_deg = [d for _, d in G.in_degree()]
    out_deg = [d for _, d in G.out_degree()]
    def pct(xs, p):
        xs2 = sorted(xs)
        return xs2[int(len(xs2) * p / 100)] if xs2 else 0
    print("\nin-degree   p50/p90/p99/max =",
          f"{pct(in_deg,50)}/{pct(in_deg,90)}/{pct(in_deg,99)}/{max(in_deg) if in_deg else 0}")
    print("out-degree  p50/p90/p99/max =",
          f"{pct(out_deg,50)}/{pct(out_deg,90)}/{pct(out_deg,99)}/{max(out_deg) if out_deg else 0}")

    kind_counts = Counter(G.nodes[n].get("kind", "?") for n in G.nodes)
    print("\nnode kinds:", dict(kind_counts.most_common()))

    print("\ntop-50 highest in-degree (should look like Eq.refl, Nat.succ, etc.):")
    top = sorted(G.in_degree(), key=lambda kv: -kv[1])[:50]
    for name, deg in top:
        print(f"  {deg:>6}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
