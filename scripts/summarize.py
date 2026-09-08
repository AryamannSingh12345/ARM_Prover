"""Print a comparison table across all results/ablation_*.jsonl files.

Per config:  problems, solved, pass_rate, mean_wall_s, mean_expanded,
             mean_failed, set(solved_ids)

Also prints: problems baseline missed but premise-graph solved, and vice versa.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    configs = sorted(RESULTS.glob("ablation_*.jsonl"))
    if not configs:
        print("no ablation_*.jsonl in results/")
        return 1

    rows: dict[str, list[dict]] = {}
    for p in configs:
        rows[p.stem.replace("ablation_", "")] = load(p)

    header = ("config", "n", "solved", "pass%", "mean_wall_s",
              "mean_exp", "mean_fail")
    print(f"{header[0]:<20} {header[1]:>4} {header[2]:>6} {header[3]:>6} "
          f"{header[4]:>12} {header[5]:>10} {header[6]:>10}")
    print("-" * 80)

    solved_by: dict[str, set[str]] = {}
    for name, rs in rows.items():
        n = len(rs)
        solved = [r for r in rs if r.get("outcome") == "solved"]
        solved_by[name] = {r["id"] for r in solved}
        mean_wall = sum(r.get("wall_s", 0) for r in rs) / max(n, 1)
        mean_exp = sum(r.get("nodes_expanded", 0) for r in rs) / max(n, 1)
        mean_fail = sum(r.get("nodes_failed_tactic", 0) for r in rs) / max(n, 1)
        pass_pct = 100.0 * len(solved) / max(n, 1)
        print(f"{name:<20} {n:>4} {len(solved):>6} {pass_pct:>5.1f}% "
              f"{mean_wall:>12.1f} {mean_exp:>10.2f} {mean_fail:>10.2f}")

    if "baseline" in solved_by:
        base = solved_by["baseline"]
        print("\n--- problems gained by premise-graph (over baseline) ---")
        gained = defaultdict(list)
        for name, ids in solved_by.items():
            if name == "baseline":
                continue
            for pid in ids - base:
                gained[name].append(pid)
        for name, pids in gained.items():
            print(f"  {name}: {sorted(pids)}")
        print("\n--- problems lost by premise-graph (baseline solved, premise did not) ---")
        for name, ids in solved_by.items():
            if name == "baseline":
                continue
            lost = sorted(base - ids)
            if lost:
                print(f"  {name}: {lost}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
