"""Reconstruct the complete, compilable Lean file for a solved problem.

A result row stores `verify_imports` and `assembled_proof` but NOT the
theorem header, so neither field alone is checkable. This rebuilds the
exact source the verifier accepted:

    <imports>

    <theorem header> := by
    <assembled proof>

by re-loading the header from the bench directory with the same loader
the runner used. The output is what you hand to someone else, or compile
independently — the point of a solve is an artifact anyone can check.

Usage:
    python scripts/show_solution.py <run-id> [problem-id] [-o DIR]
    python scripts/show_solution.py --list

`--verify` re-runs a fresh Lean compile on the reconstructed file, which
is the only thing that actually confirms the artifact (and re-checks the
`sorry`/axiom guards).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"


def _rows(jsonl: Path) -> list[dict]:
    out = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _bench_dir(row: dict) -> Path:
    bd = row.get("bench_dir") or "minif2f"
    if bd == "minif2f":
        return ROOT / "data" / "miniF2F" / "MiniF2F" / "Test"
    p = Path(bd)
    return p if p.is_absolute() else ROOT / p


def reconstruct(row: dict) -> str | None:
    """The full Lean source, or None when the row has no proof.

    Prefers `verified_header` -- the header the proof was actually
    compiled against, defs and store lemmas included. Falls back to the
    bench header for rows written before that field existed; such a row
    reconstructs only when the proof cites nothing outside Mathlib, and
    otherwise fails on unknown identifiers rather than quietly producing
    a file that is not the proof that was verified.
    """
    proof = row.get("assembled_proof")
    if not proof:
        return None
    header = row.get("verified_header")
    if not header:
        from eval.loader import load_problem
        header, _src = load_problem(row["id"], _bench_dir(row))
    imports = row.get("verify_imports") or "import Mathlib"
    return f"{imports}\n\n{header} := by\n{proof}\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?", help="run id, or a path to a .jsonl")
    ap.add_argument("problem_id", nargs="?",
                    help="problem to show; default = every solved row")
    ap.add_argument("-o", "--out-dir", default=None,
                    help="write <problem>.lean files here instead of stdout")
    ap.add_argument("--list", action="store_true",
                    help="list runs that contain at least one solved row")
    ap.add_argument("--verify", action="store_true",
                    help="re-compile the reconstructed file in a fresh Lean "
                         "session (slow; the only real confirmation)")
    args = ap.parse_args()

    if args.list or not args.run_id:
        any_found = False
        for j in sorted(RESULTS.glob("*.jsonl")):
            solved = [r for r in _rows(j) if r.get("outcome") == "solved"]
            if solved:
                any_found = True
                print(f"{j.stem:<40} {len(solved)} solved: "
                      + ", ".join(r["id"] for r in solved[:6]))
        if not any_found:
            print("no solved rows in results/")
        return 0

    p = Path(args.run_id)
    jsonl = p if p.suffix == ".jsonl" else RESULTS / f"{args.run_id}.jsonl"
    if not jsonl.exists():
        print(f"no such run: {jsonl}", file=sys.stderr)
        return 1

    rows = [r for r in _rows(jsonl)
            if r.get("outcome") == "solved"
            and (args.problem_id is None or r["id"] == args.problem_id)]
    if not rows:
        print("no solved rows matched", file=sys.stderr)
        return 1

    rc = 0
    for row in rows:
        src = reconstruct(row)
        if src is None:
            print(f"[{row['id']}] row has no assembled_proof", file=sys.stderr)
            rc = 1
            continue
        if args.out_dir:
            d = Path(args.out_dir)
            d.mkdir(parents=True, exist_ok=True)
            f = d / f"{row['id']}.lean"
            f.write_text(src, encoding="utf-8")
            print(f"wrote {f}  ({len(src)} chars)")
        else:
            print(f"{'=' * 70}\n{row['id']}  "
                  f"[{row.get('model')}, {row.get('wall_s')}s]\n{'=' * 70}")
            print(src)
        if args.verify:
            from backend.compile_verify import compile_lean
            res = compile_lean(src, timeout_s=2400)
            status = "OK" if (res.ok and not res.used_sorry) else "FAILED"
            print(f"[{row['id']}] fresh compile: {status} "
                  f"({res.elapsed_s:.0f}s, sorry={res.used_sorry})")
            if status == "FAILED":
                print((res.errors or "")[:1500])
                rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
