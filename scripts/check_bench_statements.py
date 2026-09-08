"""Compile-check benchmark STATEMENTS against the pinned Mathlib.

For each problem in a subset, build exactly the source the runners would
send to Lean — inferred imports (same `infer_imports_from_header` path as
run_dag/run_minif2f), open-scopes prepended, `:= by sorry` body — and run
it through `compile_lean`. A statement passes iff the output has no
`error:` line and no timeout; the `declaration uses 'sorry'` warning is
expected and ignored (we are checking the statement, not a proof).

No API quota is used. Rows append to a JSONL (resume-safe: already-checked
ids are skipped), so the script can be re-run after fixing stragglers.

Usage:
  python scripts/check_bench_statements.py \
      --subset data/proofnet_dev30.txt --bench-dir data/proofnet/Test
  python scripts/check_bench_statements.py \
      --subset data/putnam_slice20.txt --bench-dir data/putnambench/Test

A statement that FAILS here would also fail in a real prover run with the
same import inference — fix via import-inference rules or a per-run
--lean-imports override, or drop it from the slice and document why.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backend.compile_verify import compile_lean, _ERROR_LINE_RE  # noqa: E402
from eval.run_minif2f import load_problem, prepend_open_scopes  # noqa: E402
from search.import_inference import infer_imports_from_header  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subset", required=True)
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default="results/bench_statement_check.jsonl")
    args = ap.parse_args()

    bench_dir = ROOT / args.bench_dir
    out_path = ROOT / args.out
    out_path.parent.mkdir(exist_ok=True)

    done: set[tuple[str, str]] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done.add((r["bench_dir"], r["id"]))
            except Exception:
                pass

    ids = [l.strip() for l in (ROOT / args.subset).read_text(encoding="utf-8").splitlines()
           if l.strip() and not l.startswith("#")]
    n_ok = n_fail = 0
    for i, pid in enumerate(ids, 1):
        if (args.bench_dir, pid) in done:
            continue
        try:
            header, src = load_problem(pid, bench_dir)
        except Exception as e:
            row = {"bench_dir": args.bench_dir, "id": pid, "statement_ok": False,
                   "stage": "load", "error_head": str(e)[:500]}
            n_fail += 1
        else:
            inferred = infer_imports_from_header(header)
            # opens/prelude already live inside `header` (eval.loader)
            source = f"{inferred.imports}\n\n{header} := by sorry\n"
            # reject_sorry=False: this tool asks "does the STATEMENT
            # elaborate", so its source is sorry-terminated BY DESIGN and
            # the verdict comes from the error text, not `ok`. With the
            # default short-circuit no compile would run and every
            # statement would be reported ok — including broken ones.
            res = compile_lean(source, timeout_s=args.timeout,
                               reject_sorry=False)
            has_error = bool(_ERROR_LINE_RE.search(res.errors)) \
                or res.errors.startswith("timeout")
            row = {
                "bench_dir": args.bench_dir, "id": pid,
                "statement_ok": not has_error,
                "stage": "compile",
                "import_source": "inferred" if inferred.matched else "fallback_mathlib",
                "verify_imports": inferred.imports,
                "elapsed_s": round(res.elapsed_s, 1),
                "error_head": res.errors[:500] if has_error else None,
            }
            n_ok += not has_error
            n_fail += has_error
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(ids)}] {pid}: "
              f"{'OK' if row['statement_ok'] else 'FAIL (' + row['stage'] + ')'}",
              flush=True)

    print(f"done: {n_ok} ok, {n_fail} fail "
          f"({len(done)} previously checked, skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
