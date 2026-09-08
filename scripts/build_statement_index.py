"""Build the Mathlib/core STATEMENT index used by recognition.

    python scripts/build_statement_index.py [--out data/mathlib_statements.jsonl]

Derived entirely from the Lean sources the pin puts on disk — no network,
no model, no API spend. Rebuild it whenever `lean/lean-toolchain` or the
Mathlib pin in `lean/lake-manifest.json` changes; a stale index cannot
produce an unsound proof (every candidate is kernel-gated) but it will
quietly stop retrieving things.

The output is one JSON object per declaration:
`{name, kind, statement, module}`, plus `synthetic: true` on twins
synthesised from `@[to_additive]` — those exist in the Lean environment
but in no source file, and they are exactly the `∑` lemmas competition
problems need.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from search.recognize import build_index, source_roots   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "data"
                                         / "mathlib_statements.jsonl"))
    ap.add_argument("--project-root", default=str(ROOT))
    args = ap.parse_args()

    roots = source_roots(args.project_root)
    if not roots:
        print("no Lean sources found — is lean/.lake/packages populated?",
              file=sys.stderr)
        return 1
    print("indexing:")
    for d, prefix in roots:
        n = sum(1 for _ in d.rglob("*.lean"))
        print(f"  {n:>5} files  {prefix or '(no prefix)':<10} {d}")

    t = time.time()
    total = build_index(roots, args.out)
    size = Path(args.out).stat().st_size
    print(f"\n{total} declarations in {time.time() - t:.1f}s "
          f"-> {args.out} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
