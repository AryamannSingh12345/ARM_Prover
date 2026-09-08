"""Deterministic random subset selector for miniF2F-test.

Usage: python scripts/pick_subset.py --n 50 --seed 17 --out data/ablation_50.txt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/miniF2F/MiniF2F/Test")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="data/ablation_50.txt")
    ap.add_argument("--stratify", action="store_true",
                    help="Proportional stratified sample by problem-name "
                         "category (mathd_algebra / mathd_numbertheory / "
                         "competition / other) instead of uniform random — "
                         "slice solve rates then estimate the full-set rate.")
    args = ap.parse_args()

    src = ROOT / args.source
    all_ids = sorted(p.stem for p in src.glob("*.lean"))
    if args.stratify:
        from collections import defaultdict
        from prepare_benchmarks import _proportional_sample

        def cat(n: str) -> str:
            if n.startswith("mathd_algebra"):
                return "mathd_algebra"
            if n.startswith("mathd_numbertheory"):
                return "mathd_numbertheory"
            if n.startswith(("aime", "amc", "imo")):
                return "competition"
            return "other"

        buckets: dict[str, list[str]] = defaultdict(list)
        for pid in all_ids:
            buckets[cat(pid)].append(pid)
        picked = _proportional_sample(buckets, args.n, args.seed)
    else:
        rng = random.Random(args.seed)
        picked = sorted(rng.sample(all_ids, args.n))
    out = ROOT / args.out
    out.write_text("\n".join(picked) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(picked)} problems, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
