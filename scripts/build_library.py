"""Build a disposable Lean library for a concept, kernel-gated.

Answers the question `order5_lrs_a0_unit` raised: Mathlib has 21
`LinearRecurrence` declarations and no power-sum representation, so the
target was unreachable — the missing prerequisite had to be built first.
This builds it, or tells you cheaply that it cannot.

    python scripts/build_library.py \
      --spec "Power-sum representation for integer linear recurrence
              sequences: the general solution as a combination of geometric
              solutions when the characteristic roots are distinct." \
      --name LRSPowerSum \
      --out results/libraries/lrs_powersum \
      --max-decls 10 --attempts-per-decl 3

SANDBOX. Everything is in memory until the build finishes; the only files
written are `<out>/<name>.lean` and `<out>/manifest.json`. The persistent
cross-run bank (`results/invented_lemmas.lean`) is never touched, and
`lean/` is only used for the ordinary temp compiles every verify does. So

    rm -rf <out>            # or: --purge

is complete cleanup, and a junk build costs nothing but disk.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Never write a library here — these are real project state.
_PROTECTED = ("lean", "src", "scripts", "tests", "docs", "data")


def _check_out_dir(out: Path) -> None:
    """Refuse to write anywhere that could damage the project."""
    out = out.resolve()
    try:
        rel = out.relative_to(ROOT)
    except ValueError:
        return                      # outside the repo entirely: caller's call
    top = rel.parts[0] if rel.parts else ""
    if top in _PROTECTED:
        raise SystemExit(
            f"refusing to build into {rel} — that is project state. "
            f"Use results/libraries/<name> or a path outside the repo.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True,
                    help="informal description of the concept to build")
    ap.add_argument("--name", default="Library",
                    help="library name; also the .lean filename")
    ap.add_argument("--out", default=None,
                    help="output directory (default "
                         "results/libraries/<name>_<timestamp>)")
    ap.add_argument("--imports", default="import Mathlib")
    ap.add_argument("--max-decls", type=int, default=10)
    ap.add_argument("--attempts-per-decl", type=int, default=3)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--verify-timeout", type=int, default=1800)
    ap.add_argument("--purge", action="store_true",
                    help="delete the output directory first (a build is "
                         "disposable by design)")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only: print the declaration list and stop, "
                         "without proving anything")
    args = ap.parse_args()

    out = Path(args.out) if args.out else (
        ROOT / "results" / "libraries"
        / f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}")
    _check_out_dir(out)
    if args.purge and out.exists():
        shutil.rmtree(out)
        print(f"[purged] {out}")

    from policy.vllm_policy import make_policy
    from backend.compile_verify import verify_proof
    from search.library import build_library, plan_prompt, parse_plan
    from search.library import PLAN_SYSTEM
    from search.proof_dag import apply_pin_renames

    policy = make_policy(args.provider, args.model)

    def llm(system: str, user: str) -> str:
        s = policy.sample_topk(user, k=1, system=system,
                               max_tokens=args.max_tokens)
        return apply_pin_renames(s[0].text if s else "")

    def verify(header: str, body: str) -> dict:
        # A library declaration is compiled as a COMPLETE unit: the header
        # carries imports + everything verified so far + this declaration,
        # and the body is empty. `verify_proof` appends ":= by\n<body>",
        # so an empty body would be a syntax error — compile directly.
        from backend.compile_verify import compile_lean
        src = header if header.endswith("\n") else header + "\n"
        r = compile_lean(src, timeout_s=args.verify_timeout)
        return {"ok": r.ok and not r.used_sorry, "errors": r.errors}

    def trace(kind: str, **kw) -> None:
        bits = " ".join(f"{k}={v}" for k, v in kw.items()
                        if k not in ("names",))
        print(f"[{kind}] {bits}")
        if kw.get("names"):
            for n in kw["names"]:
                print(f"    - {n}")

    if args.dry_run:
        raw = llm(PLAN_SYSTEM,
                  plan_prompt(args.spec, args.imports, args.max_decls))
        plan, err = parse_plan(raw, args.max_decls)
        if plan is None:
            print(f"plan failed: {err}", file=sys.stderr)
            return 1
        print(f"planned {len(plan)} declarations (nothing proved):\n")
        for i, d in enumerate(plan, 1):
            print(f"{i:2d}. [{d['kind']}] {d['name']}")
            print(f"    {d['statement'][:160]}")
            if d["rationale"]:
                print(f"    -- {d['rationale'][:120]}")
        return 0

    res = build_library(
        args.spec, llm_call=llm, verify_fn=verify,
        imports=args.imports, name=args.name,
        max_decls=args.max_decls,
        attempts_per_decl=args.attempts_per_decl,
        trace=trace)

    files = res.emit(out)
    print(f"\nverified {len(res.decls)}/{res.planned} declarations "
          f"in {res.elapsed_s:.0f}s")
    for f in files:
        print(f"  wrote {f}")
    if res.failed:
        print("\nnot verified:")
        for d in res.failed:
            print(f"  - {d.name} ({d.attempts} attempts): "
                  f"{(d.error or '')[:100]}")
    print(f"\ndisposable: rm -rf {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
