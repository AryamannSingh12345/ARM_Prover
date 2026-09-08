"""Convert ProofNet (Lean 4) and PutnamBench into the harness problem format.

Emits one .lean file per problem, shaped exactly like the miniF2F files the
runners already load: imports + opens at the top, then a single
`theorem <name> ... := by sorry`. Docstrings and line comments are STRIPPED
from the emitted files — `run_dag._THEOREM_RE` anchors on the first literal
`theorem\\s+\\w+` in the file, and informal prose frequently contains the
word "theorem", which would corrupt the extracted header. The informal
statements are preserved in a companion `informal.jsonl` per benchmark.

Inputs (downloaded separately, see --proofnet-jsonl / --putnam-src):
  - ProofNet: DeepSeek-Prover-V1.5 `datasets/proofnet.jsonl` (Lean 4 port).
    Fields: name, split (valid|test), informal_prefix, formal_statement
    (ends with `:=`), header (imports + opens).
  - PutnamBench: `lean4/src/putnam_*.lean` from trishullab/PutnamBench.
    Theorems end `:= sorry`; answer-construction problems carry an
    `abbrev putnam_*_solution : T := sorry` whose real value sits in the
    `--` comment block immediately below — we substitute it when present
    (the standard "solution provided" evaluation mode).

Outputs:
  data/proofnet/Test/*.lean, data/proofnet/{valid,test}.txt, informal.jsonl
  data/putnambench/Test/*.lean, manifest.json, candidates.txt, informal.jsonl
  data/proofnet_dev30.txt   — 30-problem dev slice (valid split, stratified
                              by first opened namespace as a subject proxy)
  data/putnam_slice20.txt   — 20 pure-proving problems stratified by decade

Deterministic: sampling uses a fixed seed; rerunning is idempotent.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Must stay in sync with run_dag/run_minif2f _THEOREM_RE.
_THEOREM_RE = re.compile(r"(theorem\s+\w+.*?):=\s*by\s+sorry\s*$", re.S)

_DOCSTRING_RE = re.compile(r"/--.*?-/\s*\n?", re.S)
_LINE_COMMENT_RE = re.compile(r"^[ \t]*--[^\n]*\n", re.M)
_FINAL_SORRY_RE = re.compile(r":=\s*sorry\s*\Z")
# `abbrev`/`noncomputable abbrev` solution stub + the comment block holding
# the intended value on the following line(s).
_SOLUTION_STUB_RE = re.compile(
    r"^(?P<decl>(?:noncomputable\s+)?abbrev\s+\w+_solution[^\n]*?):=\s*sorry\s*\n"
    r"(?P<comment>(?:^--[^\n]*\n)+)?",
    re.M,
)


def _check_harness_regex(name: str, text: str, problems: list[str]) -> None:
    # Validate against the actual generalized loader (eval.loader), not a
    # regex copy: whatever the loader can't parse is a real problem.
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from eval.loader import extract_statement
    if extract_statement(text) is None:
        problems.append(f"{name}: eval.loader cannot extract a statement")


def convert_proofnet(jsonl_path: Path, out_root: Path) -> dict[str, list[str]]:
    """Returns {split: [filename ids]} and writes files + manifests."""
    test_dir = out_root / "Test"
    test_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    seen: dict[str, int] = {}
    splits: dict[str, list[str]] = defaultdict(list)
    informal: list[dict] = []
    problems: list[str] = []
    excluded: list[str] = []

    for r in rows:
        base = r["name"]
        n = seen.get(base, 0)
        seen[base] = n + 1
        # ProofNet names collide across source textbooks (no book field in
        # this export); keep the theorem name, disambiguate the file id.
        fid = base if n == 0 else f"{base}__dup{n}"

        stmt = _LINE_COMMENT_RE.sub("", r["formal_statement"]).rstrip()
        if stmt.endswith(":="):
            stmt = stmt + " by sorry"
        elif re.search(r":=\s*by\s+sorry\s*\Z", stmt):
            pass
        elif _FINAL_SORRY_RE.search(stmt):
            stmt = _FINAL_SORRY_RE.sub(":= by sorry", stmt)
        else:
            problems.append(f"{fid}: formal_statement has unexpected tail {stmt[-40:]!r}")
            continue

        header = r["header"].strip()
        # Docstrings (informal statements) are KEPT in the emitted file —
        # the generalized loader strips comments safely for parsing and
        # re-attaches the docstring so the model sees the informal text.
        informal_prefix = (r.get("informal_prefix") or "").strip()
        doc = f"{informal_prefix}\n" if informal_prefix else ""
        text = f"{header}\n\n{doc}{stmt}\n"
        (test_dir / f"{fid}.lean").write_text(text, encoding="utf-8")
        informal.append({"id": fid, "informal": r.get("informal_prefix", "")})
        # def/instance construction problems are loadable now (generalized
        # loader) — everything the loader parses goes into the split.
        _check_harness_regex(fid, text, problems)
        if problems and problems[-1].startswith(f"{fid}:"):
            excluded.append(fid)
        else:
            splits[r["split"]].append(fid)

    for split, ids in splits.items():
        (out_root / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    if excluded:
        (out_root / "excluded_def_statements.txt").write_text(
            "\n".join(excluded) + "\n", encoding="utf-8")
        print(f"[proofnet] {len(excluded)} def-statement problems excluded "
              f"from split manifests (see excluded_def_statements.txt)")
    with (out_root / "informal.jsonl").open("w", encoding="utf-8") as fh:
        for row in informal:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if problems:
        print(f"[proofnet] {len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
    print(f"[proofnet] wrote {sum(len(v) for v in splits.values())} files "
          f"({ {k: len(v) for k, v in splits.items()} })")
    return dict(splits)


def convert_putnam(src_dir: Path, out_root: Path) -> dict[str, dict]:
    """Returns manifest {id: {...}} and writes files + manifests."""
    test_dir = out_root / "Test"
    test_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    informal: list[dict] = []
    problems: list[str] = []

    for src in sorted(src_dir.glob("putnam_*.lean")):
        fid = src.stem
        text = src.read_text(encoding="utf-8")

        doc = _DOCSTRING_RE.findall(text)
        informal.append({"id": fid, "informal": "".join(doc)})
        # Docstrings stay in the file — the generalized loader handles
        # comments safely and feeds the informal statement to the model.

        has_stub = False
        substituted = True

        def _sub(m: re.Match) -> str:
            nonlocal has_stub, substituted
            has_stub = True
            comment = m.group("comment")
            if not comment:
                substituted = False
                return m.group(0)  # keep the sorry stub; excluded from candidates
            value = "\n".join(
                line[2:].strip() for line in comment.strip().splitlines()
            ).strip()
            return f"{m.group('decl')}:= {value}\n"

        text = _SOLUTION_STUB_RE.sub(_sub, text)

        stripped = text.rstrip()
        if _FINAL_SORRY_RE.search(stripped):
            text = _FINAL_SORRY_RE.sub(":= by sorry", stripped) + "\n"
        elif re.search(r":=\s*by\s+sorry\s*\Z", stripped):
            text = stripped + "\n"
        else:
            problems.append(f"{fid}: unexpected file tail {stripped[-40:]!r}")
            continue

        _check_harness_regex(fid, text, problems)
        (test_dir / f"{fid}.lean").write_text(text, encoding="utf-8")
        manifest[fid] = {
            "solution_abbrev": has_stub,
            "substituted": substituted if has_stub else None,
            "year": int(fid.split("_")[1]),
        }

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    candidates = [k for k, v in manifest.items() if not v["solution_abbrev"]]
    (out_root / "candidates.txt").write_text("\n".join(candidates) + "\n", encoding="utf-8")
    with (out_root / "informal.jsonl").open("w", encoding="utf-8") as fh:
        for row in informal:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if problems:
        print(f"[putnam] {len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
    n_stub = sum(1 for v in manifest.values() if v["solution_abbrev"])
    n_sub = sum(1 for v in manifest.values() if v.get("substituted"))
    print(f"[putnam] wrote {len(manifest)} files; {n_stub} with solution abbrev "
          f"({n_sub} substituted), {len(candidates)} pure-proving candidates")
    return manifest


def _proportional_sample(buckets: dict, k: int, seed: int) -> list[str]:
    """Proportional stratified sample (largest-remainder, >=1 per stratum).

    Quota per stratum ∝ stratum size, so the slice mirrors the full pool's
    composition — solve rates on the slice estimate solve rates on the pool.
    Every stratum keeps at least one problem so no subject/era is invisible.
    """
    rng = random.Random(seed)
    total = sum(len(v) for v in buckets.values())
    exact = {b: k * len(v) / total for b, v in buckets.items()}
    quota = {b: max(1, int(q)) for b, q in exact.items()}
    # largest-remainder top-up / trim to hit k exactly
    while sum(quota.values()) < k:
        b = max(buckets, key=lambda b: (exact[b] - quota[b], len(buckets[b])))
        quota[b] += 1
    while sum(quota.values()) > k:
        b = max((b for b in buckets if quota[b] > 1),
                key=lambda b: quota[b] - exact[b])
        quota[b] -= 1
    picked: list[str] = []
    for b in sorted(buckets):
        ids = sorted(buckets[b])
        rng.shuffle(ids)
        picked.extend(ids[:min(quota[b], len(ids))])
    return sorted(picked)


def slice_proofnet(out_root: Path, valid_ids: list[str], k: int, seed: int) -> list[str]:
    """Proportional stratified sample of the valid split, by first opened
    namespace (subject proxy)."""
    buckets: dict[str, list[str]] = defaultdict(list)
    for fid in valid_ids:
        text = (out_root / "Test" / f"{fid}.lean").read_text(encoding="utf-8")
        m = re.search(r"^open\s+(\w+)", text, re.M)
        buckets[m.group(1) if m else "_none"].append(fid)
    return _proportional_sample(buckets, k, seed)


def slice_putnam(manifest: dict[str, dict], k: int, seed: int) -> list[str]:
    """Proportional stratified sample of pure-proving problems, by decade."""
    buckets: dict[int, list[str]] = defaultdict(list)
    for fid, meta in manifest.items():
        if not meta["solution_abbrev"]:
            buckets[meta["year"] // 10].append(fid)
    return _proportional_sample({str(k_): v for k_, v in buckets.items()}, k, seed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--proofnet-jsonl", type=Path, required=True)
    ap.add_argument("--putnam-src", type=Path, required=True,
                    help="PutnamBench lean4/src directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proofnet-slice", type=int, default=30)
    ap.add_argument("--putnam-slice", type=int, default=20)
    args = ap.parse_args()

    pn_root = ROOT / "data" / "proofnet"
    pb_root = ROOT / "data" / "putnambench"

    splits = convert_proofnet(args.proofnet_jsonl, pn_root)
    manifest = convert_putnam(args.putnam_src, pb_root)

    dev = slice_proofnet(pn_root, splits.get("valid", []), args.proofnet_slice, args.seed)
    (ROOT / "data" / "proofnet_dev30.txt").write_text("\n".join(dev) + "\n", encoding="utf-8")
    print(f"[slice] data/proofnet_dev30.txt: {len(dev)} problems")

    pslice = slice_putnam(manifest, args.putnam_slice, args.seed)
    (ROOT / "data" / "putnam_slice20.txt").write_text("\n".join(pslice) + "\n", encoding="utf-8")
    print(f"[slice] data/putnam_slice20.txt: {len(pslice)} problems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
