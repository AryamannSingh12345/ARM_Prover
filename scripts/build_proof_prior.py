"""Mine a proof-prior from solved JSONL logs.

Spec: PART 4 of the proof-prior series; holdout extensions per PART C
of the proof-prior fix-up pass.

Reads every `*.jsonl` under `--results` (recursive), keeps rows where
`outcome == "solved"`, and walks `proof_tactics` to extract:

  - features at step i = features of the theorem header with
    previous_tactic_class = classify_tactic(proof_tactics[i-1]) or None.
    (We do not have per-step Lean states in JSONL, only the theorem header.)
  - one ProofPriorMove per tactic:
      template = tactic text (we keep exact tactics for v1; future versions
                 can normalise local hypothesis names to slots)
      class    = classify_tactic(tactic)
      premise  = first dotted citation from extract_used_premises (RHS of
                 `:=` is prioritised, so `have h : T := Nat.gcd_mul_lcm n k`
                 records `Nat.gcd_mul_lcm`, not `Nat.gcd`)
      source   = "solved_jsonl"

Aggregated counts/probabilities are emitted via ProofPriorIndex.save().

Theorem IDs are used ONLY to:
  - locate the per-problem source file for header feature extraction, AND
  - apply --exclude-subset / --exclude-theorem-id to skip rows whose id
    is in the held-out set.
They are NOT stored as features and they are NOT stored in the output
rows.

Holdout mode
------------
A clean evaluation run MUST be able to build a prior that excludes the
target problems. Pass `--exclude-subset path/to/subset.txt` (one id per
line, optionally with `#` comments) and/or one or more
`--exclude-theorem-id ID` flags. The output sidecar `<out>.meta.json`
records the holdout state so a later eval row can be audited:

    {
      "clean_holdout": true,
      "excluded_theorem_count": 12,
      "excluded_subset": ["mathd_numbertheory_100", ...]
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable without installing the package.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from search.proof_prior import ProofPriorIndex, ProofPriorMove, ProofStateFeatures
from search.state_features import extract_state_features
from search.tactic_classify import (
    classify_tactic, extract_used_premises, tactic_shape,
)


def _iter_jsonl_rows(results_dir: Path):
    """Yield (row, src_path) for EVERY jsonl row, solved or not.

    Caller filters by `outcome` — moving the filter out lets `build()`
    distinguish `total_rows_seen` from `solved_rows_seen`.
    """
    for jsonl in sorted(results_dir.rglob("*.jsonl")):
        try:
            with jsonl.open(encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    yield row, jsonl
        except OSError:
            continue


def _find_theorem_source(data_root: Path, problem_id: str) -> str | None:
    """Best-effort lookup of `data/miniF2F*/MiniF2F/*/<problem_id>.lean`."""
    for root in (data_root / "miniF2F", data_root / "miniF2F_v2"):
        if not root.exists():
            continue
        for candidate in root.rglob(f"{problem_id}.lean"):
            try:
                return candidate.read_text(encoding="utf-8")
            except OSError:
                continue
    return None


def _header_query_for_features(source_text: str) -> str:
    """Cheap extractor: drop imports/options/opens, keep through `:= by`."""
    out_lines: list[str] = []
    for line in source_text.splitlines():
        s = line.strip()
        if not s or s.startswith("--"):
            continue
        if s.startswith(("import ", "set_option", "open ", "noncomputable")):
            continue
        out_lines.append(line)
    text = "\n".join(out_lines)
    idx = text.find(":= by")
    if idx != -1:
        text = text[:idx]
    return text.strip()


def _features_for_problem(
    *,
    data_root: Path,
    problem_id: str,
    previous_tactic: str | None,
) -> ProofStateFeatures:
    src = _find_theorem_source(data_root, problem_id)
    header_text = _header_query_for_features(src) if src else ""
    proof_prefix = [previous_tactic] if previous_tactic else []
    return extract_state_features(header_text, proof_prefix, retrieved_premises=None)


def _load_subset_ids(path: Path) -> set[str]:
    """Load theorem IDs from a subset file (one id per line, '#' comments)."""
    if not path.exists():
        raise FileNotFoundError(f"exclude-subset file not found: {path}")
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s)
    return out


def build(
    results_dir: Path,
    data_root: Path,
    out_path: Path,
    *,
    excluded_ids: set[str] | None = None,
    excluded_subset_path: Path | str | None = None,
) -> dict:
    """Mine a prior. Returns a stats dict; writes the JSONL + sidecar.

    `excluded_subset_path` is recorded verbatim in the sidecar so the
    merge step (and downstream clean-eval audits) can show provenance
    for the holdout list without re-parsing the subset file.
    """
    idx = ProofPriorIndex()
    excluded_ids = set(excluded_ids or ())
    n_total_rows = 0
    n_solved_rows = 0
    n_excluded = 0
    n_moves = 0
    seen_problems: set[str] = set()
    for row, _src in _iter_jsonl_rows(results_dir):
        n_total_rows += 1
        if row.get("outcome") != "solved":
            continue
        n_solved_rows += 1
        problem_id = row.get("id") or ""
        if problem_id and problem_id in excluded_ids:
            n_excluded += 1
            continue
        tactics = row.get("proof_tactics") or []
        if not tactics:
            continue
        seen_problems.add(problem_id)
        previous_tactic: str | None = None
        for tactic in tactics:
            tactic = (tactic or "").strip()
            if not tactic:
                continue
            features = _features_for_problem(
                data_root=data_root,
                problem_id=problem_id,
                previous_tactic=previous_tactic,
            )
            premises = extract_used_premises(tactic)
            # Per-row provenance: record the source problem_id so the
            # clean-eval audit can detect target-trace leakage. This is
            # METADATA only — the search never reads provenance for
            # scoring or feature keying (see ProofPriorIndex docs).
            provenance: tuple[str, ...] = (
                (problem_id,) if problem_id else ()
            )
            move = ProofPriorMove(
                tactic_template=tactic,
                tactic_class=classify_tactic(tactic),
                premise=premises[0] if premises else None,
                source="solved_jsonl",
                tags=("mined",),
                requires_instantiation=False,
                provenance=provenance,
            )
            idx.add_observation(features, move, result="solved")
            # PART A (continuation fix-up): also record the
            # `previous_tactic -> tactic` transition so the live search
            # can look up "what comes after X" once X has been applied.
            # Skipped on the first tactic (no predecessor).
            if previous_tactic:
                seq_move = ProofPriorMove(
                    tactic_template=tactic,
                    tactic_class=classify_tactic(tactic),
                    premise=premises[0] if premises else None,
                    source="sequence_solved_jsonl",
                    tags=("mined", "sequence"),
                    requires_instantiation=False,
                    provenance=provenance,
                )
                idx.add_sequence_observation(
                    previous_tactic, seq_move, result="solved",
                )
                # PART D: also record the THEOREM-AGNOSTIC abstract
                # transition `class(prev) -> shape(curr)`. The runtime
                # generic instantiator turns this hint into concrete
                # candidates from the live goal_text.
                prev_class_ab = classify_tactic(previous_tactic)
                curr_shape = tactic_shape(tactic)
                idx.add_abstract_observation(
                    prev_class_ab, curr_shape,
                    next_class=classify_tactic(tactic),
                    source="abstract_continuation_prior",
                    tags=("solved_jsonl_holdout",),
                    provenance=provenance,
                    result="solved",
                )
            n_moves += 1
            previous_tactic = tactic
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx.save(out_path)

    sidecar = out_path.with_suffix(out_path.suffix + ".meta.json")
    # `excluded_subset` is retained as the list-of-IDs slot for
    # backward compatibility (existing tests + downstream readers).
    # `excluded_theorem_ids` is an explicit alias so the field name in
    # the sidecar reads as what it actually contains. The PATH the user
    # passed is recorded separately as `excluded_subset_path` so the
    # audit trail does not conflate the two.
    sorted_ids = sorted(excluded_ids)
    meta = {
        "clean_holdout": bool(excluded_ids),
        "excluded_theorem_count": len(excluded_ids),
        "excluded_subset": sorted_ids,
        "excluded_theorem_ids": sorted_ids,
        "excluded_subset_path": (
            str(excluded_subset_path) if excluded_subset_path else None
        ),
    }
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "total_rows_seen": n_total_rows,
        "solved_rows_seen": n_solved_rows,
        "rows_excluded_by_theorem_id": n_excluded,
        "unique_theorems_indexed": len(seen_problems),
        "moves_recorded": n_moves,
        "output": str(out_path),
        "sidecar": str(sidecar),
        "clean_holdout": meta["clean_holdout"],
        "excluded_theorem_count": meta["excluded_theorem_count"],
        "excluded_subset_path": meta["excluded_subset_path"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results",
                    help="Directory to walk for *.jsonl logs.")
    ap.add_argument("--data", default="data",
                    help="miniF2F data root (used only to look up per-"
                         "problem source files for header feature "
                         "extraction; theorem IDs are NOT mined into "
                         "the prior).")
    ap.add_argument("--out",
                    default="data/proof_prior/proof_prior_moves.jsonl")
    # --- Holdout-safe build flags (clean evaluation hygiene) ---
    ap.add_argument("--exclude-subset", default=None,
                    help="Path to a subset file (one theorem id per line, "
                         "'#' comments). Solved rows whose `id` is in this "
                         "set are skipped — required for clean evaluation "
                         "on the same subset (no answer-leakage).")
    ap.add_argument("--exclude-theorem-id", action="append", default=[],
                    metavar="ID",
                    help="Skip solved rows whose `id` matches this exactly. "
                         "Repeatable. Combined (set-union) with "
                         "--exclude-subset.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = (repo_root / args.results).resolve()
    data_root = (repo_root / args.data).resolve()
    out_path = (repo_root / args.out).resolve()
    if not results_dir.exists():
        print(f"[build_proof_prior] no such results dir: {results_dir}",
              file=sys.stderr)
        return 1

    excluded_ids: set[str] = set()
    excluded_subset_path: Path | None = None
    if args.exclude_subset:
        excluded_subset_path = (
            Path(args.exclude_subset).resolve()
            if Path(args.exclude_subset).is_absolute()
            else (repo_root / args.exclude_subset).resolve()
        )
        excluded_ids |= _load_subset_ids(excluded_subset_path)
    for tid in args.exclude_theorem_id:
        tid = (tid or "").strip()
        if tid:
            excluded_ids.add(tid)

    stats = build(
        results_dir, data_root, out_path,
        excluded_ids=excluded_ids,
        excluded_subset_path=excluded_subset_path,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
