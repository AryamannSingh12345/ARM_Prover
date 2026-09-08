"""Merge multiple proof-prior JSONL files into one.

Spec: PART 8 of the Mathlib-prior series.

Combine rule:
  - Group rows by (feature_key, tactic_template, tactic_class, premise).
  - Sum counts across inputs.
  - The merged row's `source` becomes `"merged"` if more than one
    distinct source contributed, else the single contributing source.
  - `tags` is the deduplicated union of contributing tags PLUS one entry
    per distinct contributing source (so downstream consumers can read
    the provenance list off `tags`).
  - `requires_instantiation` is the OR across contributors.
  - `prior_probability` is recomputed by the `ProofPriorIndex` smoothing
    after the merge.

CLI:
  python scripts/merge_proof_priors.py \
    --inputs <a.jsonl> <b.jsonl> ... \
    --out data/proof_prior/combined.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

# Make src/ importable without installing the package.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from search.proof_prior import (  # noqa: E402
    ProofPriorIndex, ProofPriorMove, _MoveRecord,
)


def _iter_input_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def _read_input_sidecar(path: Path) -> dict:
    """Read the `<input>.meta.json` sidecar for an input prior, if any.

    Returns {} when the sidecar is missing or unreadable. The sidecar
    is the authoritative source for whether an input prior was built
    against a holdout subset — without it, merging silently strips the
    clean-holdout flag and downstream eval rows look clean when they
    aren't.
    """
    side_path = path.with_suffix(path.suffix + ".meta.json")
    if not side_path.exists():
        return {}
    try:
        return json.loads(side_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _union_excluded_ids(side_meta: dict) -> set[str]:
    """Pull excluded-theorem IDs out of an input sidecar, accepting
    both legacy (`excluded_subset`) and explicit (`excluded_theorem_ids`)
    field names. Returns the union as a set."""
    out: set[str] = set()
    for field in ("excluded_subset", "excluded_theorem_ids"):
        for sid in (side_meta.get(field) or []):
            if isinstance(sid, str) and sid:
                out.add(sid)
    return out


def merge(input_paths: list[Path], out_path: Path) -> dict:
    """Merge inputs into `out_path`. Returns a stats dict."""
    idx = ProofPriorIndex()
    per_input_rows: dict[str, int] = {}
    # Holdout-metadata union across input sidecars. Spec:
    #   - clean_holdout=true if ANY input has clean_holdout=true.
    #   - excluded_subset / excluded_theorem_ids = sorted union of all
    #     contributing IDs (legacy + new field names both honoured).
    #   - excluded_theorem_count = len(union).
    # The per-input sidecar dicts are also stored under `input_sidecars`
    # so a triage script can see WHICH inputs carried the holdout.
    merged_clean_holdout = False
    merged_excluded_ids: set[str] = set()
    merged_excluded_subset_paths: list[str] = []
    input_sidecars: dict[str, dict] = {}
    for ip in input_paths:
        side_meta = _read_input_sidecar(ip)
        if side_meta:
            input_sidecars[str(ip)] = side_meta
            if bool(side_meta.get("clean_holdout")):
                merged_clean_holdout = True
            merged_excluded_ids |= _union_excluded_ids(side_meta)
            sp = side_meta.get("excluded_subset_path")
            if isinstance(sp, str) and sp and sp not in merged_excluded_subset_paths:
                merged_excluded_subset_paths.append(sp)
    for ip in input_paths:
        n = 0
        for row in _iter_input_rows(ip):
            key = row.get("feature_key")
            template = row.get("tactic_template")
            if not key or not template:
                continue
            bucket = idx._by_key.setdefault(key, {})
            rec = bucket.get(template)
            count = int(row.get("count", 1))
            sources_from_row = set()
            if isinstance(row.get("source"), str) and row["source"]:
                sources_from_row.add(row["source"])
            for t in (row.get("tags") or []):
                if isinstance(t, str):
                    sources_from_row.add(t)
            tags_from_row = tuple(row.get("tags") or [])
            prov_from_row = tuple(row.get("provenance") or [])
            move = ProofPriorMove(
                tactic_template=template,
                tactic_class=row.get("tactic_class", "unknown"),
                premise=row.get("premise"),
                prior_probability=0.0,
                prior_cost=0.0,
                source=row.get("source", "unknown"),
                tags=tags_from_row,
                requires_instantiation=bool(row.get("requires_instantiation", False)),
                provenance=prov_from_row,
            )
            if rec is None:
                bucket[template] = _MoveRecord(move=move, count=count)
                idx._total_by_key[key] = idx._total_by_key.get(key, 0) + count
            else:
                rec.count += count
                idx._total_by_key[key] = idx._total_by_key.get(key, 0) + count
                cur = rec.move
                # Union of tags (preserve order: existing first, new last).
                merged_tags = list(cur.tags)
                for t in tags_from_row:
                    if t not in merged_tags:
                        merged_tags.append(t)
                # Distinct sources contributing to this row.
                contrib_sources = set(merged_tags) | {cur.source, move.source} - {"unknown"}
                primary = "merged" if len(contrib_sources) > 1 else (cur.source or move.source)
                # Provenance union — needed by the clean-eval audit so
                # the merged file still records that target X's trace
                # was a contributor.
                merged_prov = list(cur.provenance)
                for p in move.provenance:
                    if p not in merged_prov:
                        merged_prov.append(p)
                rec.move = ProofPriorMove(
                    tactic_template=cur.tactic_template,
                    tactic_class=cur.tactic_class,
                    premise=cur.premise or move.premise,
                    prior_probability=0.0,
                    prior_cost=0.0,
                    source=primary,
                    tags=tuple(merged_tags),
                    requires_instantiation=cur.requires_instantiation or move.requires_instantiation,
                    provenance=tuple(merged_prov),
                )
            n += 1
        per_input_rows[str(ip)] = n
    idx._renormalise()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx.save(out_path)

    sidecar = out_path.with_suffix(out_path.suffix + ".meta.json")
    sorted_excluded = sorted(merged_excluded_ids)
    meta = {
        "inputs": [str(p) for p in input_paths],
        "rows_per_input": per_input_rows,
        "merged_keys": len(idx._by_key),
        "merged_moves": sum(len(b) for b in idx._by_key.values()),
        # Holdout union across input sidecars. `clean_holdout=true`
        # if any input was built with --exclude-subset / --exclude-
        # theorem-id; downstream clean-eval audits read this slot.
        "clean_holdout": merged_clean_holdout,
        "excluded_theorem_count": len(merged_excluded_ids),
        "excluded_subset": sorted_excluded,
        "excluded_theorem_ids": sorted_excluded,
        "excluded_subset_paths": list(merged_excluded_subset_paths),
        "input_sidecars": input_sidecars,
    }
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        **meta,
        "output": str(out_path),
        "sidecar": str(sidecar),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Input JSONL prior files to merge (in order).")
    ap.add_argument("--out", required=True,
                    help="Merged JSONL output path.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path.resolve() if path.is_absolute() else (repo_root / p).resolve()

    inputs = [_resolve(p) for p in args.inputs]
    for ip in inputs:
        if not ip.exists():
            print(f"[merge-proof-priors] missing input: {ip}", file=sys.stderr)
            return 1
    out_path = _resolve(args.out)
    stats = merge(inputs, out_path)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
