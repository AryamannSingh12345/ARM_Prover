"""Provenance-carrying lemma bank + eval-split/chronology filter (Point 9).

The ARM loop appends kernel-verified *invented* lemmas to an on-disk bank
(`results/invented_lemmas.lean`). Historically that was a bare ``.lean``
file deduplicated by declared **name** only. Name-dedup cannot prevent:

* semantically duplicate lemmas stored under different names, and — the
  real hazard — a lemma **derived from an evaluation target** later
  helping to solve *other* evaluation problems. A learned library that is
  not split-aware silently becomes benchmark leakage.

This module keeps the human-readable ``.lean`` append (still
splice-compatible and still name-deduped by the caller) and adds a
structured JSONL **sidecar**: one record per stored declaration carrying

    statement_hash, normalized_statement, name, proof, imports,
    source_problem, source_split, run_id, model,
    verified_mathlib_commit, ts (UTC, for chronology), tag,
    allowed_for_eval

``allowed_for_eval`` DEFAULTS TO FALSE — nothing in the bank is
eval-safe until it is explicitly promoted. `load_bank` is the filter a
*clean* evaluation MUST apply: exclude the split under test, exclude
anything learned at/after the eval cutoff (chronology), and optionally
require ``allowed_for_eval``.

Nothing here is load-bearing for the current write-only bank; it makes the
bank *safe to consult* later. No Lean, no network.

NOTE (string-op caveat, Point 7): `normalize_statement` strips the
declaration name and collapses whitespace/comments, so two identically
*shaped* lemmas under different names collapse — but binder-name and
notation differences do NOT (that needs Lean elaboration, a later phase).
The hash is a cheap prefilter, not a semantic-equality oracle.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Declaration head: keyword + name. Kept independent of proof_dag's
# `_DECL_NAME_RE` on purpose — proof_dag imports THIS module, not the
# other way round, so there is no import cycle.
_DECL_HEAD_RE = re.compile(
    r"^\s*(?:private\s+)?(?:noncomputable\s+)?"
    r"(?:lemma|theorem|def|abbrev)\s+([A-Za-z_][A-Za-z0-9_'.₀-₉]*)",
)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/-.*?-/", re.S)
_WS_RE = re.compile(r"\s+")

# Known dataset splits (open vocabulary; `unknown` is the safe default).
SPLITS = ("dev", "valid", "test", "putnam", "custom", "unknown")


def split_decl(decl: str) -> tuple[str | None, str, str]:
    """Split one Lean declaration into (name, statement, proof).

    `statement` is everything up to (but excluding) the first top-level
    ``:=`` — i.e. the signature (binders + goal). `proof` is the rest.
    Splitting on the first ``:=`` is a prototype heuristic (a ``:=`` can
    appear inside a statement, e.g. a structure literal); adequate for a
    telemetry hash, not a parser.
    """
    m = _DECL_HEAD_RE.match(decl)
    name = m.group(1) if m else None
    idx = decl.find(":=")
    if idx == -1:
        return name, decl.strip(), ""
    return name, decl[:idx].strip(), decl[idx + 2:].strip()


def normalize_statement(statement: str) -> str:
    """Normalize a declaration signature for hashing/comparison: drop the
    leading `keyword name`, strip comments, collapse whitespace. Removing
    the name is what lets two identically-shaped lemmas under different
    names collapse to one hash."""
    s = _BLOCK_COMMENT_RE.sub(" ", statement)
    s = _LINE_COMMENT_RE.sub(" ", s)
    # Strip the "lemma/theorem/def/abbrev NAME" prefix so the name does
    # not participate in the identity.
    s = _DECL_HEAD_RE.sub("", s, count=1)
    return _WS_RE.sub(" ", s).strip()


def statement_hash(normalized: str) -> str:
    """Stable content hash of a normalized statement."""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def mathlib_commit_from_manifest(lean_dir: str | Path) -> str | None:
    """Read the pinned Mathlib revision from ``lean/lake-manifest.json``.
    Returns None if the manifest is missing/unreadable — provenance stays
    honest rather than guessing a commit."""
    try:
        manifest = Path(lean_dir) / "lake-manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for pkg in data.get("packages", []):
            if pkg.get("name") == "mathlib":
                return pkg.get("rev")
    except Exception:
        return None
    return None


@dataclass(slots=True)
class LemmaRecord:
    statement_hash: str
    normalized_statement: str
    name: str | None
    proof: str
    imports: list[str] = field(default_factory=list)
    source_problem: str | None = None
    source_split: str = "unknown"
    run_id: str | None = None
    model: str | None = None
    verified_mathlib_commit: str | None = None
    ts: str = ""
    tag: str = ""
    allowed_for_eval: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LemmaRecord":
        return cls(
            statement_hash=d["statement_hash"],
            normalized_statement=d.get("normalized_statement", ""),
            name=d.get("name"),
            proof=d.get("proof", ""),
            imports=list(d.get("imports") or []),
            source_problem=d.get("source_problem"),
            source_split=d.get("source_split", "unknown"),
            run_id=d.get("run_id"),
            model=d.get("model"),
            verified_mathlib_commit=d.get("verified_mathlib_commit"),
            ts=d.get("ts", ""),
            tag=d.get("tag", ""),
            allowed_for_eval=bool(d.get("allowed_for_eval", False)),
        )


def sidecar_path(lean_path: str | Path) -> Path:
    """The JSONL provenance sidecar for a ``.lean`` bank file:
    ``results/invented_lemmas.lean`` -> ``results/invented_lemmas.jsonl``."""
    p = Path(lean_path)
    return p.with_suffix(".jsonl")


def build_records(
    decls: Iterable[str],
    *,
    tag: str = "",
    provenance: dict[str, Any] | None = None,
) -> list[LemmaRecord]:
    """Build one provenance record per declaration. `provenance` supplies
    the run/problem-level fields; anything absent stays null and
    `allowed_for_eval` stays False (conservative)."""
    prov = provenance or {}
    ts = datetime.now(timezone.utc).isoformat()
    out: list[LemmaRecord] = []
    for decl in decls:
        if not decl or not decl.strip():
            continue
        name, statement, proof = split_decl(decl)
        norm = normalize_statement(statement)
        out.append(LemmaRecord(
            statement_hash=statement_hash(norm),
            normalized_statement=norm,
            name=name,
            proof=proof,
            imports=list(prov.get("imports") or []),
            source_problem=prov.get("source_problem"),
            source_split=prov.get("source_split", "unknown"),
            run_id=prov.get("run_id"),
            model=prov.get("model"),
            verified_mathlib_commit=prov.get("verified_mathlib_commit"),
            ts=ts,
            tag=tag,
            allowed_for_eval=bool(prov.get("allowed_for_eval", False)),
        ))
    return out


def _read_records(jsonl_path: str | Path) -> list[LemmaRecord]:
    p = Path(jsonl_path)
    if not p.exists():
        return []
    recs: list[LemmaRecord] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(LemmaRecord.from_dict(json.loads(line)))
        except Exception:
            continue
    return recs


def append_records(jsonl_path: str | Path, records: list[LemmaRecord]) -> int:
    """Append records to the JSONL sidecar, skipping any whose
    `statement_hash` already appears in the file (semantic dedup — the
    improvement over name-only dedup). Returns the number written. All
    failures are swallowed (the bank is telemetry)."""
    try:
        existing = {r.statement_hash for r in _read_records(jsonl_path)}
        fresh = []
        for r in records:
            if r.statement_hash in existing:
                continue
            fresh.append(r)
            existing.add(r.statement_hash)
        if fresh:
            with Path(jsonl_path).open("a", encoding="utf-8") as fh:
                for r in fresh:
                    fh.write(json.dumps(r.to_dict(),
                                        ensure_ascii=False) + "\n")
        return len(fresh)
    except Exception:
        return 0


def load_bank(
    jsonl_path: str | Path,
    *,
    exclude_splits: Iterable[str] = (),
    allow_splits: Iterable[str] | None = None,
    before_ts: str | None = None,
    require_allowed: bool = False,
) -> list[LemmaRecord]:
    """Load the bank with the guards a clean evaluation MUST apply.

    * ``exclude_splits`` — drop records whose ``source_split`` is in this
      set (e.g. the split under test).
    * ``allow_splits`` — if given, keep ONLY these splits.
    * ``before_ts`` — chronology guard: keep only records with
      ``ts < before_ts`` (ISO8601), so a lemma learned during the eval
      cannot help a later eval problem in the same run.
    * ``require_allowed`` — keep only records explicitly promoted with
      ``allowed_for_eval = True``.

    Records are deduplicated by ``statement_hash`` keeping the earliest
    ``ts`` (the first time the lemma was learned)."""
    excl = set(exclude_splits)
    allow = set(allow_splits) if allow_splits is not None else None
    recs = _read_records(jsonl_path)
    kept: dict[str, LemmaRecord] = {}
    for r in recs:
        if r.source_split in excl:
            continue
        if allow is not None and r.source_split not in allow:
            continue
        if require_allowed and not r.allowed_for_eval:
            continue
        if before_ts is not None and not (r.ts and r.ts < before_ts):
            continue
        prev = kept.get(r.statement_hash)
        if prev is None or (r.ts and prev.ts and r.ts < prev.ts):
            kept[r.statement_hash] = r
    return list(kept.values())
