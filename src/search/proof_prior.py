"""Proof-prior data structures.

This is the JSONL-backed first cut. No neural model. The on-disk format is
one JSON object per line under `data/proof_prior/proof_prior_moves.jsonl`,
written by `scripts/build_proof_prior.py` and consumed at search time.

The index supports a coarse-fallback lookup: an exact feature_key is tried
first, then progressively coarser keys (drop previous_tactic_class, drop
constants, drop hypothesis shapes) so a never-before-seen state still
returns *something* when its namespace/symbols/target_shape look familiar.

Theorem IDs are deliberately NOT part of any feature key. The point of the
prior is to generalise across problems, not to memorise the training set.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ProofStateFeatures:
    """Shape of a proof state.

    All fields are tuples (hashable) so the bag can be a dict key. The fields
    are intentionally coarse — we want two unrelated theorems with the same
    shape to share statistics. See state_features.extract_state_features.
    """
    symbols: tuple[str, ...] = ()
    namespaces: tuple[str, ...] = ()
    target_shape: str = "unknown"
    hypothesis_shapes: tuple[str, ...] = ()
    constants: tuple[str, ...] = ()
    previous_tactic_class: str | None = None
    retrieved_premises: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "symbols": list(self.symbols),
            "namespaces": list(self.namespaces),
            "target_shape": self.target_shape,
            "hypothesis_shapes": list(self.hypothesis_shapes),
            "constants": list(self.constants),
            "previous_tactic_class": self.previous_tactic_class,
            "retrieved_premises": list(self.retrieved_premises),
        }


def _digest(parts: Iterable[str]) -> str:
    h = hashlib.sha1(usedforsecurity=False)
    for p in parts:
        h.update(b"|")
        h.update(p.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def feature_key(features: ProofStateFeatures, *, level: int = 0) -> str:
    """Stable digest of a feature bag at a coarseness `level`.

    level 0: namespaces + symbols + target_shape + hypothesis_shapes +
             constants + previous_tactic_class
    level 1: drop previous_tactic_class
    level 2: drop constants
    level 3: drop hypothesis_shapes
    level 4: namespaces + symbols + target_shape only

    Retrieved premises are NEVER part of the key — they shift per state and
    would atomise statistics.
    """
    syms = ",".join(sorted(features.symbols))
    ns = ",".join(sorted(features.namespaces))
    tgt = features.target_shape
    hyps = ",".join(sorted(features.hypothesis_shapes))
    consts = ",".join(sorted(features.constants))
    prev = features.previous_tactic_class or ""
    parts = [f"NS={ns}", f"SYM={syms}", f"TGT={tgt}"]
    if level <= 2:
        parts.append(f"HYP={hyps}")
    if level <= 1:
        parts.append(f"CONST={consts}")
    if level <= 0:
        parts.append(f"PREV={prev}")
    return _digest(parts)


@dataclass(slots=True)
class ProofPriorMove:
    """A move the prior suggests in a given state shape.

    The `provenance` slot records the theorem / declaration IDs whose
    proofs contributed observations to this move. It is METADATA only —
    no scoring or feature key uses it. The runtime audit consults it in
    clean-evaluation mode to assert that the target problem's trace was
    excluded from the prior."""
    tactic_template: str
    tactic_class: str
    premise: str | None = None
    prior_probability: float = 0.0
    prior_cost: float = 0.0
    source: str = "unknown"
    tags: tuple[str, ...] = ()
    requires_instantiation: bool = False
    provenance: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        d["provenance"] = list(self.provenance)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProofPriorMove":
        return cls(
            tactic_template=d["tactic_template"],
            tactic_class=d.get("tactic_class", "unknown"),
            premise=d.get("premise"),
            prior_probability=float(d.get("prior_probability", 0.0)),
            prior_cost=float(d.get("prior_cost", 0.0)),
            source=d.get("source", "unknown"),
            tags=tuple(d.get("tags", []) or []),
            requires_instantiation=bool(d.get("requires_instantiation", False)),
            provenance=tuple(d.get("provenance", []) or []),
        )


@dataclass(slots=True)
class _MoveRecord:
    """Internal aggregation row: one (feature_key, tactic_template) bucket."""
    move: ProofPriorMove
    count: int = 0


class ProofPriorIndex:
    """JSONL-backed proof-prior index.

    Storage layout: one JSON object per line. Each object has:
        feature_key: str
        feature_levels: list[str]   # coarse-fallback keys this row contributes to
        tactic_template: str
        tactic_class: str
        premise: str | None
        used_premises: list[str]
        count: int
        prior_probability: float
        source: str
        tags: list[str]
        requires_instantiation: bool

    On load, rows are bucketed under every `feature_levels` entry so
    `suggest()` can fall back to coarser keys without re-walking the file.
    """

    EPSILON: float = 1e-3  # smoothing for unseen-but-allowed moves

    def __init__(self) -> None:
        # feature_key -> tactic_template -> _MoveRecord
        self._by_key: dict[str, dict[str, _MoveRecord]] = {}
        # tracks distinct moves per key for probability normalisation
        self._total_by_key: dict[str, int] = {}

    # ---- persistence -------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "ProofPriorIndex":
        idx = cls()
        p = Path(path)
        if not p.exists():
            return idx
        with p.open(encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                idx._ingest_row(row)
        idx._renormalise()
        return idx

    def save(self, path: Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Emit one row per (level0_key, tactic_template). The fallback keys
        # are recomputed at load by `feature_key(features, level=L)` for the
        # caller — but we still record them so a JSONL reader can index
        # without re-deriving features.
        rows: list[dict] = []
        emitted: set[tuple[str, str]] = set()
        for key, bucket in self._by_key.items():
            for template, rec in bucket.items():
                k = (key, template)
                if k in emitted:
                    continue
                emitted.add(k)
                m = rec.move
                rows.append({
                    "feature_key": key,
                    "tactic_template": m.tactic_template,
                    "tactic_class": m.tactic_class,
                    "premise": m.premise,
                    "count": rec.count,
                    "prior_probability": m.prior_probability,
                    "source": m.source,
                    "tags": list(m.tags),
                    "requires_instantiation": m.requires_instantiation,
                    "provenance": list(m.provenance),
                })
        with p.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- sequence-prior keys (PART A of the continuation fix-up) ----------
    #
    # Sequence rows live in the same `_by_key` store under a SEQ-prefixed
    # synthetic key derived from the previous tactic template. Feature-
    # key digests never collide with these because feature_key emits a
    # bare 16-hex-char digest with no prefix.
    @staticmethod
    def sequence_key(prev_tactic: str) -> str:
        h = hashlib.sha1(usedforsecurity=False)
        h.update(b"SEQ|")
        h.update((prev_tactic or "").encode("utf-8", "replace"))
        return "SEQ:" + h.hexdigest()[:16]

    # Abstract continuation key: the previous tactic's CLASS only — no
    # template, no shape, no theorem identity. Stored values are
    # next-tactic SHAPES (e.g. "rw_at_hyp", "arithmetic_close"), used
    # at runtime to emit state-instantiated tactics.
    @staticmethod
    def abstract_key(prev_class: str) -> str:
        return "ABS:" + (prev_class or "unknown")

    # ---- read API ----------------------------------------------------------

    def suggest(self, features: ProofStateFeatures, top_k: int = 20) -> list[ProofPriorMove]:
        """Return up to `top_k` moves for this feature bag.

        Tries exact key first, then coarser keys. Deduplicates by
        tactic_template across levels (exact-key probabilities win)."""
        seen: dict[str, ProofPriorMove] = {}
        for level in range(0, 5):
            key = feature_key(features, level=level)
            bucket = self._by_key.get(key)
            if not bucket:
                continue
            for template, rec in bucket.items():
                if template in seen:
                    continue
                seen[template] = rec.move
            if len(seen) >= top_k:
                break
        ranked = sorted(
            seen.values(),
            key=lambda m: (-m.prior_probability, m.prior_cost, m.tactic_template),
        )
        return ranked[:top_k]

    def audit_target_provenance(self, target_id: str) -> dict:
        """Return a leakage audit for `target_id`. Used in clean-eval mode
        before search begins. The returned dict has:
          rows_with_target:  total rows whose provenance contains target_id
          rows_in_sequence:  subset of the above that live under a SEQ:* key
                              (mid-proof continuations — the strongest leak)
          rows_in_feature:   subset under feature-keyed slots
          sources_used:      distinct source names of leaking rows
          sample_templates:  up to 5 leaking tactic_templates for the log
        Empty / zero result means no leak detected for that target.
        """
        if not target_id:
            return {"rows_with_target": 0, "rows_in_sequence": 0,
                    "rows_in_feature": 0,
                    "sources_used": [], "sample_templates": []}
        n_total = n_seq = n_feat = 0
        sources: list[str] = []
        samples: list[str] = []
        for key, bucket in self._by_key.items():
            for tpl, rec in bucket.items():
                if target_id not in rec.move.provenance:
                    continue
                n_total += 1
                if key.startswith("SEQ:"):
                    n_seq += 1
                else:
                    n_feat += 1
                if rec.move.source and rec.move.source not in sources:
                    sources.append(rec.move.source)
                if len(samples) < 5:
                    samples.append(tpl)
        return {
            "rows_with_target": n_total,
            "rows_in_sequence": n_seq,
            "rows_in_feature": n_feat,
            "sources_used": sources,
            "sample_templates": samples,
        }

    def suggest_next(self, prev_tactic: str, top_k: int = 10) -> list[ProofPriorMove]:
        """Return up to `top_k` continuation moves that have followed
        `prev_tactic` in mined sequences. Empty list when `prev_tactic`
        is empty / unseen. Independent of the feature-based `suggest()`."""
        if not prev_tactic:
            return []
        bucket = self._by_key.get(self.sequence_key(prev_tactic))
        if not bucket:
            return []
        ranked = sorted(
            (rec.move for rec in bucket.values()),
            key=lambda m: (-m.prior_probability, m.prior_cost, m.tactic_template),
        )
        return ranked[:top_k]

    def suggest_abstract(self, prev_class: str, top_k: int = 10) -> list[ProofPriorMove]:
        """Return up to `top_k` abstract continuation hints for a
        previous-tactic CLASS. Each returned move's `tactic_template`
        is a SHAPE label (e.g. "rw_at_hyp", "arithmetic_close"); the
        caller expands it into concrete tactics from the live state."""
        if not prev_class:
            return []
        bucket = self._by_key.get(self.abstract_key(prev_class))
        if not bucket:
            return []
        ranked = sorted(
            (rec.move for rec in bucket.values()),
            key=lambda m: (-m.prior_probability, m.prior_cost, m.tactic_template),
        )
        return ranked[:top_k]

    # ---- write API ---------------------------------------------------------

    def add_observation(
        self,
        features: ProofStateFeatures,
        move: ProofPriorMove,
        result: str,
    ) -> None:
        """Record one observation.

        `result` is "solved" | "progress" | "fail" — only "solved" and
        "progress" contribute to the count. This keeps a failed-tactic
        record from poisoning the prior."""
        if result not in ("solved", "progress"):
            return
        # Ingest under every level so suggest() falls back smoothly.
        for level in range(0, 5):
            key = feature_key(features, level=level)
            bucket = self._by_key.setdefault(key, {})
            rec = bucket.get(move.tactic_template)
            if rec is None:
                rec = _MoveRecord(
                    move=ProofPriorMove(
                        tactic_template=move.tactic_template,
                        tactic_class=move.tactic_class,
                        premise=move.premise,
                        prior_probability=0.0,
                        prior_cost=0.0,
                        source=move.source,
                        tags=move.tags,
                        requires_instantiation=move.requires_instantiation,
                        provenance=move.provenance,
                    ),
                    count=0,
                )
                bucket[move.tactic_template] = rec
            else:
                # Union new provenance ids into the existing record.
                if move.provenance:
                    merged = list(rec.move.provenance)
                    for p in move.provenance:
                        if p not in merged:
                            merged.append(p)
                    rec.move.provenance = tuple(merged)
            rec.count += 1
            self._total_by_key[key] = self._total_by_key.get(key, 0) + 1
        self._renormalise()

    def add_abstract_observation(
        self,
        prev_class: str,
        next_shape: str,
        *,
        next_class: str = "unknown",
        source: str = "abstract_continuation_prior",
        tags: tuple[str, ...] = (),
        provenance: tuple[str, ...] = (),
        result: str = "solved",
    ) -> None:
        """Record one (prev_class -> next_shape) transition. Used by
        the corpus miners to learn proof-step patterns INDEPENDENT of
        any theorem identity. The stored move's `tactic_template` is
        the next-tactic SHAPE; runtime instantiation is the caller's job."""
        if result not in ("solved", "progress"):
            return
        if not prev_class or not next_shape:
            return
        key = self.abstract_key(prev_class)
        bucket = self._by_key.setdefault(key, {})
        rec = bucket.get(next_shape)
        if rec is None:
            rec = _MoveRecord(
                move=ProofPriorMove(
                    tactic_template=next_shape,
                    tactic_class=next_class,
                    premise=None,
                    prior_probability=0.0,
                    prior_cost=0.0,
                    source=source,
                    tags=tags + ("abstract", next_shape),
                    requires_instantiation=True,
                    provenance=provenance,
                ),
                count=0,
            )
            bucket[next_shape] = rec
        else:
            if provenance:
                merged = list(rec.move.provenance)
                for p in provenance:
                    if p not in merged:
                        merged.append(p)
                rec.move.provenance = tuple(merged)
        rec.count += 1
        self._total_by_key[key] = self._total_by_key.get(key, 0) + 1
        self._renormalise()

    def add_sequence_observation(
        self,
        prev_tactic: str,
        move: ProofPriorMove,
        result: str = "solved",
    ) -> None:
        """Record one (prev_tactic_template -> next_tactic_template)
        transition. Keyed off `prev_tactic` only — the live search will
        look up continuations by `state.prefix_tactics[-1]`.

        Stored in the same `_by_key` store under a `SEQ:` key so the
        on-disk JSONL needs no schema change. `suggest_next` is the only
        consumer.
        """
        if result not in ("solved", "progress"):
            return
        if not prev_tactic or not move.tactic_template:
            return
        key = self.sequence_key(prev_tactic)
        bucket = self._by_key.setdefault(key, {})
        rec = bucket.get(move.tactic_template)
        if rec is None:
            rec = _MoveRecord(
                move=ProofPriorMove(
                    tactic_template=move.tactic_template,
                    tactic_class=move.tactic_class,
                    premise=move.premise,
                    prior_probability=0.0,
                    prior_cost=0.0,
                    source=move.source,
                    tags=move.tags,
                    requires_instantiation=move.requires_instantiation,
                    provenance=move.provenance,
                ),
                count=0,
            )
            bucket[move.tactic_template] = rec
        else:
            if move.provenance:
                merged = list(rec.move.provenance)
                for p in move.provenance:
                    if p not in merged:
                        merged.append(p)
                rec.move.provenance = tuple(merged)
        rec.count += 1
        self._total_by_key[key] = self._total_by_key.get(key, 0) + 1
        self._renormalise()

    # ---- internals ---------------------------------------------------------

    def _ingest_row(self, row: dict) -> None:
        key = row.get("feature_key")
        if not key:
            return
        template = row.get("tactic_template")
        if not template:
            return
        move = ProofPriorMove(
            tactic_template=template,
            tactic_class=row.get("tactic_class", "unknown"),
            premise=row.get("premise"),
            prior_probability=float(row.get("prior_probability", 0.0)),
            prior_cost=float(row.get("prior_cost", 0.0)),
            source=row.get("source", "unknown"),
            tags=tuple(row.get("tags", []) or []),
            requires_instantiation=bool(row.get("requires_instantiation", False)),
            provenance=tuple(row.get("provenance", []) or []),
        )
        count = int(row.get("count", 1))
        bucket = self._by_key.setdefault(key, {})
        rec = bucket.get(template)
        if rec is None:
            bucket[template] = _MoveRecord(move=move, count=count)
            self._total_by_key[key] = self._total_by_key.get(key, 0) + count
        else:
            rec.count += count
            self._total_by_key[key] = self._total_by_key.get(key, 0) + count
            # Union provenance into the existing record so the audit
            # sees the contribution from every input file.
            if move.provenance:
                merged = list(rec.move.provenance)
                for p in move.provenance:
                    if p not in merged:
                        merged.append(p)
                rec.move.provenance = tuple(merged)

    def _renormalise(self) -> None:
        for key, bucket in self._by_key.items():
            total = max(1, self._total_by_key.get(key, sum(r.count for r in bucket.values())))
            for rec in bucket.values():
                p = (rec.count + self.EPSILON) / (total + self.EPSILON * max(1, len(bucket)))
                rec.move.prior_probability = p
                rec.move.prior_cost = -math.log(p) if p > 0 else float("inf")
