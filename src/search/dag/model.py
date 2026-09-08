"""Phase 1 — first-class proof-obligation model.

`HaveNode`/`Sketch` (in `search.proof_dag`) are enough to *assemble* a
proof, but not enough to *schedule*, *learn from*, *cache*, or *audit*
the work of closing each obligation. This module adds a structured
obligation model that carries per-obligation status, attempt history,
candidate provenance, failure diagnostics, and difficulty — the state the
explicit scheduler (Phase 2) and the deterministic leaf solvers (Phase 3)
need. It is purely additive: an `ObligationState` is *derived from* a
sketch (see `obligations.build_obligations_from_sketch`); the sketch
JSON contract is untouched.

Nothing here touches Lean or the network. Every dataclass round-trips
through `to_dict`/`from_dict` so obligation state can appear in
`DagResult`, JSONL rows, and traces.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Reserved segment ids, mirrored from `search.proof_dag` (a consistency
# test guards against drift; Phase 4 unifies the definition). A leaf uses
# its own have id; these name the two non-leaf obligations.
SETUP_ID = "__setup__"
CLOSER_ID = "__closer__"
HEADER_ID = "__header__"

# ---- vocabularies (open strings, documented for auditability) --------------
# source_kind: how the obligation came to exist.
SOURCE_KINDS = ("leaf", "setup", "closer", "decomposed", "abduced")
# status: lifecycle.
STATUSES = ("pending", "in_progress", "verified", "failed", "stuck",
            "abandoned")
# ObligationAttempt.result: outcome of one solver attempt.
ATTEMPT_RESULTS = ("verified", "failed", "error", "timeout", "skipped")


@dataclass(slots=True)
class CandidateMove:
    """A single candidate tactic proposed for an obligation, with its
    provenance. Sources: `fallback_ladder`, `proof_prior`, `template`,
    `dependency`, `leaf_closer`, `llm_repair`, `decompose`, `abduction`."""
    source: str
    tactic: str
    cost: float = 1.0            # heuristic cost, lower = try first
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateMove":
        return cls(source=d["source"], tactic=d["tactic"],
                   cost=float(d.get("cost", 1.0)),
                   metadata=dict(d.get("metadata") or {}))


@dataclass(slots=True)
class ObligationAttempt:
    """One attempt by one solver on one obligation. `candidate` is the
    exact tactic tried; `result` is an `ATTEMPT_RESULTS` value; `error` is
    the (warning-stripped) Lean text on failure. Immutable record."""
    solver: str
    candidate: str
    result: str
    error: str | None
    wall_s: float
    token_usage: dict[str, int] | None
    verifier_backend: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ObligationAttempt":
        tu = d.get("token_usage")
        return cls(
            solver=d["solver"], candidate=d["candidate"],
            result=d["result"], error=d.get("error"),
            wall_s=float(d.get("wall_s", 0.0)),
            token_usage=(dict(tu) if tu is not None else None),
            verifier_backend=d.get("verifier_backend", ""))

    def dedup_key(self) -> tuple[str, str, str]:
        """Attempts identical on (solver, candidate, result) against an
        unchanged obligation carry no new information — the scheduler uses
        this to avoid re-running a candidate that already failed."""
        return (self.solver, self.candidate, self.result)


@dataclass(slots=True)
class ObligationState:
    """One proof obligation (a leaf `have`, the setup block, or the closer
    posed as a pseudo-leaf). Scheduling/learning/audit state lives here;
    the tactic that finally closes it is `verified_tactic` once `status`
    is `verified`."""
    id: str
    goal_text: str
    dependencies: tuple[str, ...]
    local_context: str | None = None
    source_kind: str = "leaf"
    status: str = "pending"
    attempts: list[ObligationAttempt] = field(default_factory=list)
    verified_tactic: str | None = None
    failure_classes: list[str] = field(default_factory=list)
    unknown_identifiers: list[str] = field(default_factory=list)
    timeout_count: int = 0
    estimated_difficulty: float | None = None
    candidate_premises: list[str] = field(default_factory=list)
    candidate_moves: list[CandidateMove] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal_text": self.goal_text,
            "dependencies": list(self.dependencies),
            "local_context": self.local_context,
            "source_kind": self.source_kind,
            "status": self.status,
            "attempts": [a.to_dict() for a in self.attempts],
            "verified_tactic": self.verified_tactic,
            "failure_classes": list(self.failure_classes),
            "unknown_identifiers": list(self.unknown_identifiers),
            "timeout_count": self.timeout_count,
            "estimated_difficulty": self.estimated_difficulty,
            "candidate_premises": list(self.candidate_premises),
            "candidate_moves": [c.to_dict() for c in self.candidate_moves],
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ObligationState":
        return cls(
            id=d["id"],
            goal_text=d["goal_text"],
            dependencies=tuple(d.get("dependencies") or ()),
            local_context=d.get("local_context"),
            source_kind=d.get("source_kind", "leaf"),
            status=d.get("status", "pending"),
            attempts=[ObligationAttempt.from_dict(a)
                      for a in d.get("attempts") or []],
            verified_tactic=d.get("verified_tactic"),
            failure_classes=list(d.get("failure_classes") or []),
            unknown_identifiers=list(d.get("unknown_identifiers") or []),
            timeout_count=int(d.get("timeout_count", 0)),
            estimated_difficulty=d.get("estimated_difficulty"),
            candidate_premises=list(d.get("candidate_premises") or []),
            candidate_moves=[CandidateMove.from_dict(c)
                             for c in d.get("candidate_moves") or []],
            provenance=dict(d.get("provenance") or {}),
        )

    def metrics(self) -> dict[str, Any]:
        """Compact per-obligation metrics for DagResult / JSONL."""
        return {
            "id": self.id,
            "source_kind": self.source_kind,
            "status": self.status,
            "n_attempts": len(self.attempts),
            "n_candidates": len(self.candidate_moves),
            "failure_classes": list(dict.fromkeys(self.failure_classes)),
            "timeout_count": self.timeout_count,
            "verified_by": (self.attempts[-1].solver
                            if self.status == "verified" and self.attempts
                            else None),
        }
