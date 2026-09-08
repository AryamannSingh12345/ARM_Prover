"""Phase 1 — construct and update proof obligations.

Derives `ObligationState` objects from a parsed `Sketch` (leaves + the
setup block + the closer-as-pseudo-leaf), and provides the small set of
state transitions the scheduler will drive: record an attempt (with
identical-attempt dedup), classify a Lean failure, mark verified/failed.

This module owns NO Lean or LLM access and mutates only the obligation
objects it is handed — never the source `Sketch` (the assembly contract
stays with `proof_dag`). It is the join point between the existing
sketch representation and the Phase 2 scheduler.
"""
from __future__ import annotations

import re

from .model import (
    ObligationState, ObligationAttempt, SETUP_ID, CLOSER_ID,
)

# Failure classification is deliberately coarse and generic (no
# problem-specific patterns): the scheduler routes on these classes.
_TIMEOUT_RE = re.compile(r"\btimeout\b|maximum number of heartbeats", re.I)
_UNKNOWN_RE = re.compile(r"unknown (identifier|constant)", re.I)
_UNKNOWN_NAME_RE = re.compile(
    r"[Uu]nknown (?:identifier|constant)[^`\n]*`([^`\s]+)`")
_UNSOLVED_RE = re.compile(r"\bunsolved goals\b", re.I)
_TYPE_MISMATCH_RE = re.compile(
    r"type mismatch|application type mismatch", re.I)
_SYNTH_RE = re.compile(r"failed to synthesize", re.I)
_PARSE_RE = re.compile(r"unexpected (token|identifier)|expected ", re.I)


def classify_failure(error: str | None) -> tuple[list[str], list[str]]:
    """Return (failure_classes, unknown_identifier_names) for a Lean error
    blob. Classes are generic and may be multiple: `timeout`,
    `unknown_identifier`, `unsolved_goals`, `type_mismatch`,
    `synth_failure`, `parse_error`, or `other`."""
    text = error or ""
    classes: list[str] = []
    if _TIMEOUT_RE.search(text):
        classes.append("timeout")
    if _UNKNOWN_RE.search(text):
        classes.append("unknown_identifier")
    if _UNSOLVED_RE.search(text):
        classes.append("unsolved_goals")
    if _TYPE_MISMATCH_RE.search(text):
        classes.append("type_mismatch")
    if _SYNTH_RE.search(text):
        classes.append("synth_failure")
    if _PARSE_RE.search(text):
        classes.append("parse_error")
    if not classes and text.strip():
        classes.append("other")
    unknowns = list(dict.fromkeys(_UNKNOWN_NAME_RE.findall(text)))
    return classes, unknowns


def build_obligations_from_sketch(
    sketch, *, main_goal: str,
) -> dict[str, ObligationState]:
    """Derive one obligation per have (source_kind 'leaf'), one for the
    setup block, and the closer posed as a pseudo-leaf (source_kind
    'closer', goal = the theorem's goal, depending on every have).

    Does NOT mutate `sketch`. Returns an id → ObligationState map in a
    deterministic order (setup, haves in list order, closer)."""
    obs: dict[str, ObligationState] = {}

    if getattr(sketch, "setup", None):
        obs[SETUP_ID] = ObligationState(
            id=SETUP_ID,
            goal_text="\n".join(sketch.setup),
            dependencies=(),
            source_kind="setup",
            provenance={"origin": "sketch.setup"},
        )

    for h in sketch.haves:
        obs[h.id] = ObligationState(
            id=h.id,
            goal_text=h.type_text,
            dependencies=tuple(h.depends),
            source_kind="leaf",
            candidate_moves=[],
            provenance={"origin": "sketch.have"},
        )

    obs[CLOSER_ID] = ObligationState(
        id=CLOSER_ID,
        goal_text=main_goal,
        dependencies=tuple(h.id for h in sketch.haves),
        source_kind="closer",
        provenance={"origin": "sketch.closer"},
    )
    return obs


def record_attempt(ob: ObligationState, attempt: ObligationAttempt) -> bool:
    """Append `attempt` to the obligation's history unless an identical
    attempt (same solver + candidate + result) is already recorded — a
    re-run of a candidate that already produced the same outcome on an
    unchanged obligation carries no information. Also folds the attempt's
    failure diagnostics into the obligation. Returns True if recorded."""
    key = attempt.dedup_key()
    if any(a.dedup_key() == key for a in ob.attempts):
        return False
    ob.attempts.append(attempt)
    if attempt.result == "timeout":
        ob.timeout_count += 1
    if attempt.result in ("failed", "error", "timeout"):
        classes, unknowns = classify_failure(attempt.error)
        for c in classes:
            if c not in ob.failure_classes:
                ob.failure_classes.append(c)
        for u in unknowns:
            if u not in ob.unknown_identifiers:
                ob.unknown_identifiers.append(u)
        if ob.status == "pending":
            ob.status = "in_progress"
    return True


def mark_verified(ob: ObligationState, tactic: str) -> None:
    """Mark the obligation closed by `tactic`. Idempotent."""
    ob.status = "verified"
    ob.verified_tactic = tactic


def mark_failed(ob: ObligationState, classes: list[str] | None = None) -> None:
    ob.status = "failed"
    for c in classes or []:
        if c not in ob.failure_classes:
            ob.failure_classes.append(c)


def is_stuck(ob: ObligationState, streak_threshold: int = 2) -> bool:
    """Scheduler-facing stuckness: broken repeatedly, or dying on a
    timeout/heartbeat — the condition that escalates to decomposition or
    theory abduction. Mirrors the current `broken_streak >= 2 or timeout`
    logic but reads it off obligation history instead of a side dict."""
    failed = sum(1 for a in ob.attempts
                 if a.result in ("failed", "error", "timeout"))
    return failed >= streak_threshold or ob.timeout_count > 0
