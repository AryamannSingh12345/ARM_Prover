"""Structured decision events for the conservative scheduler migration.

The legacy `attempt_dag_proof` already emits a rich `trace(kind, **payload)`
stream at every decision point. Rather than modify the live loop, the
migration *observes* that stream: `EventRecorder` wraps an (optional)
inner trace callable, forwards every call unchanged, and additionally
distils the repair-relevant events into an ordered list of structured
`DecisionEvent`s. This gives golden snapshots of ACTUAL legacy behaviour
and the per-round context the scheduler's parity check consumes — with
zero behavioural change to proving.

Stage identifiers name the seven repair stages in as-built order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Repair stages, in the exact as-built order of the legacy loop.
STAGE_BARE_TIMEOUT_THEORY = "bare_timeout_theory"
STAGE_LEAF_CLOSER = "leaf_closer"
STAGE_THEORY_LEAF = "theory_leaf_stuck"
STAGE_THEORY_CLOSER = "theory_closer_stuck"
STAGE_EAGER_ABDUCE = "eager_abduce"
STAGE_DECOMPOSE = "decompose"
STAGE_LLM_REPAIR = "llm_repair"

STAGE_ORDER = (
    STAGE_BARE_TIMEOUT_THEORY,
    STAGE_LEAF_CLOSER,
    STAGE_THEORY_LEAF,
    STAGE_THEORY_CLOSER,
    STAGE_EAGER_ABDUCE,
    STAGE_DECOMPOSE,
    STAGE_LLM_REPAIR,
)


@dataclass(slots=True)
class RoundDecision:
    """One repair round as observed from the trace: the entry context
    (from the `repair_round` event) and the stages that actually fired."""
    round_no: int
    broken: tuple[str, ...]
    closer_broken: bool
    setup_broken: bool
    stages_fired: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DecisionEvent:
    kind: str
    payload: dict[str, Any]


class EventRecorder:
    """Trace wrapper: forwards to `inner` (may be None) and records a
    structured, replayable view of the repair decisions.

    Never raises into proving — a recorder failure must not break a run
    (the trace contract). Use `.round_decisions()` for parity comparison
    and `.stages_in_round(n)` for per-round stage sequences."""

    def __init__(self, inner: Callable[..., None] | None = None):
        self._inner = inner
        self.events: list[DecisionEvent] = []
        self._rounds: list[RoundDecision] = []

    def __call__(self, kind: str, **payload: Any) -> None:
        try:
            self.events.append(DecisionEvent(kind, dict(payload)))
            self._absorb(kind, payload)
        except Exception:
            pass
        if self._inner is not None:
            self._inner(kind, **payload)

    def _absorb(self, kind: str, payload: dict) -> None:
        if kind == "repair_round":
            self._rounds.append(RoundDecision(
                round_no=int(payload.get("round", len(self._rounds) + 1)),
                broken=tuple(payload.get("broken") or ()),
                closer_broken=bool(payload.get("closer_broken")),
                setup_broken=bool(payload.get("setup_broken")),
            ))
            return
        if not self._rounds:
            # A stage-ish event before any repair_round: the bare-timeout
            # theory branch fires without a repair_round. Seed a synthetic
            # round so it is still captured.
            if kind == "abduce_theory" and "bare-timeout" in str(
                    payload.get("stage", "")):
                self._rounds.append(RoundDecision(
                    round_no=0, broken=(), closer_broken=False,
                    setup_broken=False,
                    stages_fired=[STAGE_BARE_TIMEOUT_THEORY]))
            return
        cur = self._rounds[-1]
        stage = self._stage_of(kind, payload, cur.stages_fired)
        if stage and stage not in cur.stages_fired:
            cur.stages_fired.append(stage)

    # theory stages: all events inside ONE `_abduce_theory` call belong to
    # that single trigger. Only the trigger markers (closer-stuck /
    # bare-timeout) name a stage directly; the leaf-stuck path has no
    # marker, so its first `proposed` event names it — but only if no
    # other theory stage has already fired this round.
    _THEORY_STAGES = (STAGE_THEORY_LEAF, STAGE_THEORY_CLOSER,
                      STAGE_BARE_TIMEOUT_THEORY)

    @classmethod
    def _stage_of(cls, kind: str, payload: dict,
                  fired: list[str]) -> str | None:
        if kind == "leaf_closer":
            return STAGE_LEAF_CLOSER
        if kind == "decompose":
            return STAGE_DECOMPOSE
        if kind in ("repair_applied", "repair_not_applied"):
            return STAGE_LLM_REPAIR
        if kind == "abduce":               # eager per-leaf abduction
            return STAGE_EAGER_ABDUCE
        if kind == "abduce_theory":
            s = str(payload.get("stage", ""))
            if "closer-stuck" in s:
                return STAGE_THEORY_CLOSER
            if "bare-timeout" in s:
                return STAGE_BARE_TIMEOUT_THEORY
            # a fresh proposal with no theory trigger yet this round = the
            # leaf-stuck path; otherwise it belongs to the active call.
            if s == "proposed" and not any(t in fired
                                           for t in cls._THEORY_STAGES):
                return STAGE_THEORY_LEAF
            return None
        return None

    def round_decisions(self) -> list[RoundDecision]:
        return list(self._rounds)

    def stages_in_round(self, round_no: int) -> list[str]:
        for r in self._rounds:
            if r.round_no == round_no:
                return list(r.stages_fired)
        return []
