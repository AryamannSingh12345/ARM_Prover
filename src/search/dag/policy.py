"""Point 5 — canonical explicit repair policy (stage objects).

The legacy repair control flow is a sequence of accumulated conditionals
inside `attempt_dag_proof`. The scheduler migration first captured its
ORDER as a pure function (`scheduler.plan_round`); this module gives that
order a first-class, ablatable object form, as the architecture review
recommended: an ordered list of `RepairStage`s, each exposing

    can_handle(ctx)      -> bool     # is this stage's gate open?
    estimated_cost()     -> float    # for the future cost-ordered ablation
    attempt(ctx, engine) -> AttemptResult

CONSERVATIVE-MIGRATION CONTRACT (unchanged by this module):

* `RepairPolicy` is **planning / shadow only**. `plan(ctx)` returns the
  stages whose gate is open, in policy order, and — for the default
  as-built policy — MUST equal `scheduler.plan_round(ctx)` stage-for-
  stage. This is verified exhaustively in `tests/test_repair_policy.py`.
  `plan` is an INDEPENDENT second implementation of the order (it does
  not call `plan_round`); the parity test binds the two so neither can
  drift — the migration's standard method.

* `attempt` is the execution seam for scheduler-CONTROLLED execution,
  which is gated behind human review. The default stages raise
  `NotImplementedError` from `attempt`, so nothing here can run the live
  loop yet. Wiring each stage's `attempt` to its legacy capability
  (Points 3/8) and flipping execution are later, separately-gated
  patches. Constructing a policy and calling `plan` is always safe.

* Reordering the stages (e.g. cheapest-cost-first, the review's
  suggestion) is an explicit ABLATION, not the default: it changes
  `plan` output and so must be opted into and measured. `default_policy`
  returns the as-built order; `cost_ordered_policy` is provided for the
  ablation but is NOT the migration baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .events import (
    STAGE_BARE_TIMEOUT_THEORY, STAGE_LEAF_CLOSER, STAGE_THEORY_LEAF,
    STAGE_THEORY_CLOSER, STAGE_EAGER_ABDUCE, STAGE_DECOMPOSE,
    STAGE_LLM_REPAIR,
)
from .model import CLOSER_ID
from .scheduler import RoundContext


@dataclass(slots=True)
class AttemptResult:
    """Outcome of executing one repair stage (used once `attempt` is wired
    for scheduler-controlled execution). `fixed` are obligation ids the
    stage closed; `terminal` means the stage ended the repair loop (e.g.
    an empty/identical llm_repair response)."""
    stage: str
    fixed: tuple[str, ...] = ()
    changed: bool = False
    terminal: bool = False
    detail: str = ""


def _in_normal_branch(ctx: RoundContext) -> bool:
    """The legacy loop takes the bare-timeout branch (mutually exclusive
    with every normal pass) only when there are no attributable errors AND
    the setup is not broken. Every normal stage is gated on the negation."""
    return ctx.has_attributable_errors or ctx.setup_broken


class RepairStage:
    """Base stage. Subclasses set `name`, `cost`, and `_open`."""
    name: str = ""
    cost: float = 1.0

    def can_handle(self, ctx: RoundContext) -> bool:  # pragma: no cover
        raise NotImplementedError

    def estimated_cost(self) -> float:
        return self.cost

    def attempt(self, ctx: RoundContext, engine: Any) -> AttemptResult:
        # The wired scheduler engine AUTHORISES stages via plan() and lets
        # the legacy pass bodies execute them (attempt_dag_proof,
        # dag_repair_engine="scheduler"). `attempt` is the seam for a
        # future full dispatcher (stage bodies owned by the stage objects,
        # enabling true reordering); it stays unimplemented until that
        # extraction, so it cannot silently fork the execution path.
        raise NotImplementedError(
            f"stage {self.name!r}: stage-owned execution is not extracted "
            "yet — the scheduler engine authorises stages via "
            "RepairPolicy.plan(); see attempt_dag_proof(dag_repair_engine).")


class _BareTimeoutTheory(RepairStage):
    name = STAGE_BARE_TIMEOUT_THEORY
    cost = 5.0

    def can_handle(self, ctx: RoundContext) -> bool:
        return (not _in_normal_branch(ctx)
                and ctx.abduce_lemmas and ctx.abduce_mode == "theory"
                and not ctx.theory_spent and ctx.has_probe and ctx.has_haves)


class _LeafCloser(RepairStage):
    name = STAGE_LEAF_CLOSER
    cost = 2.0

    def can_handle(self, ctx: RoundContext) -> bool:
        return (_in_normal_branch(ctx)
                and ctx.has_leaf_closer and bool(ctx.broken_leaf_ids))


class _TheoryLeaf(RepairStage):
    name = STAGE_THEORY_LEAF
    cost = 5.0

    def can_handle(self, ctx: RoundContext) -> bool:
        broken = ctx.broken_leaf_ids
        if not (_in_normal_branch(ctx) and ctx.abduce_lemmas and broken
                and ctx.abduce_mode == "theory" and not ctx.theory_spent):
            return False
        return (ctx.trigger == "always" or any(
            ctx.streaks.get(i, 0) >= 2 or ctx.leaf_timeout.get(i, False)
            for i in broken))


class _TheoryCloser(RepairStage):
    name = STAGE_THEORY_CLOSER
    cost = 5.0

    def can_handle(self, ctx: RoundContext) -> bool:
        if not (_in_normal_branch(ctx) and ctx.abduce_lemmas
                and ctx.abduce_mode == "theory" and not ctx.theory_spent
                and not ctx.broken_leaf_ids and ctx.closer_broken
                and ctx.has_probe and ctx.header_splittable):
            return False
        return (ctx.trigger == "always"
                or ctx.streaks.get(CLOSER_ID, 0) >= 2 or ctx.closer_timeout)


class _EagerAbduce(RepairStage):
    name = STAGE_EAGER_ABDUCE
    cost = 4.0

    def can_handle(self, ctx: RoundContext) -> bool:
        return (_in_normal_branch(ctx) and ctx.abduce_lemmas
                and bool(ctx.broken_leaf_ids) and ctx.abduce_mode == "eager")


class _Decompose(RepairStage):
    name = STAGE_DECOMPOSE
    cost = 4.0

    def can_handle(self, ctx: RoundContext) -> bool:
        broken = ctx.broken_leaf_ids
        return (_in_normal_branch(ctx) and ctx.decompose_depth > 0
                and bool(broken)
                and any(ctx.streaks.get(i, 0) >= 2 for i in broken))


class _LlmRepair(RepairStage):
    name = STAGE_LLM_REPAIR
    cost = 3.0

    def can_handle(self, ctx: RoundContext) -> bool:
        return (_in_normal_branch(ctx)
                and (bool(ctx.broken_leaf_ids) or ctx.closer_broken
                     or ctx.setup_broken))


# As-built order (identical to events.STAGE_ORDER).
_DEFAULT_STAGE_CLASSES = (
    _BareTimeoutTheory, _LeafCloser, _TheoryLeaf, _TheoryCloser,
    _EagerAbduce, _Decompose, _LlmRepair,
)


@dataclass(slots=True)
class RepairPolicy:
    """An ordered list of repair stages. `plan(ctx)` returns the names of
    the stages whose gate is open, in policy order."""
    stages: list[RepairStage] = field(default_factory=list)

    def plan(self, ctx: RoundContext) -> list[str]:
        return [s.name for s in self.stages if s.can_handle(ctx)]

    def stage(self, name: str) -> RepairStage | None:
        for s in self.stages:
            if s.name == name:
                return s
        return None


def default_policy() -> RepairPolicy:
    """The migration baseline: stages in exact as-built order. `plan` on
    this policy equals `scheduler.plan_round` (parity-tested)."""
    return RepairPolicy([cls() for cls in _DEFAULT_STAGE_CLASSES])


def cost_ordered_policy() -> RepairPolicy:
    """ABLATION ONLY (not the baseline): stages sorted by estimated cost,
    cheapest first. Changes `plan` output — must be measured against the
    baseline, never silently adopted."""
    stages = [cls() for cls in _DEFAULT_STAGE_CLASSES]
    stages.sort(key=lambda s: s.estimated_cost())
    return RepairPolicy(stages)
