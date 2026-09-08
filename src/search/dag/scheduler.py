"""Repair scheduler — shadow-mode scaffolding for the conservative migration.

This module encodes the EXACT as-built repair order of the legacy
`attempt_dag_proof` loop as a pure, testable decision function
(`plan_round`). It does not yet control execution. Three engine modes:

- ``legacy``  — the current loop runs unchanged; the scheduler is inert
  (nothing here is constructed or invoked on the live path).
- ``shadow``  — the legacy loop makes every real decision; the scheduler
  independently computes what it *would* have planned from the same
  round context and reports divergences. It calls no Lean, spends no
  tokens, mutates nothing.
- ``scheduler`` — reserved for scheduler-controlled execution; DEFERRED
  pending review. Constructing this mode raises, by design.

`plan_round` mirrors the legacy gating one-to-one. Cheapest-first
reordering is a later
ablation, explicitly NOT part of the migration — the migration must first
prove byte-parity with the current order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .events import (
    STAGE_BARE_TIMEOUT_THEORY, STAGE_LEAF_CLOSER, STAGE_THEORY_LEAF,
    STAGE_THEORY_CLOSER, STAGE_EAGER_ABDUCE, STAGE_DECOMPOSE,
    STAGE_LLM_REPAIR,
)
from .model import CLOSER_ID

ENGINE_MODES = ("legacy", "shadow", "scheduler")


@dataclass(slots=True)
class RoundContext:
    """Everything the legacy loop's gating reads at repair-round entry.
    Reconstructable from the `repair_round` trace event plus the run's
    static configuration."""
    has_attributable_errors: bool
    setup_broken: bool
    closer_broken: bool
    broken_leaf_ids: tuple[str, ...]
    streaks: dict[str, int] = field(default_factory=dict)
    leaf_timeout: dict[str, bool] = field(default_factory=dict)
    closer_timeout: bool = False
    # static config
    abduce_lemmas: bool = False
    abduce_mode: str | None = None          # "eager" | "theory" | None
    theory_spent: bool = False
    has_leaf_closer: bool = False
    has_probe: bool = False
    decompose_depth: int = 0
    trigger: str = "stuck"                   # "stuck" | "always"
    has_haves: bool = True
    header_splittable: bool = True


def plan_round(ctx: RoundContext) -> list[str]:
    """Return the ordered list of repair stages the LEGACY loop would
    ATTEMPT for this round context. One-to-one with the as-built passes;
    no reordering, no new stages."""
    broken = set(ctx.broken_leaf_ids)

    # The bare-timeout branch is mutually exclusive with the normal passes:
    # the legacy loop enters it only when there are no attributable errors
    # AND the setup is not broken.
    if not ctx.has_attributable_errors and not ctx.setup_broken:
        if (ctx.abduce_lemmas and ctx.abduce_mode == "theory"
                and not ctx.theory_spent and ctx.has_probe and ctx.has_haves):
            return [STAGE_BARE_TIMEOUT_THEORY]
        return []  # legacy breaks out of the repair loop

    stages: list[str] = []

    # Pass 1 — specialised leaf closer
    if ctx.has_leaf_closer and broken:
        stages.append(STAGE_LEAF_CLOSER)

    # Pass 1.5-T — theory abduction, leaf-stuck
    if (ctx.abduce_lemmas and broken and ctx.abduce_mode == "theory"
            and not ctx.theory_spent):
        if ctx.trigger == "always" or any(
                ctx.streaks.get(i, 0) >= 2 or ctx.leaf_timeout.get(i, False)
                for i in broken):
            stages.append(STAGE_THEORY_LEAF)

    # Pass 1.5-TC — theory abduction, closer-stuck
    if (ctx.abduce_lemmas and ctx.abduce_mode == "theory"
            and not ctx.theory_spent and not broken and ctx.closer_broken
            and ctx.has_probe):
        if (ctx.trigger == "always"
                or ctx.streaks.get(CLOSER_ID, 0) >= 2 or ctx.closer_timeout):
            if ctx.header_splittable:
                stages.append(STAGE_THEORY_CLOSER)

    # Pass 1.5 — eager lemma abduction
    if ctx.abduce_lemmas and broken and ctx.abduce_mode == "eager":
        stages.append(STAGE_EAGER_ABDUCE)

    # Pass 2 — recursive decomposition
    if ctx.decompose_depth > 0 and broken:
        if any(ctx.streaks.get(i, 0) >= 2 for i in broken):
            stages.append(STAGE_DECOMPOSE)

    # Pass 3 — sketch-LLM repair
    if broken or ctx.closer_broken or ctx.setup_broken:
        stages.append(STAGE_LLM_REPAIR)

    return stages


@dataclass(slots=True)
class Divergence:
    round_no: int
    legacy_stages: list[str]
    scheduler_stages: list[str]
    reason: str


def _is_ordered_subsequence(sub: list[str], full: list[str]) -> bool:
    """True iff every element of `sub` appears in `full` in the same
    relative order (contiguity not required)."""
    it = iter(full)
    return all(x in it for x in sub)


def compare_round(round_no: int, legacy_stages: list[str],
                  ctx: RoundContext) -> Divergence | None:
    """Shadow parity for one round.

    The scheduler plans from ROUND-ENTRY state and cannot observe that a
    stage which succeeded mid-round (e.g. a committed closer-stuck theory)
    cleared the obligation a later stage would have handled — that is a
    runtime outcome, and this migration deliberately adds NO mid-round
    execution hook to the legacy loop. So shadow parity asserts the
    weaker, sound property the observation supports:

      the stages the legacy loop FIRED must be an ordered subsequence of
      the scheduler's plan — same order, same eligibility, with any
      planned-but-unfired stage explained by an earlier stage resolving
      the work.

    A divergence here means the legacy loop fired a stage the scheduler
    did NOT plan, or fired stages OUT of the as-built order — the only
    things that would make scheduler-controlled execution unsafe. Full
    execution parity (identical short-circuiting) is what the gated
    ``scheduler`` engine mode must demonstrate before it may become
    default. Returns a `Divergence` iff the subsequence property fails."""
    planned = plan_round(ctx)
    if _is_ordered_subsequence(list(legacy_stages), planned):
        return None
    return Divergence(
        round_no=round_no, legacy_stages=list(legacy_stages),
        scheduler_stages=planned,
        reason="legacy fired a stage the scheduler did not plan, or out "
               "of the as-built order")


class RepairEngine:
    """SHADOW-OBSERVATION handle (legacy|shadow only). Scheduler-authorised
    EXECUTION does not go through this object — it is selected via
    `attempt_dag_proof(dag_repair_engine="scheduler")`, which gates each
    repair pass on `RepairPolicy.plan` (see `dag/policy.py`). Constructing
    this handle in ``scheduler`` mode therefore stays an error: there is
    nothing for a scheduler-mode *observer* to observe."""

    def __init__(self, mode: str = "legacy"):
        if mode not in ENGINE_MODES:
            raise ValueError(f"unknown repair engine mode: {mode!r}")
        if mode == "scheduler":
            raise NotImplementedError(
                "RepairEngine is the shadow-observation handle; "
                "scheduler-authorised execution is selected via "
                "attempt_dag_proof(dag_repair_engine='scheduler') / "
                "--dag-repair-engine scheduler, not via this object.")
        self.mode = mode
        self.divergences: list[Divergence] = []

    @property
    def is_shadow(self) -> bool:
        return self.mode == "shadow"

    def observe(self, round_no: int, legacy_stages: list[str],
                ctx: RoundContext) -> None:
        """Shadow-only: record any divergence for this round. No effect in
        legacy mode (and legacy never constructs this object)."""
        if not self.is_shadow:
            return
        d = compare_round(round_no, legacy_stages, ctx)
        if d is not None:
            self.divergences.append(d)

    def report(self) -> dict:
        return {
            "mode": self.mode,
            "n_divergences": len(self.divergences),
            "divergences": [
                {"round": d.round_no, "legacy": d.legacy_stages,
                 "scheduler": d.scheduler_stages, "reason": d.reason}
                for d in self.divergences],
        }
