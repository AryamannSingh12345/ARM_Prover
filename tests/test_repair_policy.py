"""Point 5 — canonical RepairPolicy stage objects.

The critical test is PARITY: `default_policy().plan(ctx)` must equal the
reference `scheduler.plan_round(ctx)` for every round context, so the
object form is a faithful, drift-proof restatement of the as-built order.
Also checks the execution gate (attempt raises) and the cost-ordered
ablation is a real reordering.
"""
from __future__ import annotations

import itertools
import random

import pytest

from search.dag.scheduler import RoundContext, plan_round
from search.dag import policy as P
from search.dag.model import CLOSER_ID
from search.dag.events import STAGE_ORDER


def _ctx(**kw) -> RoundContext:
    base = dict(
        has_attributable_errors=True, setup_broken=False,
        closer_broken=False, broken_leaf_ids=(), streaks={},
        leaf_timeout={}, closer_timeout=False, abduce_lemmas=False,
        abduce_mode=None, theory_spent=False, has_leaf_closer=False,
        has_probe=False, decompose_depth=0, trigger="stuck",
        has_haves=True, header_splittable=True)
    base.update(kw)
    return RoundContext(**base)


# ---------------- exhaustive-ish parity vs plan_round -----------------------

def _iter_contexts():
    """Structured cartesian product across every decision-relevant field,
    including streak / timeout variants tied to the broken set. Exercises
    all seven stages' gates and the bare-timeout mutual exclusion."""
    bools = (False, True)
    broken_sets = ((), ("h0",), ("h0", "h1"))
    modes = (None, "eager", "theory")
    for (hae, setup_b, closer_b, ablem, tspent, hlc, hprobe,
         ctimeout, hhaves, hsplit) in itertools.product(bools, repeat=10):
        for broken in broken_sets:
            for mode in modes:
                for depth in (0, 1):
                    for trig in ("stuck", "always"):
                        # streak / timeout variants
                        for streak_val in (0, 2):
                            streaks = {i: streak_val for i in broken}
                            streaks[CLOSER_ID] = streak_val
                            for lt in (False, True):
                                leaf_timeout = {i: lt for i in broken}
                                yield _ctx(
                                    has_attributable_errors=hae,
                                    setup_broken=setup_b,
                                    closer_broken=closer_b,
                                    broken_leaf_ids=broken,
                                    streaks=streaks,
                                    leaf_timeout=leaf_timeout,
                                    closer_timeout=ctimeout,
                                    abduce_lemmas=ablem, abduce_mode=mode,
                                    theory_spent=tspent, has_leaf_closer=hlc,
                                    has_probe=hprobe, decompose_depth=depth,
                                    trigger=trig, has_haves=hhaves,
                                    header_splittable=hsplit)


def test_default_policy_plan_matches_plan_round_exhaustively():
    pol = P.default_policy()
    checked = 0
    for ctx in _iter_contexts():
        assert pol.plan(ctx) == plan_round(ctx), ctx
        checked += 1
    # Sanity: the generator actually produced a large battery.
    assert checked > 100_000


def test_default_policy_stage_order_is_as_built():
    pol = P.default_policy()
    assert [s.name for s in pol.stages] == list(STAGE_ORDER)


# ---------------- per-stage gating spot checks ------------------------------

def test_bare_timeout_theory_is_mutually_exclusive():
    # No attributable errors, no setup breakage → only bare-timeout eligible.
    ctx = _ctx(has_attributable_errors=False, setup_broken=False,
               abduce_lemmas=True, abduce_mode="theory", has_probe=True,
               has_haves=True, broken_leaf_ids=("h0",), has_leaf_closer=True,
               streaks={"h0": 2})
    assert P.default_policy().plan(ctx) == ["bare_timeout_theory"]


def test_normal_branch_orders_leaf_closer_then_repair():
    ctx = _ctx(has_attributable_errors=True, broken_leaf_ids=("h0",),
               has_leaf_closer=True)
    assert P.default_policy().plan(ctx) == ["leaf_closer", "llm_repair"]


def test_theory_leaf_requires_trigger():
    common = dict(has_attributable_errors=True, broken_leaf_ids=("h0",),
                  abduce_lemmas=True, abduce_mode="theory")
    # streak 0, trigger 'stuck' → theory_leaf NOT eligible.
    assert "theory_leaf_stuck" not in P.default_policy().plan(
        _ctx(**common, streaks={"h0": 0}))
    # streak 2 → eligible.
    assert "theory_leaf_stuck" in P.default_policy().plan(
        _ctx(**common, streaks={"h0": 2}))
    # trigger 'always' → eligible even at streak 0.
    assert "theory_leaf_stuck" in P.default_policy().plan(
        _ctx(**common, streaks={"h0": 0}, trigger="always"))


def test_decompose_requires_depth_and_streak():
    common = dict(has_attributable_errors=True, broken_leaf_ids=("h0",))
    assert "decompose" not in P.default_policy().plan(
        _ctx(**common, decompose_depth=0, streaks={"h0": 2}))
    assert "decompose" in P.default_policy().plan(
        _ctx(**common, decompose_depth=1, streaks={"h0": 2}))


# ---------------- execution gate + ablation ---------------------------------

def test_attempt_is_gated():
    stage = P.default_policy().stages[0]
    with pytest.raises(NotImplementedError):
        stage.attempt(_ctx(), engine=None)


def test_cost_ordered_is_a_reordering():
    default_names = [s.name for s in P.default_policy().stages]
    cost_names = [s.name for s in P.cost_ordered_policy().stages]
    assert sorted(default_names) == sorted(cost_names)
    assert default_names != cost_names  # genuinely reordered
    costs = [s.estimated_cost() for s in P.cost_ordered_policy().stages]
    assert costs == sorted(costs)


def test_estimated_cost_positive():
    for s in P.default_policy().stages:
        assert s.estimated_cost() > 0
