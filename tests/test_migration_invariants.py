"""Automated confirmation of the migration preservation invariants.

These are tripwires: they fail loudly if a migration patch silently
changes the live prover. They must hold at every commit until the review
gate is lifted.

Invariants (re-baselined 2026-07-29 — the original review gate was
lifted deliberately for the scheduler-engine wiring, the leaf portfolio
and causal attribution; each landed flag-gated with its own
parity/legacy-when-off tests):
1. `attempt_dag_proof` (and `_apply_repairs`) source is frozen at the
   CURRENT frozen state — the tripwire still fails loudly on any
   FUTURE silent change to the live control flow. Re-freezing the hash
   requires the same explicit sign-off this baseline had.
2. default repair engine is `legacy` (scheduler is opt-in via
   `--dag-repair-engine scheduler`; shadow maps to legacy execution).
3. `RepairEngine` (the shadow-observation handle) still refuses
   `scheduler` — execution selection lives in attempt_dag_proof only.
4. all Phase 3 leaf-solver flags default to disabled.
5. disabled leaf solvers are neither imported nor instantiated.
6. shadow mode imports no Lean/LLM/backend module and mutates no input.
"""
from __future__ import annotations

import hashlib
import inspect
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from search.proof_dag import attempt_dag_proof, _apply_repairs  # noqa: E402
from search.dag.scheduler import RepairEngine  # noqa: E402
from search.dag.solvers import LeafSolverConfig, build_leaf_solvers  # noqa: E402

# Re-frozen 2026-07-29 after the July-29 round (scheduler
# engine, Pass-1.25 portfolio, causal attribution, minimize-verify,
# quality gate + prior — all flag-gated OFF, each with legacy-when-off
# tests). The original baseline-phase1 hash was retired deliberately.
# Changing either hash again REQUIRES the same explicit sign-off — that
# is the tripwire.
#
# `attempt_dag_proof` re-frozen again 2026-07-29 (later the same day) for
# the p25-autopsy round: the
# run-scoped lemma store and the failure-class prove-retry budget. Both
# arrive as new keyword arguments defaulting to None, and both are
# legacy-when-off down to the revision-ledger wording — see
# tests/test_lemma_store.py::test_retry_budget_off_* and
# ::test_legacy_discards_proved_lemmas_on_abandonment. `_apply_repairs`
# is UNCHANGED by that round and keeps its existing hash, which is the
# tripwire doing its job: the diff is confined to the theory loop.
#
# Re-frozen 2026-08-03 for the blueprint round: `sketch_mode="blueprint"`
# plus the sorry-stubbed sufficiency
# probe. Default is `sketch_mode="tactics"`, which takes the original
# prompt/parser and runs NO probe — see
# tests/test_blueprint.py::test_default_mode_uses_the_legacy_sketch_prompt.
# `_apply_repairs` is again UNCHANGED: the diff is confined to sketch
# generation and does not touch the repair loop.
# Re-frozen 2026-08-04 for the store-feedback round: proved
# store lemmas are spliced into the header for later sketch attempts
# (--store-feedback, default off) and error-class lessons accumulate
# across theory rounds. Defaults unchanged; _apply_repairs untouched.
# Re-frozen 2026-08-04 (bugfix): --store-feedback now splices the DEFS a
# stored lemma depends on before the lemma itself. Without that the
# feature produced a header-level error and killed putnam_2020_a2_v1.
# See tests/test_store_feedback_defs.py.
# Re-frozen 2026-08-04: minimal-edit repair on parse-class prove
# failures (syntax_only). See tests/test_bigop_syntax.py.
# Re-frozen 2026-08-05 for the reframe wiring: `--reframe-on-abandon`
# (default OFF)
# adds ONE proposal round at the END of the theory loop, reached only
# when every in-language round has failed and only for a SINGLE stuck
# leaf. The reframing is rendered into the ordinary theory-JSON shape by
# `reframe.theory_as_proposal`, so the circularity guard, quality gate,
# satisfaction probe, per-lemma kernel proving and all-or-nothing commit
# are REUSED, not duplicated — the conversion is the only new code path.
# `search.reframe` is imported inside the guarded branch, so the module
# is not even loaded when the flag is off (invariant 5, tested in
# tests/test_reframe_wiring.py). `_apply_repairs` is UNCHANGED again:
# the diff is confined to the theory loop.
# Re-frozen 2026-08-05 for the Stage 3 round: `--reframe-build-library`
# (default OFF) builds a small
# support library for a reframing's NEW OBJECTS before proving its
# claims, since a frame can fail purely for want of elementary facts
# about objects nothing is yet known about. Results are split by kind —
# definitions to `defs`, and proved lemmas into the PROVED-LEMMA BANK and
# back as claims, so the prove step reuses them for zero compiles (a
# proved lemma placed in `defs` would be silently dropped by
# `_DEF_DECL_RE`). Requires `library_compile_fn`, which compiles a
# COMPLETE source: `verify_proof` appends ":= by <body>" and would turn
# every finished declaration into a syntax error. Any failure returns
# nothing and the reframe proceeds unchanged. See
# tests/test_reframe_library.py. `_apply_repairs` UNCHANGED.
# Re-frozen 2026-08-05 for the recognition wiring: `--recognize-leaves`
# (default OFF) checks each stuck leaf
# against Mathlib's STATEMENTS before any theory is proposed, because
# inventing a lemma Mathlib already carries is the most expensive way to
# fail. `_recognize_then_abduce` wraps `_abduce_theory` at all three
# trigger sites; with the flag off it delegates immediately, so the
# legacy path is byte-identical in behaviour. A recognised leaf is CLOSED
# (its tactic was kernel-accepted on its own standalone statement) and
# stays fixed even if the theory loop later abandons the rest. Ordering
# is the point: recognize -> abduce -> reframe, cheapest first. See
# tests/test_recognize_wiring.py. `_apply_repairs` UNCHANGED.
# Re-frozen 2026-08-05 (BUGFIX, pre-existing): the satisfaction
# gate read a TIMED-OUT probe as a pass. A timeout carries no
# `file:line:col:` markers, so error attribution blamed no segment
# and the "errors are outside our leaves" branch returned True.
# On p2020a2_recognize_0805 a 600s probe timeout was gated OK and
# the run then spent a 103s LLM call and a further 600s compile
# proving a theory that had never been validated. Now fails closed.
# See tests/test_abduce_theory.py::test_probe_timeout_is_not_a_passing_gate.
# Re-frozen 2026-08-06 for the post-run round:
# a DUPLICATE-PROOF GUARD in the prove step. Identical proof text
# cannot earn a different verdict, so the compile is skipped, the
# attempt is still consumed, and the model is told it repeated
# itself. On p2020a2_recognize_0805 the same proof arrived three
# times for one lemma and cost three compiles (229s+283s+331s) for
# one bit of information. Keyed per LEMMA on whitespace-normalised
# proof text. See tests/test_abduce_theory.py::
# test_identical_proof_is_not_compiled_twice. The seen-set is CLEARED
# on an import refresh: an identical proof can compile once the
# environment moves, which is what the refresh is for.
# `_apply_repairs` UNCHANGED.
# Re-frozen 2026-08-08 (BUGFIX): --store-feedback re-spliced lemmas the
# theory-commit path had ALREADY written into theorem_header, because it
# deduped only on its own spliced-set. Lean then reported "has already
# been declared", a HEADER error, which aborts the run by design. On
# p1968a1_v1 that killed a 5-hour run holding NINE proved lemmas.
# `_surface_store_lemmas` now also skips any record whose declared name
# is already present in the header. See tests/test_store_feedback_defs.py.
# Re-frozen 2026-08-08 (BUGFIX, mirror case): the theory loop could
# re-propose a lemma that --store-feedback had already spliced into
# theorem_header, giving "has already been declared" and killing the
# round. Measured on p2023a2_v1 at t=10932s. Such a lemma is already
# in scope, so it is DROPPED from the proposal rather than rejected.
# See tests/test_store_feedback_defs.py.
# Re-frozen 2026-08-09 (BUGFIX): a proof body citing the theorem it
# proves makes Lean elaborate the declaration as RECURSIVE ("fail to
# show termination"), a HEADER error that aborts the attempt.
# p1967a2_v1 lost BOTH remaining sketch attempts to it (t=931s,
# t=2747s). Now caught BEFORE the compile, with the reason fed back.
# The check excludes a trailing `_` so the companion `..._solution`
# abbrev, which proofs legitimately cite, is not flagged.
# See tests/test_self_reference.py.
# Re-frozen 2026-08-17 (TACTIC HYGIENE): submitted
# lemma proofs now pass through `harden_tactic_block` before the gate
# compile — `try`-wrapping no-op-prone normalisation tactics and sweeping
# trivial residual goals. MEASURED on the putnam_1967_b5 run (15.3 h, 64
# compiles): 9 of 13 lost lemma attempts (69%) died of `simp made no
# progress` or an unclosed `1 + (1 + A) = 2 + A`-class goal, including
# the crux lemma three times. Cannot break a passing proof: `try t` is
# `t` whenever `t` succeeds, and `all_goals try …` is vacuous with no
# goals open. Gated by `tactic_hygiene` (default True).
# See tests/test_tactic_hygiene.py.
_FROZEN = {
# Re-frozen 2026-08-17 (PARTIAL COMMIT): when the
# theory rounds are exhausted, the proved subset is re-probed for
# sufficiency before the theory is discarded, and committed if it closes
# the stuck leaves. MEASURED on putnam_1967_b5 t=46563s: the crux lemma
# `aux_weighted_transform` was kernel-PROVED and the theory thrown away
# anyway, because three helper lemmas written only to support it had
# failed; ten proved lemmas then sat unused for two more hours.
# SOUNDNESS: cannot manufacture a solve — commit requires the ordinary
# satisfaction probe to pass with those declarations spliced for real,
# and the result still faces the normal verify loop and a fresh compile.
# See tests/test_partial_commit.py and test_lemma_store.py.
# Re-frozen 2026-08-18 (A2 REPAIR ROUND): three
# changes inside the loop, all traced to p1963a2_dag_v1.
#   A. theory LEAF TACTICS now pass through harden_tactic_block. Four
#      consecutive theory rounds died at the gate on the SAME tactic
#      coercion (`h_f1 : f 1 = 1` where `1 ≤ f 1` was wanted), ~1700s of
#      probe compiles, while the model rewrote the lemma each time.
#   B. a gate failure now states that every lemma was ASSUMED (sorry-
#      stubbed) and so cannot be the cause — the fault is a tactic, the
#      closer or the setup. Without that the revision optimised the
#      wrong object.
#   C. a COMMITTED theory earns one extra repair round (bounded at 1 per
#      sketch). Sketch 1 committed on the final round, fixed `h_lower`,
#      and had no iteration left for the separately-broken closer.
# See tests/test_a2_repair_round.py.
# Re-frozen 2026-08-18 for the two
# defects MEASURED on p1963a2_dag_v2, which died having used 1 of 3
# sketch attempts:
#   A. SPURIOUS-INTRO. In closer-stuck mode the closer is posed to the
#      model as a standalone goal, so it opened with `intro m hm` while
#      the sketch's setup had already introduced those binders. Lean:
#      "Tactic `introN` failed: There are no additional binders". This
#      killed theory rounds 0 AND 4 — and round 4 was the GOOD theory
#      (elementary gap-persistence / dyadic fixed points, proposed after
#      the over-general `rpow` classification was abandoned). The repair
#      is ERROR-DRIVEN: nothing is stripped until the kernel has emitted
#      that diagnostic against the segment, capped at ONE re-probe, and
#      written back into `tactics` so a committing theory carries the
#      fixed tactic into the real verify.
#   B. PARTIAL COMMIT vs UNPROVED NAMES. The probe reused the round's
#      tactics verbatim, so a tactic citing a lemma that never proved
#      died on `unknown identifier` rather than answering the
#      sufficiency question. Cost 455s to learn nothing. Such tactics
#      are now dropped first, and the probe is SKIPPED entirely when
#      none survives.
# Neither can manufacture a solve: both only change which candidate is
# probed, and every commit still faces the normal verify loop and a
# final fresh compile.
# See tests/test_a2_v2_repairs.py.
# Re-frozen 2026-08-22 for a header-error defect MEASURED on
# p1965a2_dag_v1. The theory
# revised its auxiliary def between rounds — same NAME, different body —
# and `LemmaStore.required_defs()` deduplicated by SOURCE TEXT, so both
# bodies survived. `_declared` is the header BEFORE the splice and so could
# not stop two same-named defs arriving in the SAME batch. Lean reported
# "`aux_chooseDeviationTerm` has already been declared" at t=41666s; a
# header error stops the repair loop by design, so sketch attempt 3 died on
# its first compile with three kernel-proved lemmas in hand, after 11h34m.
# Fix is in two parts: dedup by declaration NAME in the store, and a
# within-batch name guard plus withholding of records that depend on a
# superseded def body (a lemma proved under def version A is not valid
# under version B). Cannot manufacture a solve — it only removes
# declarations from a header. See tests/test_store_feedback_dup_def.py.
# Re-frozen 2026-08-23 (fix the success-path record + write the cell) for
# a PROVENANCE defect found while auditing the
# putnam_2020_a2 solve. The verified return recorded neither the store lemmas
# the proof used nor the header it was compiled against, so `p2020a2_dag_v3`
# produced a row whose `assembled_proof` cites 2 defs and 9 `aux_*` lemmas that
# appear nowhere in it: they reached the header via `_surface_store_lemmas`
# AFTER the theory was abandoned at t=29677s. The audited artifact had to be
# rebuilt from the TRACE. The verified return now carries `lemma_store_stats`,
# `lemma_store_kept`, `prove_retry_log` (all three already recorded on the
# FAILURE path) and the new `verified_header`. Recording only — no control flow,
# no compile, no verdict is touched, so it cannot manufacture a solve. See
# tests/test_solve_provenance.py.
# Re-frozen 2026-08-25 (fix the header blindness bug)
# because the prove step now tells the model what is already in scope. The
# lemma header has ALWAYS been `prelude + defs + proved siblings + stmt`, but
# `prove_lemma_prompt` was handed only the statement, the last Lean error and
# BM25 Mathlib names — so the model could not cite what the compiler could
# already see. MEASURED on p1963a2_dag_v4: `aux_odd_isRelPrime_two` was proved
# at t=15007s and spliced into the header, after which the run failed
# `aux_isRelPrime_two_three` (one application of it) across seven attempts and
# two theory rounds, guessed the Coprime/IsRelPrime bridge six wrong ways, and
# never found `Nat.coprime_iff_isRelPrime` — roughly six hours of a
# sixteen-hour run spent re-deriving what was already in the file. The diff inside
# `attempt_dag_proof` is two statements: compute `declarations_in_scope` of the
# text the compile already uses, and pass it to the prompt. No control flow, no
# extra compile, no LLM call, and no verdict is touched — a listing can only
# change what the model WRITES, and the kernel still adjudicates every word of
# it. See tests/test_scope_visibility.py.
    "attempt_dag_proof":
        "c1f9cdb2ac48d6684b19b4dad14e4075c4d0a69874ee6807fac8417fcecc0c26",
    "_apply_repairs":
        "370dda0844afcc976fec8147d413cf01b8a28f778c46634d12bf28e2dde36f86",
}


# ---- 1. control flow frozen --------------------------------------------------

def test_attempt_dag_proof_source_frozen():
    h = hashlib.sha256(inspect.getsource(attempt_dag_proof).encode()).hexdigest()
    assert h == _FROZEN["attempt_dag_proof"], (
        "attempt_dag_proof source changed — any edit to the live control "
        "flow requires a deliberate hash re-baseline "
        "in this test")


def test_apply_repairs_source_frozen():
    h = hashlib.sha256(inspect.getsource(_apply_repairs).encode()).hexdigest()
    assert h == _FROZEN["_apply_repairs"]


# ---- 2. default engine legacy ------------------------------------------------

def test_repair_engine_default_is_legacy():
    assert RepairEngine().mode == "legacy"


def test_run_dag_repair_engine_defaults_legacy():
    src = (ROOT / "src" / "eval" / "run_dag.py").read_text(encoding="utf-8")
    m = re.search(r'--dag-repair-engine.*?default="(\w+)"', src, re.S)
    assert m and m.group(1) == "legacy"


# ---- 3. scheduler hard-gated -------------------------------------------------

def test_scheduler_engine_raises():
    with pytest.raises(NotImplementedError):
        RepairEngine("scheduler")


def test_run_dag_scheduler_is_optin_and_shadow_maps_to_legacy():
    src = (ROOT / "src" / "eval" / "run_dag.py").read_text(encoding="utf-8")
    # Scheduler execution is selected explicitly, never implicitly…
    assert 'args.dag_repair_engine == "scheduler"' in src
    # …and shadow NEVER executes: it observes while legacy runs.
    assert 'else "legacy"' in src


# ---- 4. Phase 3 flags default disabled --------------------------------------

def test_leaf_solver_config_defaults_all_false():
    cfg = LeafSolverConfig()
    assert not cfg.any_enabled
    assert not (cfg.proof_prior or cfg.template_tactics
                or cfg.dependency_exploration or cfg.llm_local_fallback)


def test_run_dag_phase3_flags_are_store_true():
    src = (ROOT / "src" / "eval" / "run_dag.py").read_text(encoding="utf-8")
    for flag in ("--dag-proof-prior", "--dag-template-tactics",
                 "--dag-dependency-exploration", "--dag-llm-local-fallback"):
        m = re.search(re.escape(flag) + r'".*?action="(\w+)"', src, re.S)
        assert m and m.group(1) == "store_true", flag


# ---- 5. disabled leaf solvers not imported/instantiated ----------------------

def test_disabled_portfolio_empty_and_imports_nothing(monkeypatch):
    for m in ("search.proof_prior", "search.template_tactics",
              "search.dependency_explorer"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    assert build_leaf_solvers(LeafSolverConfig()) == []
    for m in ("search.proof_prior", "search.template_tactics",
              "search.dependency_explorer"):
        assert m not in sys.modules

