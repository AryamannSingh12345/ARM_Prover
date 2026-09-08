"""Run-scoped lemma store + failure-class retry budgets.

Two properties matter most and are tested end-to-end against the real
`attempt_dag_proof` loop rather than only at unit level:

1. **Legacy-when-off** — with both features disabled the loop behaves
   exactly as before (same number of prove calls, same ledger wording,
   lemmas discarded on abandonment).
2. **Survival** — with the store on, lemmas the kernel accepted before an
   abandonment are still available to the NEXT theory, so they are not
   re-proved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.dag import retry_budget as rb            # noqa: E402
from search.dag.lemma_store import (                 # noqa: E402
    LemmaStore, imports_of, statement_key,
)
from search import proof_dag                         # noqa: E402


# --------------------------------------------------------------------------
# LemmaStore unit level
# --------------------------------------------------------------------------

PRELUDE = "import Mathlib.Data.Nat.Digits.Defs\nimport Mathlib.Order.Basic\n"
STMT = "lemma aux_cbrt_pos (n : ℕ) (hn : 0 < n) : (0:ℝ) < (n:ℝ) ^ ((1:ℝ)/3)"
DECL = STMT + " := by\n  positivity"


def test_put_then_get_round_trip():
    s = LemmaStore()
    s.put(STMT, "aux_cbrt_pos", DECL, PRELUDE)
    rec = s.get(STMT, PRELUDE)
    assert rec is not None
    assert rec.decl == DECL and rec.name == "aux_cbrt_pos"


def test_key_is_whitespace_insensitive():
    s = LemmaStore()
    s.put(STMT, "aux_cbrt_pos", DECL, PRELUDE)
    spaced = STMT.replace(" : ", "  :  ").replace("(n : ℕ)", "(n  :  ℕ)")
    assert s.get(spaced, PRELUDE) is not None
    assert statement_key(spaced) == statement_key(STMT)


def test_miss_on_unknown_statement():
    s = LemmaStore()
    s.put(STMT, "aux_cbrt_pos", DECL, PRELUDE)
    assert s.get("lemma aux_other : True", PRELUDE) is None
    assert s.stats()["misses"] == 1


def test_rename_is_a_miss():
    """The declared name is part of the key: a renamed lemma must be
    re-proved, because the proposing theory's leaf tactics call it by
    the new name."""
    s = LemmaStore()
    s.put(STMT, "aux_cbrt_pos", DECL, PRELUDE)
    renamed = STMT.replace("aux_cbrt_pos", "aux_cube_root_pos")
    assert s.get(renamed, PRELUDE) is None


def test_import_shrink_rejects_the_hit():
    """Imports mutate live during a problem (refresh_imports_call). A
    lemma proved under imports no longer in scope must not be handed
    back."""
    s = LemmaStore()
    s.put(STMT, "aux_cbrt_pos", DECL, PRELUDE)
    assert s.get(STMT, "import Mathlib.Order.Basic\n") is None
    assert s.stats()["import_rejects"] == 1


def test_import_growth_still_hits():
    s = LemmaStore()
    s.put(STMT, "aux_cbrt_pos", DECL, PRELUDE)
    bigger = PRELUDE + "import Mathlib.Analysis.SpecialFunctions.Pow.Real\n"
    assert s.get(STMT, bigger) is not None


def test_first_proof_wins():
    s = LemmaStore()
    s.put(STMT, "aux_cbrt_pos", DECL, PRELUDE)
    s.put(STMT, "aux_cbrt_pos", DECL + " -- second", PRELUDE)
    assert s.get(STMT, PRELUDE).decl == DECL
    assert len(s) == 1


def test_imports_of_parses_import_lines_only():
    assert imports_of("import A.B\nimport C\nlemma foo : True := trivial") == (
        frozenset({"A.B", "C"}))


def test_stats_shape():
    s = LemmaStore()
    s.put(STMT, "aux_cbrt_pos", DECL, PRELUDE)
    s.get(STMT, PRELUDE)
    st = s.stats()
    assert st["size"] == 1 and st["hits"] == 1
    assert st["names"] == ["aux_cbrt_pos"]


# --------------------------------------------------------------------------
# retry_budget unit level
# --------------------------------------------------------------------------

def test_empty_or_forbidden_gets_one_attempt():
    cls = rb.classify_prove_failure(None, empty_or_forbidden=True)
    assert cls == [rb.EMPTY_OR_FORBIDDEN]
    assert rb.budget_for(cls) == 1


def test_static_errors_get_more_than_legacy():
    for err in ("error: Unknown identifier `le_of_pow_le_pow_left`",
                "error: unknown tactic",
                "error: unexpected token ';'"):
        cls = rb.classify_prove_failure(err)
        assert rb.budget_for(cls) > rb.LEGACY_ATTEMPTS, err


def test_unknown_tactic_is_not_other():
    cls = rb.classify_prove_failure("error: unknown tactic")
    assert rb.UNKNOWN_TACTIC in cls and "other" not in cls


def test_mathematical_failure_keeps_legacy_budget():
    cls = rb.classify_prove_failure("error: unsolved goals\n n : ℕ")
    assert rb.budget_for(cls) == rb.LEGACY_ATTEMPTS


def test_timeout_gets_one():
    cls = rb.classify_prove_failure(
        "maximum number of heartbeats (200000) has been reached")
    assert rb.budget_for(cls) == 1


def test_budget_takes_the_max_over_classes():
    """An unresolved name causing downstream open goals is worth the
    static budget, not the mathematical one."""
    cls = rb.classify_prove_failure(
        "error: Unknown identifier `foo`\nerror: unsolved goals")
    assert "unknown_identifier" in cls and "unsolved_goals" in cls
    assert rb.budget_for(cls) == rb.DEFAULT_BUDGETS["unknown_identifier"]


def test_budget_respects_hard_cap():
    assert rb.budget_for(["unknown_identifier"], {"unknown_identifier": 999}) \
        == rb.HARD_CAP


def test_unknown_class_falls_back_to_legacy():
    assert rb.budget_for(["something_new"]) == rb.LEGACY_ATTEMPTS


def test_ledger_hint_present_for_empty_and_absent_for_unknown_class():
    assert "never actually attempted" in rb.ledger_hint([rb.EMPTY_OR_FORBIDDEN])
    assert rb.ledger_hint(["something_new"]) == ""


# --------------------------------------------------------------------------
# End-to-end against the real theory loop
# --------------------------------------------------------------------------

HEADER = "import Mathlib\n\ntheorem tgt (n : ℕ) : n + 0 = n"

SKETCH_JSON = (
    '{"haves": [{"id": "h1", "type": "n + 0 = n", "tactic": "simp", '
    '"depends": []}], "closer": "exact h1"}'
)

# Two lemmas; the first proves, the second never does. That is the
# p25 shape: a theory that reaches `revision rounds exhausted` having
# kernel-proved part of itself.
THEORY_JSON = (
    '{"defs": [], "lemmas": ['
    '{"name": "aux_good", "statement": "lemma aux_good : (1:ℕ) + 0 = 1"},'
    '{"name": "aux_bad", "statement": "lemma aux_bad : (2:ℕ) + 0 = 2"}'
    '], "leaf_tactics": {"h1": "simp"}, "rationale": "r"}'
)


class _Harness:
    """Drives `attempt_dag_proof` through: sketch -> failing verify ->
    theory -> prove(good ok / bad fails) -> revision -> abandon.

    Records every prove request so a test can count how often a given
    lemma was re-proved.
    """

    def __init__(self, *, bad_error: str, bad_proof: str = "rfl",
                 partial_suffices: bool = False):
        self.bad_error = bad_error
        self.bad_proof = bad_proof
        #: Does the PROVED subset (`aux_good` alone) close the leaves?
        #: Default False — in this scenario the theory genuinely needs
        #: both lemmas, so the run abandons, which is what the store
        #: retention tests below are about. Set True to exercise the
        #: sufficiency-based partial commit instead.
        self.partial_suffices = partial_suffices
        self.prove_calls: list[str] = []
        self.theory_calls = 0

    def llm(self, system: str, user: str) -> str:
        if system == proof_dag.SKETCH_SYSTEM:
            return SKETCH_JSON
        if system == proof_dag.THEORY_SYSTEM:
            self.theory_calls += 1
            return THEORY_JSON
        if system == proof_dag.PROVE_LEMMA_SYSTEM:
            target = "aux_bad" if "aux_bad" in user else "aux_good"
            self.prove_calls.append(target)
            if target == "aux_good":
                return '{"proof": "rfl"}'
            return '{"proof": "%s"}' % self.bad_proof
        return '{"repairs": []}'

    def verify(self, header: str, body: str) -> dict:
        # Lemma gate: `aux_good` is accepted, `aux_bad` is refused with
        # the configured error. Any other verify (the main theorem) fails
        # so the loop keeps going into repair/theory.
        if header.rstrip().endswith("lemma aux_good : (1:ℕ) + 0 = 1"):
            return {"ok": True, "errors": None, "body_line_offset": 1}
        if header.rstrip().endswith("lemma aux_bad : (2:ℕ) + 0 = 2"):
            return {"ok": False, "errors": self.bad_error,
                    "body_line_offset": 1}
        return {"ok": False, "errors": "error: unsolved goals",
                "body_line_offset": 1}

    def probe(self, header: str, body: str) -> dict:
        # Satisfaction gate always passes: we are testing the PROVE step.
        #
        # EXCEPT the partial-commit probe, which is distinguishable: it
        # splices the PROVED subset for real (`aux_good` only), whereas
        # the round gate stubs every proposed lemma (`aux_good` AND
        # `aux_bad`). Whether the subset alone suffices is the question
        # that probe exists to ask, so the harness answers it explicitly.
        # NB the failure signal is timeout-shaped on purpose. `_satisfied`
        # treats an unattributable error ("error: unsolved goals" with no
        # `file:line:col:` marker) as "not our leaves' fault" and reports
        # SATISFIED; it fails closed only on a timeout. So a plain error
        # string here would silently assert the opposite of what we mean.
        if "aux_good" in header and "aux_bad" not in header:
            return {"ok": self.partial_suffices,
                    "errors": (None if self.partial_suffices
                               else "timeout after 600s"),
                    "body_line_offset": 1}
        return {"ok": True, "errors": None, "body_line_offset": 1}

    def run(self, **kw):
        return proof_dag.attempt_dag_proof(
            HEADER,
            sketch_llm_call=self.llm,
            verify_fn=self.verify,
            probe_fn=self.probe,
            sketch_attempts=1,
            repair_rounds=2,
            abduce_lemmas=True,
            abduce_mode="theory",
            abduce_theory_trigger="always",
            abduce_theory_rounds=1,
            **kw,
        )


def test_partial_commit_when_proved_subset_suffices():
    """The b5 case: the lemma that mattered proved, its scaffolding did not.

    With `aux_good` proved, `aux_bad` failed, and the probe confirming
    that `aux_good` alone closes the leaves, the theory must COMMIT the
    subset rather than discard a finished proof. (putnam_1967_b5 t=46563s
    discarded a kernel-proved crux exactly this way.)
    """
    h = _Harness(bad_error="error: unsolved goals", partial_suffices=True)
    res = h.run()
    # committed, so the ordinary abandonment marker is absent
    assert "theory_revision_exhausted" not in res.repair_errors
    # and the proved lemma reached the header
    assert any("aux_good" in d for d in res.abduced_lemmas)
    # the FAILED lemma must never be spliced
    assert not any("aux_bad" in d for d in res.abduced_lemmas)


def test_partial_commit_refused_when_subset_does_not_suffice():
    """If the proved subset does not close the leaves, abandon as before."""
    h = _Harness(bad_error="error: unsolved goals", partial_suffices=False)
    res = h.run()
    assert "theory_revision_exhausted" in res.repair_errors
    assert not res.abduced_lemmas


def test_legacy_discards_proved_lemmas_on_abandonment():
    """Baseline: with no store, `aux_good` is re-proved in the revision
    round and nothing survives the abandonment."""
    h = _Harness(bad_error="error: unsolved goals")
    res = h.run()
    assert not res.verified
    assert "theory_revision_exhausted" in res.repair_errors
    assert res.lemma_store_stats is None
    assert res.lemma_store_kept == []


def test_store_retains_proved_lemma_across_abandonment():
    """With the store on, `aux_good` survives the abandoned theory and
    appears in the result even though nothing committed."""
    store = LemmaStore()
    h = _Harness(bad_error="error: unsolved goals")
    res = h.run(lemma_store=store)
    assert not res.verified
    assert "theory_revision_exhausted" in res.repair_errors
    assert res.lemma_store_stats["size"] == 1
    assert res.lemma_store_stats["names"] == ["aux_good"]
    assert any("aux_good" in d for d in res.lemma_store_kept)


def test_store_prevents_reproving_across_theory_rounds():
    """The point of the store: `aux_good` is proved once, then served
    from cache on every later round."""
    store = LemmaStore()
    h = _Harness(bad_error="error: unsolved goals")
    h.run(lemma_store=store)
    assert h.prove_calls.count("aux_good") == 1
    assert store.stats()["hits"] >= 1


def test_store_survives_into_a_second_problem_attempt():
    """A store handed to a SECOND `attempt_dag_proof` call (what the
    per-problem owner does across sketch attempts) serves the lemma
    without any prove call at all."""
    store = LemmaStore()
    _Harness(bad_error="error: unsolved goals").run(lemma_store=store)
    h2 = _Harness(bad_error="error: unsolved goals")
    h2.run(lemma_store=store)
    assert h2.prove_calls.count("aux_good") == 0


def test_retry_budget_off_reasks_empty_response():
    """Legacy: an empty proof response is re-asked once (2 attempts)."""
    h = _Harness(bad_error="error: unsolved goals", bad_proof="")
    res = h.run()
    # 2 attempts per theory round, 2 rounds (initial + 1 revision).
    assert h.prove_calls.count("aux_bad") == 4
    assert res.prove_retry_log == []


def test_retry_budget_on_stops_reasking_empty_response():
    """With budgets on, an empty response is NOT re-asked: one attempt
    per round instead of two."""
    h = _Harness(bad_error="error: unsolved goals", bad_proof="")
    res = h.run(prove_retry_budget=rb.DEFAULT_BUDGETS)
    assert h.prove_calls.count("aux_bad") == 2
    assert any("empty_or_forbidden" in r for r in res.prove_retry_log)


def test_retry_budget_on_extends_static_errors():
    """A static error earns more than the legacy two attempts."""
    h = _Harness(bad_error="error: Unknown identifier `nope`")
    res = h.run(prove_retry_budget=rb.DEFAULT_BUDGETS)
    per_round = rb.DEFAULT_BUDGETS["unknown_identifier"]
    assert h.prove_calls.count("aux_bad") == per_round * 2
    assert any("unknown_identifier" in r for r in res.prove_retry_log)


def test_retry_budget_off_is_flat_two_for_static_errors():
    h = _Harness(bad_error="error: Unknown identifier `nope`")
    h.run()
    assert h.prove_calls.count("aux_bad") == 2 * 2


def test_retry_budget_off_keeps_legacy_ledger_wording():
    """Legacy-when-off extends to the prompt text: the revision ledger
    entry must be byte-identical to the pre-change wording, since it is
    fed to the model."""
    h = _Harness(bad_error="error: unsolved goals")
    res = h.run()
    assert res.prove_retry_log == []
    joined = " ".join(res.repair_errors)
    assert "theory_lemma_unproved" in joined
    # The legacy phrasing, not the class-specific one.
    assert "after 1 attempt(s)" not in joined


# ---- mechanical residue: a near-miss is not a mathematical failure ---------

def test_omega_failure_is_a_mechanical_residue():
    """`omega`/`ring`/`linarith` are only reached once the mathematical
    work is done, so their complaints mark an end-of-proof leftover."""
    from search.dag.retry_budget import (
        MECHANICAL_RESIDUE, budget_for, classify_prove_failure)
    err = ("error: omega could not prove the goal: a possible "
           "counterexample may satisfy the constraints d >= 0 c >= 0")
    classes = classify_prove_failure(err)
    assert MECHANICAL_RESIDUE in classes
    assert budget_for(classes) == 4


def test_ac_rearrangement_is_a_mechanical_residue():
    """The exact first failure of `aux_negative_binomial_prefix` on
    p2020a2_recognize_v2 — the single lemma between that run and a solved
    Putnam A2. A 37-line induction that ends on associativity is one
    `ring_nf` from closing, but it was classified `unsolved_goals` and
    given the mathematical budget of 2 while a typo gets 4."""
    from search.dag.retry_budget import (
        MECHANICAL_RESIDUE, budget_for, classify_prove_failure)
    err = ("error: unsolved goals m k n r : ℕ ⊢ "
           "1 + ∑ i ∈ Finset.range r, n.choose (i + 1) + n.choose (r + 1) = "
           "∑ i ∈ Finset.range r, n.choose (i + 1) + 1 + n.choose (r + 1)")
    classes = classify_prove_failure(err)
    assert MECHANICAL_RESIDUE in classes
    assert budget_for(classes) == 4


def test_a_genuinely_open_goal_keeps_the_mathematical_budget():
    """The detector must not simply promote every `unsolved goals`."""
    from search.dag.retry_budget import (
        MECHANICAL_RESIDUE, budget_for, classify_prove_failure)
    err = "error: unsolved goals ⊢ ∑ i ∈ Finset.range n, f i = 2 ^ n"
    classes = classify_prove_failure(err)
    assert MECHANICAL_RESIDUE not in classes
    assert budget_for(classes) == 2


def test_ac_detector_needs_the_same_terms_both_sides():
    from search.dag.retry_budget import is_ac_rearrangement
    assert is_ac_rearrangement("⊢ a + b + c = c + a + b")
    assert not is_ac_rearrangement("⊢ a + b = a + c")
    assert not is_ac_rearrangement("⊢ x = x")
    assert not is_ac_rearrangement("no turnstile here")


def test_ac_detector_respects_brackets():
    """The `+` inside `n.choose (i + 1)` must not become an operand."""
    from search.dag.retry_budget import is_ac_rearrangement
    assert is_ac_rearrangement(
        "⊢ 1 + f (i + 1) + g (r + 1) = f (i + 1) + 1 + g (r + 1)")
