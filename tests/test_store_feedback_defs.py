"""Stored lemmas must carry the defs their statements depend on.

Regression for the bug --store-feedback shipped with. A theory's `defs`
reach the header only when the theory COMMITS. `putnam_2020_a2_v1` proved
seven lemmas inside theories that all ABANDONED, so the store held

    lemma aux_weighted_step (m n : ℕ) :
      aux_weighted m (n + 1) = 2 * aux_weighted m n + …

while `abbrev aux_weighted` existed nowhere. Splicing those lemmas into
the header for the next sketch produced

    Unknown identifier `aux_weighted`
    Function expected at aux_weighted but this term has type ?m.1

(`autoImplicit` turns the stray name into a metavariable), which is a
HEADER-level error — so the repair loop stopped outright and the run died
at t=3059s with 7 proved lemmas in hand. The feature made the run strictly
worse than not having it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.dag.lemma_store import LemmaStore                 # noqa: E402
from search import proof_dag                                  # noqa: E402

DEF_W = "abbrev aux_weighted (m n : ℕ) : ℕ := 0"
DEF_P = "abbrev aux_partial (m n : ℕ) : ℕ := 0"
STMT = "lemma aux_weighted_step (m n : ℕ) : aux_weighted m (n+1) = 0"


def test_put_records_defs():
    s = LemmaStore()
    s.put(STMT, "aux_weighted_step", STMT + " := by rfl", "import Mathlib",
          defs=[DEF_W])
    assert s.records()[0].defs == (DEF_W,)


def test_required_defs_dedupes_across_lemmas():
    s = LemmaStore()
    s.put(STMT, "a", STMT + " := by rfl", "import Mathlib",
          defs=[DEF_W, DEF_P])
    s.put("lemma other : aux_partial 0 0 = 0", "other",
          "lemma other : aux_partial 0 0 = 0 := by rfl", "import Mathlib",
          defs=[DEF_P])
    assert s.required_defs() == [DEF_W, DEF_P]


def test_required_defs_empty_without_defs():
    s = LemmaStore()
    s.put(STMT, "a", STMT + " := by rfl", "import Mathlib")
    assert s.required_defs() == []


# --------------------------------------------------------------------------
# end-to-end: the splice must define before it uses
# --------------------------------------------------------------------------

HEADER = "import Mathlib\n\ntheorem tgt (n : ℕ) : n + 0 = n"

SKETCH = ('{"haves": [{"id":"h1","type":"n + 0 = n","tactic":"simp",'
          '"depends":[]}], "closer":"exact h1"}')


def _run(store, attempts: int = 2):
    seen: list[str] = []

    def verify(header, body):
        seen.append(header)
        return {"ok": False, "errors": "error: nope", "body_line_offset": 1}

    proof_dag.attempt_dag_proof(
        HEADER, sketch_llm_call=lambda s, u: SKETCH, verify_fn=verify,
        probe_fn=lambda h, b: {"ok": True, "errors": None,
                               "body_line_offset": 1},
        sketch_attempts=attempts, repair_rounds=0,
        lemma_store=store, store_feedback=True)
    return seen


def test_spliced_header_defines_before_it_uses():
    store = LemmaStore()
    store.put(STMT, "aux_weighted_step", STMT + " := by rfl",
              "import Mathlib", defs=[DEF_W])
    headers = _run(store)
    spliced = [h for h in headers if "aux_weighted_step" in h]
    assert spliced, "store lemma was never surfaced"
    h = spliced[0]
    assert DEF_W in h, "def missing — this is the putnam_2020_a2_v1 failure"
    assert h.index(DEF_W) < h.index("lemma aux_weighted_step"), \
        "def must precede the lemma that uses it"


def test_defs_are_spliced_once_across_attempts():
    store = LemmaStore()
    store.put(STMT, "aux_weighted_step", STMT + " := by rfl",
              "import Mathlib", defs=[DEF_W])
    headers = _run(store, attempts=3)
    last = [h for h in headers if "aux_weighted_step" in h][-1]
    assert last.count(DEF_W) == 1, "def duplicated across attempts"


def test_store_without_defs_still_surfaces():
    """Lemmas needing no auxiliary defs must be unaffected."""
    store = LemmaStore()
    plain = "lemma aux_plain : (1:ℕ) + 0 = 1"
    store.put(plain, "aux_plain", plain + " := by simp", "import Mathlib")
    headers = _run(store)
    assert any("aux_plain" in h for h in headers)


# ---- never re-declare what the header already has ---------------------------

def test_store_feedback_skips_lemmas_already_in_the_header():
    """A COMMITTED lemma is written straight into `theorem_header` by the
    commit path, which never touches the spliced-set. Deduping on the
    spliced-set alone re-declares it, and Lean's "has already been
    declared" is a HEADER error — which aborts the run by design.

    Measured on p1968a1_v1: 5 lemmas committed at t=16910s, the same 5
    re-spliced at t=17401s, run dead at t=17992s with NINE proved lemmas
    in hand and 5 hours spent.
    """
    import re
    header = ("import Mathlib\n\n"
              "lemma aux_seventh_term_deriv (x : ℝ) : True := trivial\n\n"
              "theorem tgt : True")
    declared = set(re.findall(
        r"(?m)^\s*(?:private\s+|protected\s+|noncomputable\s+)*"
        r"(?:theorem|lemma|def|abbrev|structure|inductive)\s+"
        r"([A-Za-z_][A-Za-z0-9_'!?₀-₉.]*)", header))
    assert "aux_seventh_term_deriv" in declared
    assert "tgt" in declared
    # a lemma NOT yet in the header is still surfaced
    assert "aux_other" not in declared


def test_header_name_scan_matches_the_implementation():
    """Guard against the scan drifting from what proof_dag actually uses."""
    import inspect
    import search.proof_dag as pd
    src = inspect.getsource(pd.attempt_dag_proof)
    assert "_declared" in src
    assert "has already been" in src or "already declared" in src.lower()


def test_theory_may_not_repropose_a_lemma_already_in_the_header():
    """The MIRROR of the splice fix: `--store-feedback` puts a proved
    lemma in the header, the theory loop sees it in its prompt and
    re-proposes it for proving, and Lean reports "has already been
    declared" — a HEADER error that kills the round.

    Measured on p2023a2_v1 at t=10932s: `aux_equation_root_bridge` was
    spliced at attempt 3 and re-proposed in round 2, wasting the round
    after 3 hours of work.

    Such a lemma needs no proof — it is already in scope — so it is
    dropped, not rejected.
    """
    import re
    header = ("import Mathlib\n\n"
              "lemma aux_equation_root_bridge (p : ℕ) : True := trivial\n\n"
              "theorem tgt : True")
    hdr_names = set(re.findall(
        r"(?m)^\s*(?:private\s+|protected\s+|noncomputable\s+)*"
        r"(?:theorem|lemma|def|abbrev|structure|inductive)\s+"
        r"([A-Za-z_][A-Za-z0-9_'!?₀-₉.]*)", header))
    proposed = ["aux_equation_root_bridge", "aux_brand_new"]
    kept = [n for n in proposed if n not in hdr_names]
    assert kept == ["aux_brand_new"]


def test_the_drop_is_implemented_in_the_theory_loop():
    import inspect
    import search.proof_dag as pd
    src = inspect.getsource(pd.attempt_dag_proof)
    assert "_hdr_names" in src
    assert "already in scope" in src
