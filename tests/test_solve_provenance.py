"""A SOLVE must record what it depended on.

Regression for the defect the `putnam_2020_a2` audit found. The failure
path (`_final`) recorded `lemma_store_kept`, `lemma_store_stats` and
`prove_retry_log`; the verified path did not, and NOTHING recorded the
header the proof was compiled against. So the row for a solve carried an
`assembled_proof` citing names that existed only in that header.

MEASURED on `p2020a2_dag_v3` (solved, 31675s): the proof cites two defs
and nine `aux_*` lemmas that reached the header via `_surface_store_lemmas`
AFTER the theory was abandoned at t=29677s. `abduced_lemmas` and
`lemma_store_kept` were both empty, so `scripts/show_solution.py` emitted a
file that died on unknown identifiers and the audited artifact had to be
rebuilt from the trace. A solve that cannot be re-verified from its own
record is not an auditable result.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from search.dag.lemma_store import LemmaStore                 # noqa: E402
from search import proof_dag                                  # noqa: E402

DEF = "def aux_weighted (m n : ℕ) : ℕ := 0"
STMT = "lemma aux_weighted_zero (m : ℕ) : aux_weighted m 0 = 0"
DECL = STMT + " := by rfl"

HEADER = "import Mathlib\n\ntheorem tgt (n : ℕ) : n + 0 = n"
SKETCH = ('{"haves": [{"id":"h1","type":"n + 0 = n","tactic":"simp",'
          '"depends":[]}], "closer":"exact h1"}')


def _solve_after_surfacing():
    """Fail attempt 1, succeed once the store lemma is in the header.

    That is the shape of the measured run: the lemmas exist only because
    an abandoned theory left them in the store.
    """
    store = LemmaStore()
    store.put(STMT, "aux_weighted_zero", DECL, "import Mathlib", defs=[DEF])
    headers: list[str] = []

    def verify(header, body):
        headers.append(header)
        if "aux_weighted_zero" in header:
            return {"ok": True, "errors": None, "body_line_offset": 1}
        return {"ok": False, "errors": "error: nope", "body_line_offset": 1}

    res = proof_dag.attempt_dag_proof(
        HEADER, sketch_llm_call=lambda s, u: SKETCH, verify_fn=verify,
        probe_fn=lambda h, b: {"ok": True, "errors": None,
                               "body_line_offset": 1},
        sketch_attempts=2, repair_rounds=0,
        lemma_store=store, store_feedback=True)
    return res, headers


def test_solve_records_the_header_it_was_verified_against():
    res, headers = _solve_after_surfacing()
    assert res.verified, "fixture did not reach the verified path"
    assert res.verified_header, "a solve recorded no header"
    assert res.verified_header == headers[-1], \
        "recorded header is not the one the verifier accepted"
    assert DECL in res.verified_header
    assert DEF in res.verified_header, \
        "the def the lemma needs must travel with it"


def test_solve_records_the_store_lemmas_it_used():
    res, _ = _solve_after_surfacing()
    assert res.lemma_store_kept == [DECL], \
        "a solve must name the lemmas it depended on"
    assert res.lemma_store_stats is not None


def test_row_reconstructs_without_the_bench_file():
    """`show_solution.reconstruct` must prefer the recorded header.

    No bench directory is touched here: that is the point. A row carrying
    `verified_header` is self-contained, which is what makes step 3 of the
    audit standard (independent re-compilation) executable at all.
    """
    from show_solution import reconstruct
    res, _ = _solve_after_surfacing()
    row = {"id": "tgt", "assembled_proof": res.assembled_proof,
           "verify_imports": "import Mathlib",
           "verified_header": res.verified_header}
    src = reconstruct(row)
    assert src is not None
    assert DEF in src and DECL in src
    assert "theorem tgt" in src
    assert src.endswith("\n")


def test_failure_row_still_has_no_verified_header():
    """Only a SOLVE may claim a verified header."""
    store = LemmaStore()
    store.put(STMT, "aux_weighted_zero", DECL, "import Mathlib", defs=[DEF])
    res = proof_dag.attempt_dag_proof(
        HEADER, sketch_llm_call=lambda s, u: SKETCH,
        verify_fn=lambda h, b: {"ok": False, "errors": "error: nope",
                                "body_line_offset": 1},
        probe_fn=lambda h, b: {"ok": True, "errors": None,
                               "body_line_offset": 1},
        sketch_attempts=1, repair_rounds=0,
        lemma_store=store, store_feedback=True)
    assert not res.verified
    assert res.verified_header is None
    # the failure path kept recording these all along
    assert res.lemma_store_kept == [DECL]
