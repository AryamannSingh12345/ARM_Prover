"""Regression tests for sufficiency-based PARTIAL COMMIT (2026-08-17).

Origin: putnam_1967_b5, t=46563s. The theory loop kernel-PROVED
`aux_weighted_transform` — the lemma the whole 15-hour run turned on —
and then discarded the entire theory, because three helper lemmas
written *only to support it* had failed. Ten proved lemmas sat unused
while the run continued for another two hours.

All-or-nothing asks "did every PROPOSED lemma prove?". The question that
decides the proof is "do the PROVED lemmas close the stuck leaves?".
Those come apart precisely when the model proposes scaffolding for a
lemma it then proves directly — which is the productive case.

Soundness property pinned here: a partial commit can never manufacture a
solve. It happens only when the ordinary satisfaction probe passes with
the proved declarations spliced for real, and the result still faces the
normal verify loop plus a final fresh compile.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.proof_dag import CLOSER_ID  # noqa: E402


def _partial_commit_block(src: str) -> str:
    """The partial-commit block, delimited STRUCTURALLY.

    A fixed character window silently stops covering the block when code
    is inserted (it did, when the unproved-lemma filter was added), so
    the slice runs from the section marker to the abandonment line that
    ends it.
    """
    i = src.index("PARTIAL COMMIT ON SUFFICIENCY")
    j = src.index('repair_errors.append("theory_revision_exhausted")', i)
    return src[i:j]


def test_partial_commit_is_reachable_in_source():
    """The commit path exists and is gated on a satisfaction probe."""
    src = (Path(__file__).resolve().parents[1]
           / "src" / "search" / "proof_dag.py").read_text(encoding="utf-8")
    assert "PARTIAL COMMIT ON SUFFICIENCY" in src
    tail = _partial_commit_block(src)
    # gated on the probe, not on the proposal
    assert "_satisfied(_pdefs + _pdecls, [], _ptactics)" in tail
    # and only commits inside the ok branch
    assert "if _ok:" in tail
    assert "insert_lemmas_before_decl" in tail


def test_partial_state_is_only_recorded_when_something_proved():
    """`_partial` is assigned only under `if proved_decls:`.

    Asserted on the enclosing region rather than on an exact adjacent
    string, so an explanatory comment between the two lines does not
    read as a behaviour change.
    """
    src = (Path(__file__).resolve().parents[1]
           / "src" / "search" / "proof_dag.py").read_text(encoding="utf-8")
    assigns = [m.start() for m in re.finditer(r"^\s*_partial = \(",
                                              src, re.M)]
    assert assigns, "no _partial assignment found"
    for pos in assigns:
        preceding = src[max(0, pos - 600):pos]
        assert "if proved_decls:" in preceding, (
            "a _partial assignment is not guarded by `if proved_decls:`")


def test_rejected_partial_is_traced_not_committed():
    src = (Path(__file__).resolve().parents[1]
           / "src" / "search" / "proof_dag.py").read_text(encoding="utf-8")
    assert 'stage="partial commit rejected"' in src
    # the fall-through still records the ordinary abandonment
    i = src.index('stage="partial commit rejected"')
    assert 'repair_errors.append("theory_revision_exhausted")' in src[i:i + 600]


def test_closer_pseudo_leaf_is_handled():
    """A closer-stuck theory must be able to partially commit too."""
    src = (Path(__file__).resolve().parents[1]
           / "src" / "search" / "proof_dag.py").read_text(encoding="utf-8")
    tail = _partial_commit_block(src)
    assert "CLOSER_ID in broken_ids" in tail
    assert "sketch.closer = _ptactics[CLOSER_ID]" in tail
    assert CLOSER_ID  # imported symbol is the one the loop uses


def test_partial_commit_records_provenance_like_full_commit():
    src = (Path(__file__).resolve().parents[1]
           / "src" / "search" / "proof_dag.py").read_text(encoding="utf-8")
    tail = _partial_commit_block(src)
    assert "abduced_lemma_decls.extend" in tail
    assert "lemma_library_path" in tail
    assert "(partial)" in tail
