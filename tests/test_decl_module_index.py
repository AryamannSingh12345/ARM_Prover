"""Declaration→module resolver (b5_bare_v4 regression: the LLM guessed
`Data.Fin.Basic` for a betweenness lemma while the local declaration
graph knew the true module all along).

The resolution tests hit the real `data/mathlib_graph/nodes.jsonl` and
skip when it is absent (the graph archive is gitignored).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.decl_module_index import (  # noqa: E402
    resolve_decl_module,
    unknown_names,
    _NODES,
)

needs_graph = pytest.mark.skipif(
    not _NODES.exists(), reason="mathlib_graph archive not present")


def test_unknown_names_extraction():
    text = ("f.lean:30:8: error(lean.unknownIdentifier): Unknown "
            "identifier `mem_segment_iff_wbtw.mp`\n"
            "f.lean:31:8: error: Unknown constant `Wbtw`\n"
            "f.lean:32:8: error(lean.unknownIdentifier): Unknown "
            "identifier `Wbtw`")
    assert unknown_names(text) == ["mem_segment_iff_wbtw.mp", "Wbtw"]
    assert unknown_names("") == []
    assert unknown_names("error: unsolved goals") == []


@needs_graph
def test_resolves_the_v4_killers():
    assert resolve_decl_module("mem_segment_iff_wbtw") == \
        "Mathlib.Analysis.Convex.Between"
    assert resolve_decl_module("Wbtw") == "Mathlib.Analysis.Convex.Between"
    assert resolve_decl_module("dist_add_dist_eq_iff") == \
        "Mathlib.Analysis.Convex.StrictConvexBetween"


@needs_graph
def test_accessor_suffix_stripped():
    # Errors report the name as written, accessor and all.
    assert resolve_decl_module("mem_segment_iff_wbtw.mp") == \
        "Mathlib.Analysis.Convex.Between"
    assert resolve_decl_module("dist_add_dist_eq_iff.mp") == \
        "Mathlib.Analysis.Convex.StrictConvexBetween"


@needs_graph
def test_hallucinated_name_resolves_to_none():
    # A name that does not exist in the pin must add NO import.
    # (Fun fact: the first candidate for this test,
    # `dist_add_dist_of_mem_segment`, turned out to be a REAL theorem
    # in Mathlib.Analysis.Normed.Affine.Convex — v4's model reached
    # for genuine API twice over and only the import set failed it.)
    assert resolve_decl_module("aux_totally_made_up_lemma_xyz") is None
    assert resolve_decl_module("") is None


@needs_graph
def test_v4_second_reach_also_resolves():
    assert resolve_decl_module("dist_add_dist_of_mem_segment") == \
        "Mathlib.Analysis.Normed.Affine.Convex"
