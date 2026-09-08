"""`unknown tactic` must not mask the real error.

Lean emits `unknown tactic` as a COMPANION diagnostic when a nested
`by …` inside a `first | …` alternative fails to elaborate. The
informative error sits at a lower column on the same line. Real output
from putnam6_easy_v1 / putnam_1963_a2:

    20:59: error: unknown tactic
    20:55: error: unsolved goals
            ⊢ IsRelPrime 1 1

Consumers take `v[0]` from the per-segment list, and the artifact is
emitted first, so the repair loop was handed "unknown tactic" while
`⊢ IsRelPrime 1 1` — the whole content of the failure — was thrown away.
One lemma in putnam_easy8_sol_v1 burned four identical retries against
that; the class fired eight times across three unrelated lemmas.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.dag.diagnostics import (                          # noqa: E402
    attribute_errors, _rank_messages,
)

# One have spanning body lines 1-3, offset 17 -> file lines 18-20.
SEGMENTS = [("h_f1", 1, 3)]
OFFSET = 17

LIVE_ERRORS = (
    "C:\\x\\Try.lean:20:59: error: unknown tactic\n"
    "C:\\x\\Try.lean:20:55: error: unsolved goals\n"
    "f : \u2115 \u2192 \u2115\n"
    "\u22a2 IsRelPrime 1 1\n"
)


def test_informative_error_is_reported_first():
    out = attribute_errors(LIVE_ERRORS, SEGMENTS, OFFSET)
    assert "h_f1" in out
    assert "unsolved goals" in out["h_f1"][0]
    assert "IsRelPrime 1 1" in out["h_f1"][0]


def test_artifact_is_kept_but_demoted():
    """Demoted, never dropped — it may still help a human reading a trace."""
    out = attribute_errors(LIVE_ERRORS, SEGMENTS, OFFSET)
    assert any("unknown tactic" in m for m in out["h_f1"])
    assert "unknown tactic" not in out["h_f1"][0]


def test_lone_artifact_is_preserved():
    """A weak error beats no error."""
    errs = "C:\\x\\Try.lean:20:59: error: unknown tactic\n"
    out = attribute_errors(errs, SEGMENTS, OFFSET)
    assert out["h_f1"] == ["unknown tactic"]


def test_ranking_is_a_noop_without_artifacts():
    msgs = ["unsolved goals\n⊢ A", "type mismatch\nfoo"]
    assert _rank_messages(msgs) == msgs


def test_ranking_preserves_order_among_informative_messages():
    msgs = ["unknown tactic", "first real", "unknown tactic", "second real"]
    assert _rank_messages(msgs) == [
        "first real", "second real", "unknown tactic", "unknown tactic"]


def test_single_message_untouched():
    assert _rank_messages(["unknown tactic"]) == ["unknown tactic"]
    assert _rank_messages([]) == []


def test_only_bare_unknown_tactic_is_treated_as_an_artifact():
    """A message that merely mentions the phrase must not be demoted."""
    msgs = ["unknown tactic 'foo' in this context", "unsolved goals"]
    assert _rank_messages(msgs) == msgs
