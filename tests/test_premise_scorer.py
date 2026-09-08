"""Premise scorer unit test — uses a mock LLM; runs without any API key.

Asserts: a 'relevant' premise scores higher than an 'irrelevant' one in the
returned numpy array, and that caching works (second call doesn't re-query).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_relevant_scores_higher(monkeypatch, tmp_path):
    # Redirect cache + log to tmp so the test is hermetic.
    monkeypatch.setattr("policy.premise_scorer.CACHE_PATH", tmp_path / "scores.sqlite")
    monkeypatch.setattr("policy.premise_scorer.LOG_PATH", tmp_path / "log.jsonl")
    from policy.premise_scorer import PremiseScorer, Premise

    call_count = {"n": 0}

    def fake_llm(prompt: str) -> str:
        call_count["n"] += 1
        # Mock: looks for "succ" in the premise list, scores those high.
        # Returns a JSON array of integers, one per premise.
        # We don't actually parse the prompt; we look at how many "{n}." lines appear.
        import re
        lines = re.findall(r"^\d+\.\s+(\S+)", prompt, flags=re.M)
        scores = [9 if "succ" in name.lower() else 1 for name in lines]
        return str(scores)

    scorer = PremiseScorer(llm_call=fake_llm, batch_size=8)
    goal = "n : Nat\n  ⊢ n.succ + 0 = n + 1"
    premises = [
        Premise("Nat.succ_eq_add_one", "∀ n, Nat.succ n = n + 1"),
        Premise("Real.sqrt_pos", "∀ x, 0 < Real.sqrt x ↔ 0 < x"),
        Premise("Nat.succ_pos", "∀ n, 0 < Nat.succ n"),
        Premise("List.length_append", "..."),
    ]
    scores = scorer.score(goal, premises)
    assert scores.shape == (4,)
    # Relevant premises (succ_*) must outrank irrelevant ones.
    assert scores[0] > scores[1]
    assert scores[2] > scores[3]
    assert call_count["n"] == 1, "expected one batched LLM call"

    # Second call: same goal+premises -> all cache hits, no new LLM call.
    scores2 = scorer.score(goal, premises)
    assert np.allclose(scores2, scores)
    assert call_count["n"] == 1, "cache should suppress repeat LLM call"


def test_parse_handles_bad_output(monkeypatch, tmp_path):
    monkeypatch.setattr("policy.premise_scorer.CACHE_PATH", tmp_path / "scores.sqlite")
    monkeypatch.setattr("policy.premise_scorer.LOG_PATH", tmp_path / "log.jsonl")
    from policy.premise_scorer import PremiseScorer, Premise

    def bad_llm(prompt: str) -> str:
        return "I am refusing to follow the format. Sorry."

    scorer = PremiseScorer(llm_call=bad_llm, batch_size=8)
    scores = scorer.score("goal", [Premise("foo"), Premise("bar")])
    # Parse failure -> all zeros, but no crash.
    assert scores.shape == (2,)
    assert (scores == 0).all()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
