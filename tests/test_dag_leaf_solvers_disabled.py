"""Phase 3 scaffold safety — disabled-by-default guarantee.

When no --dag-* leaf-solver flag is set, the portfolio must be empty AND
none of the candidate-generation modules may be imported or instantiated.
This is the contract that keeps a legacy run byte-identical: no new code
executes, prompts unchanged, verifier/probe counts unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.dag.solvers import (  # noqa: E402
    LeafSolverConfig, build_leaf_solvers,
)


def test_default_config_builds_empty_portfolio():
    assert LeafSolverConfig().any_enabled is False
    assert build_leaf_solvers(LeafSolverConfig()) == []


def test_disabled_build_imports_no_candidate_modules(monkeypatch):
    # Drop any already-imported candidate modules, then build a disabled
    # portfolio and assert none were (re)imported.
    candidate_mods = [
        "search.proof_prior", "search.template_tactics",
        "search.dependency_explorer", "search.state_features",
    ]
    for m in candidate_mods:
        monkeypatch.delitem(sys.modules, m, raising=False)
    build_leaf_solvers(LeafSolverConfig())            # all flags off
    for m in candidate_mods:
        assert m not in sys.modules, f"{m} imported by a disabled portfolio"


def test_enabled_flag_constructs_that_provider():
    cfg = LeafSolverConfig(template_tactics=True)
    provs = build_leaf_solvers(cfg)
    assert [p.name for p in provs] == ["template_tactics"]


def test_provider_order_is_prior_template_dependency():
    cfg = LeafSolverConfig(proof_prior=True, template_tactics=True,
                           dependency_exploration=True)
    assert [p.name for p in build_leaf_solvers(cfg)] == [
        "proof_prior", "template_tactics", "dependency_exploration"]


def test_llm_local_fallback_alone_builds_no_deterministic_provider():
    # the llm fallback is not a deterministic provider; portfolio is empty
    # of providers but the flag still counts as "enabled" for wiring.
    cfg = LeafSolverConfig(llm_local_fallback=True)
    assert cfg.any_enabled is True
    assert build_leaf_solvers(cfg) == []


# ---- real-provider regression (the fake-provider tests missed a crash) ------

def test_real_providers_use_correct_move_field(tmp_path):
    """The ablation run crashed with `'ProofPriorMove' object has no attribute
    'tactic'` — the providers must read `.tactic_template`. Exercise the REAL
    proof_prior + dependency providers (fakes hid this) and assert they map
    moves to CandidateMoves without AttributeError. Skips if the mined prior
    file is absent."""
    from search.dag.solvers import SolverContext
    from search.dag.model import ObligationState
    prior = (Path(__file__).resolve().parents[1] / "data" / "proof_prior"
             / "combined_proof_prior_moves.jsonl")
    if not prior.exists():
        import pytest
        pytest.skip("mined proof-prior file not present")
    cfg = LeafSolverConfig(proof_prior=True, dependency_exploration=True,
                           proof_prior_path=str(prior))
    provs = build_leaf_solvers(cfg)
    ob = ObligationState(id="h", goal_text="5 * (x : Int) = y",
                         dependencies=())
    ctx = SolverContext(theorem_header="theorem t : True",
                        premises=("Nat.mul_comm",), extra={})
    for p in provs:
        cands = p.candidates(ob, ctx)          # must NOT raise
        assert isinstance(cands, list)
        for c in cands:
            assert isinstance(c.tactic, str) and c.tactic  # non-empty tactic
