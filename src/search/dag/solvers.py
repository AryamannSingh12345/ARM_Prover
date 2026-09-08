"""Phase 3 scaffold — deterministic leaf-candidate providers (OPT-IN).

Makes the step pipeline's deterministic candidate machinery (proof prior,
tactic templates, dependency exploration) available to the DAG leaf
solver — but entirely behind explicit flags and NOT yet wired into the
live `attempt_dag_proof` loop. This module only assembles the provider
portfolio; wiring is Phase 3 proper, gated behind the same review.

Hard contract (tested in `test_dag_leaf_solvers_disabled.py`): when every
flag is off, `build_leaf_solvers` returns an EMPTY portfolio and imports
NONE of the candidate-generation modules — no new code executes, prompts
are unchanged, verifier/probe counts are unchanged. Modules are imported
lazily inside each enabled branch precisely so a disabled run cannot even
load them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .model import CandidateMove, ObligationState


@dataclass(slots=True)
class LeafSolverConfig:
    """All default False: a legacy run builds an empty portfolio."""
    proof_prior: bool = False
    template_tactics: bool = False
    dependency_exploration: bool = False
    llm_local_fallback: bool = False
    proof_prior_path: str | None = None

    @property
    def any_enabled(self) -> bool:
        return (self.proof_prior or self.template_tactics
                or self.dependency_exploration or self.llm_local_fallback)


class LeafCandidateProvider(Protocol):
    name: str

    def candidates(self, obligation: ObligationState,
                   context: "SolverContext") -> list[CandidateMove]:
        ...


@dataclass(slots=True)
class SolverContext:
    """Read-only inputs a provider may consult. No Lean/LLM handles here —
    providers are deterministic; verification stays with the caller."""
    theorem_header: str
    premises: tuple[str, ...] = ()
    extra: dict[str, Any] = None  # type: ignore[assignment]


# ---- providers (thin adapters over existing deterministic modules) ----------
# Each is constructed ONLY when its flag is on; its heavy imports live
# inside __init__/candidates so a disabled portfolio loads nothing.

class _ProofPriorProvider:
    name = "proof_prior"

    def __init__(self, cfg: LeafSolverConfig):
        self._cfg = cfg

    def candidates(self, obligation, context):
        from search.state_features import extract_state_features
        from search.proof_prior import ProofPriorIndex
        from pathlib import Path
        path = self._cfg.proof_prior_path
        if not path or not Path(path).exists():
            return []
        idx = ProofPriorIndex.load(Path(path))
        feats = extract_state_features(obligation.goal_text, proof_prefix=[])
        return [CandidateMove(source=self.name, tactic=m.tactic_template,
                              cost=0.5,
                              metadata={"tactic_class": getattr(
                                  m, "tactic_class", "")})
                for m in idx.suggest(feats)[:12]]


class _TemplateProvider:
    name = "template_tactics"

    def candidates(self, obligation, context):
        from search.state_features import extract_state_features
        from search.template_tactics import generate_template_tactics
        feats = extract_state_features(obligation.goal_text, proof_prefix=[])
        out = generate_template_tactics(
            feats, list(context.premises), goal_text=obligation.goal_text)
        return [CandidateMove(source=self.name, tactic=t, cost=float(c),
                              metadata=dict(meta))
                for (t, c, meta) in out]


class _DependencyProvider:
    name = "dependency_exploration"

    def candidates(self, obligation, context):
        from search.state_features import extract_state_features
        from search.dependency_explorer import dependency_exploration_candidates
        from search.premise_retrieval import _load_graph
        feats = extract_state_features(obligation.goal_text, proof_prefix=[])
        moves = dependency_exploration_candidates(
            feats, _load_graph(), list(context.premises))
        return [CandidateMove(source=self.name, tactic=m.tactic_template,
                              cost=0.6, metadata={}) for m in moves]


def build_leaf_solvers(
    cfg: LeafSolverConfig,
) -> list[LeafCandidateProvider]:
    """Assemble the enabled providers, in cheapest-sound-first order.
    Returns [] (and imports nothing) when every flag is off."""
    if not cfg.any_enabled:
        return []
    providers: list[LeafCandidateProvider] = []
    if cfg.proof_prior:
        providers.append(_ProofPriorProvider(cfg))
    if cfg.template_tactics:
        providers.append(_TemplateProvider())
    if cfg.dependency_exploration:
        providers.append(_DependencyProvider())
    # llm_local_fallback is handled by the existing repair path, not a
    # deterministic provider; the flag is recorded for Phase-3 wiring.
    return providers
