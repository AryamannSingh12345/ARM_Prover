"""Heuristic proof-difficulty estimate + decomposition-quality gate (Point 4).

ARM's satisfaction gate only asks "would these lemmas IMPLY the stuck
obligation?" It does not ask "are they substantially EASIER to prove?"
So a model can propose a non-circular but *theorem-sized* lemma: the
satisfaction gate passes, then the prove phase faces almost the original
difficulty. The circularity guard catches exact restatements; it does not
catch near-restatements or disguised complexity preservation (a
generalisation, an argument reorder, a lemma containing 90% of the
conclusion).

This module adds a cheap, deterministic difficulty *proxy* over a goal's
surface syntax and a relative gate: a proposed theory should not contain a
lemma nearly as hard as the hardest obligation it discharges. It is a
heuristic, not an oracle — it never blocks a proof, only asks the model to
revise a theory that fails to reduce difficulty. No Lean, no network, no
problem-specific vocabulary (only generic mathematical structure markers).

Gate criterion (the robust `max` form of the review's inequality):

    max_i D(L_i) < factor · D(G)

where D(G) is the hardest stuck obligation. The `max` form is preferred
over `Σ_i w_i D(L_i) + C_assembly < D(G)`: a genuine decomposition into
many small lemmas can have a large *sum* while every piece is easy —
decomposition reduces the difficulty of the HARDEST node, not necessarily
the total. Catching "one lemma ≈ as hard as the goal" is exactly the
complexity-preservation failure we want to reject.

Extension point (Point 3 / proof-prior): `estimate_difficulty` takes an
optional `provability_prior` in [0,1] — a historical closure rate for
similarly-shaped obligations — which scales the estimate down when priors
say such goals usually close. Not wired yet (needs the prior index); the
hook keeps the interface stable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_WS_RE = re.compile(r"\s+")

# Generic structural markers, weighted by how much each tends to add to
# proof effort. These are mathematical structure, NOT benchmark
# vocabulary (no problem names, no per-problem lemma lists).
_QUANTIFIERS = ("∀", "∃", "∃!")
_BIG_OPS = ("∑", "∏", "⨆", "⨅", "∫", "Finset.sum", "Finset.prod",
            "tsum", "∮")
_GEOMETRIC = ("Collinear", "Wbtw", "Sbtw", "SameRay", "affineSpan",
              "segment", "∠", "dist ", "EuclideanSpace", "Convex")
_INDEXED = ("Fin ", "Matrix", "Finset", "iSup", "iInf", "Set.",
            "Multiset", "List.")
_CONNECTIVES = ("↔", "→", "∧", "∨", "¬")

_W_QUANT = 3.0
_W_BIGOP = 5.0
_W_GEO = 4.0
_W_INDEXED = 2.0
_W_CONNECTIVE = 1.0
_W_SIZE = 0.05        # per normalized char
_W_BINDER = 1.5       # per explicit binder group "( … : … )"

# A genuine binder group `(x : T)` / `(a b : T)` — the part before the
# colon must be identifier-like (bound variable names), so a type
# ascription on a literal such as `(0 : ℝ)` or `(2 : ℤ)` in a conclusion
# is NOT miscounted as a binder (it was, inflating the difficulty of any
# lemma using typed literals and causing false quality-gate offender flags).
_BINDER_RE = re.compile(
    r"\(\s*[A-Za-z_][A-Za-z0-9_'₀-₉ₐ-ₜ]*(?:\s+[A-Za-z_][A-Za-z0-9_'₀-₉ₐ-ₜ]*)*"
    r"\s*:[^()]*\)")


@dataclass(slots=True)
class DifficultyFeatures:
    size: int = 0
    n_binders: int = 0
    n_quantifiers: int = 0
    n_big_ops: int = 0
    n_geometric: int = 0
    n_indexed: int = 0
    n_connectives: int = 0

    def to_dict(self) -> dict:
        return {
            "size": self.size, "n_binders": self.n_binders,
            "n_quantifiers": self.n_quantifiers, "n_big_ops": self.n_big_ops,
            "n_geometric": self.n_geometric, "n_indexed": self.n_indexed,
            "n_connectives": self.n_connectives,
        }


def _count_any(text: str, needles: tuple[str, ...]) -> int:
    return sum(text.count(n) for n in needles)


def difficulty_features(goal_text: str) -> DifficultyFeatures:
    """Extract the surface features used by `estimate_difficulty`. Public
    for transparency/testing — the estimate is just their weighted sum."""
    norm = _WS_RE.sub(" ", goal_text or "").strip()
    return DifficultyFeatures(
        size=len(norm),
        n_binders=len(_BINDER_RE.findall(norm)),
        n_quantifiers=_count_any(norm, _QUANTIFIERS),
        n_big_ops=_count_any(norm, _BIG_OPS),
        n_geometric=_count_any(norm, _GEOMETRIC),
        n_indexed=_count_any(norm, _INDEXED),
        n_connectives=_count_any(norm, _CONNECTIVES),
    )


def estimate_difficulty(goal_text: str,
                        *, provability_prior: float | None = None) -> float:
    """A deterministic difficulty proxy for a goal's surface syntax.
    Higher = harder. Monotone in size and in every structure marker.

    `provability_prior` (optional, in [0,1]) is a historical closure rate
    for similarly-shaped goals; when given it scales the estimate by
    (1 - prior), so goals that priors say usually close look easier. Not
    yet supplied by any caller (Point-3 hook)."""
    f = difficulty_features(goal_text)
    score = (
        _W_SIZE * f.size
        + _W_BINDER * f.n_binders
        + _W_QUANT * f.n_quantifiers
        + _W_BIGOP * f.n_big_ops
        + _W_GEO * f.n_geometric
        + _W_INDEXED * f.n_indexed
        + _W_CONNECTIVE * f.n_connectives
    )
    if provability_prior is not None:
        prior = max(0.0, min(1.0, provability_prior))
        score *= (1.0 - prior)
    return round(score, 4)


@dataclass(slots=True)
class QualityReport:
    ok: bool
    goal_difficulty: float
    threshold: float
    factor: float
    lemma_difficulty: dict[str, float] = field(default_factory=dict)
    offenders: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "goal_difficulty": self.goal_difficulty,
            "threshold": self.threshold, "factor": self.factor,
            "lemma_difficulty": dict(self.lemma_difficulty),
            "offenders": list(self.offenders), "reason": self.reason,
        }


def theory_reduces_difficulty(
    goal_texts: list[str],
    lemma_goals: list[tuple[str, str]],
    *,
    factor: float = 0.9,
    prior_fn=None,
) -> QualityReport:
    """Decision for the decomposition-quality gate.

    `goal_texts` — the conclusion(s) of the stuck obligation(s).
    `lemma_goals` — (name, conclusion) for each proposed lemma.
    `prior_fn` — optional ``goal_text -> float|None`` provability prior
    (historical closure rate for similarly-shaped goals, e.g. from the
    proof-prior index). Applied to BOTH sides of the inequality, so a
    lemma shaped like goals the prior often closes reads as easier. A
    None return (or a raising prior) contributes nothing.

    Passes iff every proposed lemma is meaningfully easier than the
    hardest stuck obligation: ``max_i D(L_i) < factor · D(G)``. A theory
    with no lemmas (pure defs/tactics) trivially passes. If the goal's
    estimated difficulty is non-positive the gate abstains (passes) — it
    cannot assess a reduction it cannot measure."""
    def _est(g: str) -> float:
        prior = None
        if prior_fn is not None:
            try:
                prior = prior_fn(g)
            except Exception:
                prior = None
        return estimate_difficulty(g, provability_prior=prior)

    d_goal = max((_est(g) for g in goal_texts), default=0.0)
    lemma_diff = {(name or f"lemma_{i}"): _est(g)
                  for i, (name, g) in enumerate(lemma_goals)}
    threshold = factor * d_goal
    if d_goal <= 0.0 or not lemma_diff:
        return QualityReport(ok=True, goal_difficulty=d_goal,
                             threshold=threshold, factor=factor,
                             lemma_difficulty=lemma_diff)
    offenders = [n for n, d in lemma_diff.items() if d >= threshold]
    if offenders:
        return QualityReport(
            ok=False, goal_difficulty=d_goal, threshold=threshold,
            factor=factor, lemma_difficulty=lemma_diff, offenders=offenders,
            reason=(f"lemma(s) {offenders} are ≈ as hard as the stuck "
                    f"obligation (difficulty ≥ {threshold:.2f} of "
                    f"D(goal)={d_goal:.2f}) — the theory preserves "
                    f"complexity instead of reducing it"))
    return QualityReport(ok=True, goal_difficulty=d_goal,
                         threshold=threshold, factor=factor,
                         lemma_difficulty=lemma_diff)
