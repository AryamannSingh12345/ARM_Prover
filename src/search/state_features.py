"""Extract `ProofStateFeatures` from a Lean proof state.

PART 3 of the proof-prior series.

This is a syntactic extractor over goal/hypothesis text. It does NOT use the
Lean elaborator, so it has to be permissive about formatting (Lean's
`Nat.gcd n 40` vs `n.gcd 40`, `≤` vs `<=`, Unicode types vs ASCII names).

Design rules:
  - Feature bag is small and stable (~dozens of distinct values per slot)
    so feature_key() in proof_prior.py is meaningful.
  - Never include theorem IDs, variable names, or hypothesis names. The bag
    must generalise across problems.
"""
from __future__ import annotations

import re

from search.proof_prior import ProofStateFeatures


_NAMESPACES = ("Nat", "Int", "Rat", "Real", "Complex", "Finset", "Set",
               "Polynomial", "Matrix", "Multiset", "List")

# (canonical-feature, alternate spellings as plain substring or regex). Order
# matters in the regex case: longer/more specific should appear first.
_SYMBOL_SUBSTRINGS: list[tuple[str, tuple[str, ...]]] = [
    ("gcd", ("gcd", ".gcd")),
    ("lcm", ("lcm", ".lcm")),
    ("sqrt", ("sqrt", "Real.sqrt")),
    ("sum", ("∑", "Finset.sum", ".sum ")),
    ("prod", ("∏", "Finset.prod", ".prod ")),
    ("range", ("Finset.range", ".range ", "range ")),
    ("factorial", ("Nat.factorial", "factorial", "!")),
    ("divisibility", ("∣", "Nat.dvd", "Dvd.dvd", "∣ ")),
    ("modulo", ("%", "Nat.mod", ".mod ")),
    ("power", ("^",)),
    ("inequality", ("≤", "≥", "<", ">", "<=", ">=")),
    ("equality", ("=",)),
    ("absolute", ("|", "abs ", "Real.abs")),
    ("logarithm", ("Real.log", "Nat.log")),
]

_TYPE_TOKENS = {
    "ℕ": "Nat",
    "ℤ": "Int",
    "ℚ": "Rat",
    "ℝ": "Real",
    "ℂ": "Complex",
}

# Numeric constants. Lookbehind/lookahead forbid letters, underscore AND
# DIGITS so identifier-embedded numerics like `mathd_numbertheory_100`
# don't bleed a `00` capture into the constants slot.
_NUMERIC_RE = re.compile(r"(?<![A-Za-z_0-9])(\d+(?:\.\d+)?)(?![A-Za-z_0-9])")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
# Goal targets we recognise. The matcher is intentionally loose — Lean
# pretty-prints the same proposition many ways depending on coercions.
_TARGET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("conjunction", re.compile(r"\b\w[^∧]{0,80}∧")),
    ("disjunction", re.compile(r"\b\w[^∨]{0,80}∨")),
    ("existential", re.compile(r"^\s*∃")),
    ("universal", re.compile(r"^\s*∀")),
    ("set_membership", re.compile(r"∈")),
    ("divisibility", re.compile(r"∣")),
    ("inequality", re.compile(r"[<>≤≥]|<=|>=")),
    ("equality", re.compile(r"\s=\s|≠")),
]


def _detect_namespaces(text: str) -> list[str]:
    out: list[str] = []
    for ns in _NAMESPACES:
        # "Nat" as a token boundary or as a dotted-name prefix.
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(ns)}(?:\.|\b)", text):
            out.append(ns)
    # Unicode types map onto their canonical namespace.
    for sym, ns in _TYPE_TOKENS.items():
        if sym in text and ns not in out:
            out.append(ns)
    return out


def _detect_symbols(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for canon, needles in _SYMBOL_SUBSTRINGS:
        if canon in seen:
            continue
        for needle in needles:
            if needle in text:
                out.append(canon)
                seen.add(canon)
                break
    return out


def _detect_target_shape(target_text: str) -> str:
    # Strip leading `⊢ ` if present so anchors work.
    t = target_text.strip()
    t = re.sub(r"^⊢\s*", "", t)
    for shape, pat in _TARGET_PATTERNS:
        if pat.search(t):
            return shape
    return "unknown"


_HYPOTHESIS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("gcd_eq", re.compile(r"gcd[^=]{0,40}=")),
    ("lcm_eq", re.compile(r"lcm[^=]{0,40}=")),
    ("positivity", re.compile(r"0\s*<\s*|<\s*0|Pos|positive", re.IGNORECASE)),
    ("nonzero", re.compile(r"≠\s*0|!=\s*0|ne_zero|NeZero")),
    ("membership", re.compile(r"∈|MemFinset|Mem ")),
    ("bound", re.compile(r"≤|≥|<=|>=|<|>")),
    ("recurrence", re.compile(r"\bsucc\b|\.succ\b|n\s*\+\s*1")),
]


def _detect_hypothesis_shapes(hyp_text: str) -> list[str]:
    out: list[str] = []
    for shape, pat in _HYPOTHESIS_PATTERNS:
        if pat.search(hyp_text):
            out.append(shape)
    return out


def _detect_constants(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _NUMERIC_RE.finditer(text):
        val = m.group(1)
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
        if len(out) >= 8:
            break
    return out


def _split_goal_text(goal_text: str) -> tuple[str, str]:
    """Return (hypotheses_block, target_block).

    Lean pretty-prints state as one hypothesis per line followed by `⊢ <goal>`.
    If there is no `⊢`, treat the whole string as the target.
    """
    if not goal_text:
        return "", ""
    if "⊢" in goal_text:
        head, _, tail = goal_text.rpartition("⊢")
        return head.strip(), ("⊢ " + tail.strip()).strip()
    return "", goal_text.strip()


def _classify_previous(proof_prefix: list[str]) -> str | None:
    if not proof_prefix:
        return None
    last = proof_prefix[-1].strip()
    if not last:
        return None
    try:
        from search.tactic_classify import classify_tactic
    except Exception:
        return None
    cls = classify_tactic(last)
    return cls if cls != "unknown" else None


def extract_state_features(
    goal_text: str,
    proof_prefix: list[str],
    retrieved_premises: list[str] | None = None,
) -> ProofStateFeatures:
    """Build a `ProofStateFeatures` from raw goal text + proof prefix.

    Conservative: any extractor failure returns an empty value for that slot
    rather than raising. The point is to produce a *signature*, not to parse.
    """
    hyp_text, target_text = _split_goal_text(goal_text or "")
    namespaces = tuple(_detect_namespaces(goal_text or ""))
    symbols = tuple(_detect_symbols(goal_text or ""))
    target_shape = _detect_target_shape(target_text)
    hypothesis_shapes = tuple(_detect_hypothesis_shapes(hyp_text))
    constants = tuple(_detect_constants(goal_text or ""))
    previous_tactic_class = _classify_previous(proof_prefix or [])
    retrieved = tuple(retrieved_premises or [])[:32]
    return ProofStateFeatures(
        symbols=symbols,
        namespaces=namespaces,
        target_shape=target_shape,
        hypothesis_shapes=hypothesis_shapes,
        constants=constants,
        previous_tactic_class=previous_tactic_class,
        retrieved_premises=retrieved,
    )
