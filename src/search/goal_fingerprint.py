"""Goal fingerprints for the circularity guard (Point 7a).

The circularity guard used to compare goal conclusions by stripping ALL
whitespace and testing string equality (`proof_dag._norm_ws`). That is
too weak: it treats `x + x = 2 * x` and `a + a = 2 * a` as different, so
a model can evade the guard by renaming bound variables, and it is blind
to binder reordering and notation/definitional equivalence.

This module provides three escalating comparisons, cheapest first:

1. ``text`` — the legacy behaviour (whitespace-stripped equality). Kept
   as the exact default so nothing changes unless a mode is chosen.
2. ``syntactic`` — an **α-invariant** fingerprint (compile-free): bound
   variables (from binders and ∀/∃/λ) are renamed to positional
   placeholders before whitespace normalization, so alpha-equivalent
   conclusions collapse. Strictly stronger than ``text``.
3. ``elaborated`` — escalation via a caller-supplied ``defeq_probe``
   callback that decides definitional equality in Lean (the only true
   oracle for reducible-wrapper / notation-expanded equivalence). This
   module stays Lean-agnostic; it only orchestrates the callback and
   ALWAYS treats an inconclusive probe (``None``) as "not circular", so
   the escalation can never be *less* safe than ``syntactic`` and can
   never manufacture a false solve — it only ever rejects more theories.

All of this is a heuristic guard on the *input* to proving; the Lean
verifier remains the sole oracle for whether a proof is real.
"""
from __future__ import annotations

import re

# Lean identifier characters (incl. primes, dotted names, subscripts).
_IDENT = r"[A-Za-z_][A-Za-z0-9_'.₀-₉ₐ-ₜ]*"
_IDENT_RE = re.compile(_IDENT)
_WS_RE = re.compile(r"\s+")

# Named bracket binders: (xs : T) {xs : T} ⦃xs : T⦄ — capture the names
# before the colon. Unnamed instance binders like [Foo] have no colon and
# introduce no name, so they are skipped.
_BRACKET_BINDER_RE = re.compile(
    r"[\(\{⦃]\s*([^:\(\)\{\}\[\]⦃⦄]+?)\s*:")
# Quantifier / lambda binders: ∀ a b : T,  ∃ x,  fun a b =>  λ x,
_QUANT_BINDER_RE = re.compile(
    r"(?:∀|∃|Σ|Π|λ|\bfun\b)\s*([A-Za-z_][^,:=]*?)\s*(?:[,:]|=>)")


def _bound_names(prefix: str, conclusion: str) -> list[str]:
    """Collect binder-introduced identifier names, in first-appearance
    order, from the statement prefix (binders) and any quantifiers/lambdas
    in either the prefix or the conclusion."""
    names: list[str] = []
    seen: set[str] = set()

    def _add_group(group: str) -> None:
        for tok in _IDENT_RE.findall(group):
            if tok not in seen:
                seen.add(tok)
                names.append(tok)

    for text in (prefix, conclusion):
        for m in _BRACKET_BINDER_RE.finditer(text):
            _add_group(m.group(1))
        for m in _QUANT_BINDER_RE.finditer(text):
            _add_group(m.group(1))
    return names


def _rename(text: str, mapping: dict[str, str]) -> str:
    """Whole-identifier replacement of bound names by their placeholders."""
    def sub(m: re.Match) -> str:
        return mapping.get(m.group(0), m.group(0))
    return _IDENT_RE.sub(sub, text)


# Final declaration keyword + optional name, used to peel the prelude and
# `kw NAME` off a statement so the binder telescope is left.
_DECL_KW_RE = re.compile(
    r"(?:theorem|lemma|def|abbrev|instance|example)"
    r"\s+([A-Za-z_][A-Za-z0-9_'.]*)?")


def statement_to_prop(statement: str, *, split_fn) -> str | None:
    """Turn a theorem statement into its closed proposition, ``∀ binders,
    goal``, for a Lean definitional-equality probe. `split_fn` yields
    (prefix, goal); the binders are whatever follows the final
    ``kw NAME`` in the prefix. Returns None when the statement does not
    split or has no goal. String-level (the review's Point-7 caveat
    applies) — but the result is only ever handed to Lean, which is the
    real oracle; a malformed conversion just makes the probe error out
    (→ 'not circular', safe)."""
    parts = split_fn(statement)
    if parts is None:
        return None
    prefix, goal = parts
    goal = (goal or "").strip()
    if not goal:
        return None
    matches = list(_DECL_KW_RE.finditer(prefix))
    if not matches:
        return None
    binders = prefix[matches[-1].end():].strip()
    return f"∀ {binders}, {goal}" if binders else goal


def _key(statement: str, *, mode: str, split_fn) -> str | None:
    """Comparison key for one statement's conclusion, or None when the
    header does not split. None NEVER matches, in any mode — reproducing
    the legacy guard, which silently ignored unsplittable statements
    rather than risk a false circular rejection.

    - ``text``: whitespace-stripped conclusion (exactly `_norm_ws(goal)`).
    - otherwise (``syntactic``/``elaborated`` prefilter): α-invariant —
      bound names renamed to positional placeholders first."""
    parts = split_fn(statement)
    if parts is None:
        return None
    prefix, concl = parts
    if mode == "text":
        return _WS_RE.sub("", concl)
    names = _bound_names(prefix, concl)
    mapping = {n: f"\x00{i}\x00" for i, n in enumerate(names)}
    return _WS_RE.sub("", _rename(concl, mapping))


def conclusion_fingerprint(statement: str, *, split_fn) -> str:
    """Public α-invariant fingerprint of a statement's conclusion (for
    tests / external callers). Falls back to the whitespace-stripped raw
    statement when the header does not split."""
    k = _key(statement, mode="syntactic", split_fn=split_fn)
    return k if k is not None else _WS_RE.sub("", statement)


def text_fingerprint(statement: str, *, split_fn) -> str:
    """Public legacy key (whitespace-stripped conclusion); raw fallback
    when the header does not split."""
    k = _key(statement, mode="text", split_fn=split_fn)
    return k if k is not None else _WS_RE.sub("", statement)


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_'.₀-₉ₐ-ₜ]*")


def _concl_tokens(statement: str, *, split_fn) -> set[str]:
    parts = split_fn(statement)
    concl = parts[1] if parts is not None else statement
    return set(_TOKEN_RE.findall(concl or ""))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_circular(
    lemma_stmt: str,
    goal_stmts: list[str],
    *,
    mode: str,
    split_fn,
    defeq_probe=None,
    defeq_similarity: float = 0.6,
) -> bool:
    """True iff `lemma_stmt` restates one of `goal_stmts`.

    Cheap prefilter first (text or α-invariant fingerprint equality;
    unsplittable statements never match). In 'elaborated' mode, if the
    prefilter finds no match and a `defeq_probe` is supplied, escalate to
    Lean definitional equality — BUT only against goals whose conclusion
    shares ≥ `defeq_similarity` token-overlap (Jaccard) with the lemma.
    The escalation is one Lean compile per pair, so probing every
    obviously-unrelated pair is ruinous (amc12a_2021_p25: ~200 compiles,
    all `defeq:false`); a lemma that is definitionally equal to a goal but
    not syntactically equal is essentially always highly token-similar, so
    the overlap gate loses ~nothing while skipping the unrelated pairs.
    `defeq_probe(...)` returns True (defeq → circular), False (distinct),
    or None (undecidable → NOT circular; keeps escalation as safe as the
    syntactic mode)."""
    key = _key(lemma_stmt, mode=mode, split_fn=split_fn)
    if key is not None:
        goal_keys = {k for g in goal_stmts
                     if (k := _key(g, mode=mode, split_fn=split_fn))
                     is not None}
        if key in goal_keys:
            return True
    if mode == "elaborated" and defeq_probe is not None:
        lem_toks = _concl_tokens(lemma_stmt, split_fn=split_fn)
        for g in goal_stmts:
            if _jaccard(lem_toks,
                        _concl_tokens(g, split_fn=split_fn)) < defeq_similarity:
                continue  # too dissimilar to be defeq — skip the compile
            try:
                verdict = defeq_probe(lemma_stmt, g)
            except Exception:
                verdict = None
            if verdict is True:
                return True
    return False
