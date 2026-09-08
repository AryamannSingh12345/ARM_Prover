"""Deterministic template tactic generation.

Spec: PART 6 of the proof-prior series.

Given a `ProofStateFeatures` bag, a list of retrieved Mathlib premises, and
(optionally) the raw goal text for term extraction, emit a list of
`(tactic_text, heuristic_cost, metadata)` triples. Metadata is the dict
expected downstream by the move scorer:

    {"source": "template",
     "premise": <Mathlib name or None>,
     "tactic_class": <classify_tactic(tactic)>,
     "tags": (...)}

Two generation paths:
  - PER-PREMISE: for each retrieved premise P, emit the basic citations
    (`exact P`, `apply P`, `rw [P]`, `simp [P]`, `have h := P`) plus a
    handful of arg-instantiated `have h := P a [b ...]` variants when P
    looks instantiable.
  - DOMAIN: shape-specific templates that don't need a premise (e.g.
    `nlinarith` for real inequalities, `norm_num`/`ring_nf` for sums).

Hard rule: no placeholder tactics (`?_`, `_`, `?a`, `<FILL>`) are ever
emitted for execution. If we cannot instantiate concretely, we don't emit
the template.
"""
from __future__ import annotations

import math
import re
from typing import Iterable

from search.proof_prior import ProofStateFeatures
from search.tactic_classify import classify_tactic


# --- argument extraction ----------------------------------------------------

_IDENT_RE = re.compile(r"\b([a-z][a-z0-9]?)\b")
# Numeric constants — lookbehind/lookahead forbid letters, underscore AND
# digits so `mathd_numbertheory_100` doesn't leak a `00` argument into
# the gcd/lcm domain template.
_NUMERIC_RE = re.compile(r"(?<![A-Za-z_0-9])(\d+)(?![A-Za-z_0-9])")
_RESERVED = frozenset({
    "by", "in", "fun", "do", "let", "if", "to", "as", "of", "or",
})
_GENERIC_TERMS: tuple[str, ...] = ("n", "k", "m", "a", "b", "c", "x", "y", "z")


def _candidate_terms(goal_text: str | None,
                     features: ProofStateFeatures) -> list[str]:
    """Best-effort: locals from the goal, then numeric constants, then
    generic single-letter fallbacks. Deduped, order preserved."""
    out: list[str] = []
    seen: set[str] = set()

    def push(t: str) -> None:
        if t and t not in seen:
            seen.add(t)
            out.append(t)

    if goal_text:
        for m in _IDENT_RE.finditer(goal_text):
            tok = m.group(1)
            if tok in _RESERVED:
                continue
            push(tok)
        for m in _NUMERIC_RE.finditer(goal_text):
            push(m.group(1))
    for c in features.constants:
        push(c)
    for g in _GENERIC_TERMS:
        push(g)
    return out


# --- premise-driven templates ----------------------------------------------

_HEURISTIC_COST = {
    "exact":     0.6,
    "apply":     0.7,
    "rw":        0.7,
    "simp":      0.8,
    "have":      0.8,
    "instant":   0.9,
}


# ---------- premise-name shape filters (PART B fix) ------------------------
#
# Mathlib decl names lookup yields many *function/type* declarations whose
# tail is a data constructor or type, not a proposition. Emitting
# `exact Nat.log` or `apply Nat.P` against such a decl is a guaranteed
# type error and just burns lake budget. We use a tight allow-rule for
# `exact`/`apply`/`rw`/`simp`:
#
#   1. Hard-reject specific data-constructor / function tails.
#   2. Reject uppercase-leading tails (types / structures / typeclasses).
#   3. Otherwise accept iff the tail contains a `_` OR is a known
#      bare-lemma name. `_`-containing tails are how Mathlib lemmas
#      are spelled (`gcd_mul_lcm`, `sqrt_nonneg`, `add_comm`), so this
#      catches >95% of real lemmas while excluding `log`, `bit`, etc.
#
# `have h := P` and arg-instantiations of the same are still emitted
# even when the predicate is False — they at most introduce a useless
# hypothesis, which is cheap to discard.

_EXACT_REJECT_TAILS: frozenset[str] = frozenset({
    "P", "log", "bit", "nth", "fib", "dist", "Upto", "xgcd", "psub",
    "beta", "ceil", "clog", "pair", "card", "bell", "floor", "ppred",
    "choose", "digits", "factorial",
    # additional common-noise tails from the Nat namespace
    "size", "isEven", "isOdd", "iterate", "rec", "casesOn", "find",
    "succ", "pred", "zero", "one",
})

# Theorem-suffix tokens that almost always mark a proposition. We allow
# `exact` / `apply` for these even on short tails without `_`.
_THEOREM_SUFFIX_TOKENS: frozenset[str] = frozenset({
    "eq", "lt", "le", "ge", "gt", "ne", "pos", "neg", "nonneg",
    "nonpos", "comm", "assoc", "iff", "mul", "add", "sub", "pow",
    "sqrt", "gcd", "lcm", "mod", "dvd", "card", "prime", "sum",
    "prod", "range",
})

_BARE_LEMMA_ALLOWLIST: frozenset[str] = frozenset({
    "rfl", "id", "absurd", "trivial", "sq_nonneg", "add_pos", "mul_pos",
    "ne_of_gt",
})

# PART D: lemmas whose plain `exact`/`apply` form almost always
# type-errors because they take explicit arguments. For these we skip
# the bare `exact P` / `apply P` templates and rely on the instantiation
# block (`have h := P a b`) plus rw/simp to surface them. Adding entries
# is conservative — only premises where the bare `exact P` form is a
# known dead end. The matching is exact-name based so `Nat.gcd_mul_lcm`
# does NOT silence `exact Nat.gcd_mul_lcm_aux` or similar siblings.
_PREFER_INSTANTIATION: frozenset[str] = frozenset({
    "Nat.gcd_mul_lcm", "gcd_mul_lcm",
})


def _is_theorem_like_for_exact(premise: str) -> bool:
    """True iff `premise` is safe to wrap in `exact`/`apply`/`rw`/`simp`.

    Conservative: rejects type-shaped tails, hand-listed noise names,
    and bare lowercase tails without an underscore. Lemma-shaped names
    (`Nat.gcd_mul_lcm`, `add_comm`, `Real.sqrt_pos`) pass."""
    if not premise:
        return False
    if premise in _BARE_LEMMA_ALLOWLIST:
        return True
    tail = premise.rsplit(".", 1)[-1]
    if tail in _EXACT_REJECT_TAILS:
        return False
    if tail[:1].isupper():
        return False
    if "_" in tail:
        # Strong signal of a Mathlib lemma; double-check no rejected
        # token is a *piece* (e.g. avoid `card_log_X` if it ever appears).
        pieces = set(tail.split("_"))
        if pieces & _EXACT_REJECT_TAILS:
            return False
        return True
    # Bare lowercase tail without an underscore is almost always a
    # function or interface name (`Nat.gcd`, `Nat.dvd`, `Real.sqrt`),
    # not a proposition — `exact Nat.gcd` would type-error. Reject.
    return False


def _instantiable(premise: str) -> bool:
    """Heuristic: lemmas about Nat./Real./Int./Finset./Polynomial. and
    bare-name lemmas are usually applied to one or two arguments. We accept
    any premise — the cost gates risky instantiations."""
    return bool(premise) and "." in premise or premise in {
        "sq_nonneg", "add_pos", "mul_pos", "ne_of_gt",
    }


def _per_premise_templates(
    premise: str,
    *,
    terms: list[str],
    max_per_premise: int,
) -> list[tuple[str, float, str]]:
    """Return (tactic_text, cost, kind_tag) for one premise. kind_tag is
    used to set tactic_class; emitted text contains no placeholders.

    Ordering matters: the final `[:max_per_premise]` truncates the tail,
    so we emit the high-value templates first. Order:
      1. `exact P` / `apply P` / `rw [P]` / `simp [P]` / `have h := P`
         (5 basic citations)
      2. `have h := P <var> <num>` — the variable+constant pair pattern
         that closes gcd/lcm/factorial-style hypotheses
      3. `have h := P <var>` and `have h := P <num>` — 1-arg variants
      4. Other 2-arg combinations
    """
    out: list[tuple[str, float, str]] = []
    used: set[str] = set()

    def push(stmt: str, cost: float, kind: str) -> None:
        if stmt in used:
            return
        used.add(stmt)
        out.append((stmt, cost, kind))

    # Per-premise templates. Order of emission matters because the
    # caller truncates at `max_per_premise`.
    #
    # PART C (fix-up): bare `exact P` / `apply P` are dropped
    # unconditionally — almost every Mathlib lemma takes explicit
    # arguments, so `exact P` is a near-certain type error and burns
    # lake budget. The arg-instantiation block below produces
    # `have h := P a b` which DOES typecheck for the same lemmas.
    #
    # PART B: rw/simp are still gated on theorem-likeness — for junk
    # names (`Nat.P`, `Nat.log`, `Nat.bit`) even `rw [P]` would error.
    # `have h := P` is always emitted because it cannot type-error
    # (it just binds h to whatever term P is, useful for downstream
    # tactics that consume the bound hypothesis).
    if _is_theorem_like_for_exact(premise):
        push(f"rw [{premise}]",       _HEURISTIC_COST["rw"],    "rw")
        push(f"simp [{premise}]",     _HEURISTIC_COST["simp"],  "simp")
    push(f"have h := {premise}",      _HEURISTIC_COST["have"],  "have")

    if _instantiable(premise) and terms:
        var_terms = [t for t in terms if not t.isdigit()][:3]
        num_terms = [t for t in terms if t.isdigit()][:3]
        # 2a. variable + numeric (highest-value pattern for arithmetic lemmas).
        for v in var_terms[:2]:
            for n in num_terms[:2]:
                push(f"have h := {premise} {v} {n}",
                     _HEURISTIC_COST["instant"], "have")
        # 2b. 1-arg variants — variable first, then numeric.
        for v in var_terms[:2]:
            push(f"have h := {premise} {v}",
                 _HEURISTIC_COST["instant"] + 0.05, "have")
        for n in num_terms[:2]:
            push(f"have h := {premise} {n}",
                 _HEURISTIC_COST["instant"] + 0.05, "have")
        # 2c. variable + variable.
        for v1 in var_terms[:2]:
            for v2 in var_terms[:2]:
                if v1 != v2:
                    push(f"have h := {premise} {v1} {v2}",
                         _HEURISTIC_COST["instant"] + 0.1, "have")
    return out[:max_per_premise]


# --- hypothesis-name extraction ---------------------------------------------
#
# Used by the D.2/D.3 fix-up to splice live hypothesis names into
# nlinarith-with-hyps and local-equality-rewrite templates. Handles
# BOTH theorem-header parens (`(h₀ : 0 < n)`) and Lean state-format
# binder lines (`h₁ : Nat.gcd n 40 = 10`) so the same generator fires
# at root (header text) and after the first step (state text).

# Parenthesised binders: `(h₀ : 0 < n)`, `(x y : ℝ)`, `(S : Finset ℝ)`.
_PAREN_BINDER_RE = re.compile(
    r"\(\s*([A-Za-z_][A-Za-z0-9_'₀-₉]*"   # first name
    r"(?:\s+[A-Za-z_][A-Za-z0-9_'₀-₉]*)*"  # optional additional names
    r")\s*:\s*([^()]*?)\s*\)"
)
# State-format binder lines: one per line, `name : type`. Excludes the
# goal line `⊢ …`.
_STATE_BINDER_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_'₀-₉]*)\s*:\s*(.+?)\s*$",
    re.MULTILINE,
)
# Tactic / declaration keywords that the binder regex MUST never pick
# up as a hypothesis name (e.g. `theorem foo` matching the first regex).
_BINDER_NAME_BLOCKLIST: frozenset[str] = frozenset({
    "theorem", "lemma", "example", "instance", "def", "abbrev",
    "by", "let", "if", "do", "fun", "match", "show",
})


def _extract_hypothesis_bindings(text: str) -> list[tuple[str, str]]:
    """Return `[(name, type_text), …]` for every hypothesis-like binder
    in `text`. Deduped on `name`, order preserved (header order at root,
    state order after).

    The matcher is conservative: it skips group-binders with multiple
    names (e.g. `(x y : ℝ)` — those are type bindings, not propositions)
    and any name in `_BINDER_NAME_BLOCKLIST`. Returns an empty list when
    `text` is empty.
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    # Parenthesised binders (theorem-header format).
    for m in _PAREN_BINDER_RE.finditer(text):
        head = m.group(1).strip()
        typ = m.group(2).strip()
        toks = head.split()
        # Skip multi-name binders — they're type vars, not propositions.
        if len(toks) != 1:
            continue
        name = toks[0]
        if name in _BINDER_NAME_BLOCKLIST or name in seen:
            continue
        seen.add(name)
        out.append((name, typ))
    # State-format binder lines (after entering tactic mode).
    for m in _STATE_BINDER_RE.finditer(text):
        name = m.group(1).strip()
        typ = m.group(2).strip()
        if name in _BINDER_NAME_BLOCKLIST or name in seen:
            continue
        # The state ⊢-line is filtered by the regex anchors, but a
        # `theorem foo (...)` line in a header would match the first
        # token as `theorem`; the blocklist above handles it.
        # Skip multi-token bindings on a single line (`n k : ℕ`).
        if len(name.split()) != 1:
            continue
        seen.add(name)
        out.append((name, typ))
    return out


_PROPOSITION_OPS_RE = re.compile(
    r"\s=\s|≠|<=|>=|<|>|≤|≥|∣|∈|∀|∃|→|↔|∧|∨"
)


def _is_propositional_type(type_text: str) -> bool:
    """True iff `type_text` looks like a proposition (an inhabitant of
    `Prop`) rather than a data type. Used to count "useful" hypotheses
    for the arithmetic-closer penalty and the nlinarith-with-hyps
    template generator."""
    if not type_text:
        return False
    # Bare type names — `ℕ`, `ℤ`, `Finset ℝ`, `Polynomial ℝ` — almost
    # never propositions. The conservative check: at least one
    # propositional operator (= / ≤ / < / ≠ / ∈ / ∀ / ∃ / →) appears.
    return bool(_PROPOSITION_OPS_RE.search(type_text))


def _propositional_hypotheses(text: str) -> list[tuple[str, str]]:
    """Subset of `_extract_hypothesis_bindings` whose type is a proposition."""
    return [
        (n, t) for n, t in _extract_hypothesis_bindings(text)
        if _is_propositional_type(t)
    ]


def _equality_hypotheses(text: str) -> list[str]:
    """Names of hypotheses whose type is a top-level equality
    (`X = Y`). Excludes inequalities and disequalities."""
    out: list[str] = []
    for n, t in _propositional_hypotheses(text):
        if "≠" in t or "≤" in t or "≥" in t:
            continue
        if " = " in t or t.endswith("="):
            out.append(n)
    return out


def _scalar_real_variables(text: str) -> list[str]:
    """Names whose type is a numeric scalar (ℝ, ℚ, ℤ, ℕ). Single-name
    or grouped binders both count. Used by sq_nonneg hint generation."""
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _PAREN_BINDER_RE.finditer(text):
        head = m.group(1).strip()
        typ = m.group(2).strip()
        # Match bare scalar types only.
        if typ not in {"ℝ", "ℚ", "ℤ", "ℕ",
                        "Real", "Rat", "Int", "Nat"}:
            continue
        for name in head.split():
            if (name not in _BINDER_NAME_BLOCKLIST
                    and name not in seen
                    and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_'₀-₉]*", name)):
                seen.add(name)
                out.append(name)
    return out


def context_richness_score(text: str) -> int:
    """Coarse measure used by the arithmetic-closer penalty.
    Higher = more for an arithmetic
    closer to chew on; 0 means linarith/nlinarith/omega have nothing
    to derive from. Public so the runner can call it without depending
    on private regex internals.
    """
    props = _propositional_hypotheses(text or "")
    # Each propositional hypothesis adds 1; an equality or inequality
    # adds another 1 (these are the shapes linarith / nlinarith / omega
    # actually consume).
    score = 0
    for _name, typ in props:
        score += 1
        if (" = " in typ or "≤" in typ or "≥" in typ
                or "<" in typ or ">" in typ):
            score += 1
    return score


# --- domain templates -------------------------------------------------------

def _gcd_lcm_templates(terms: list[str]) -> list[tuple[str, float, str]]:
    """gcd/lcm domain template: `have h := Nat.gcd_mul_lcm a b` plus
    arithmetic/ring closers.

    Uses the conventional hypothesis name `h` only — per the
    continuation-fix-up contract we do NOT bake in `hprod`/`h_…` names
    tailored to specific solved traces. The state-conditional rewrite
    step (`rw [eqs] at <recent>`) is the abstract_continuation_prior's
    job; this domain template only contributes the lemma instantiation."""
    out: list[tuple[str, float, str]] = []
    # Choose the first *meaningful* numeric constant as the second arg;
    # otherwise fall back to `k`. We skip 0 and 1 because they are
    # almost always positivity / step bounds (`0 < n`, `n + 1 ≤ m`),
    # not the second operand of a gcd / lcm hypothesis. The first
    # single-letter local variable (`n` / `m` / `a`) is the first arg.
    nums = [t for t in terms if t.isdigit() and t not in ("0", "1")][:1]
    locals_ = [t for t in terms if not t.isdigit() and len(t) == 1][:1]
    a = locals_[0] if locals_ else "n"
    b = nums[0] if nums else "k"
    out.append((f"have h := Nat.gcd_mul_lcm {a} {b}", 0.35, "have"))
    out.append(("omega",   0.30, "omega"))
    out.append(("ring_nf", 0.70, "ring_nf"))
    return out


def _real_ineq_templates(terms: list[str]) -> list[tuple[str, float, str]]:
    """Real inequality shape: linarith / nlinarith, plus `sq_nonneg t`
    hints for instantiable real terms (locals or constants)."""
    out: list[tuple[str, float, str]] = [
        ("linarith",   0.5, "linarith"),
        ("nlinarith",  0.5, "nlinarith"),
        ("positivity", 0.6, "positivity"),
    ]
    for t in terms[:3]:
        if not t.isdigit():
            out.append((f"nlinarith [sq_nonneg {t}]", 0.55, "nlinarith"))
    return out


def _finset_sum_templates(_terms: list[str]) -> list[tuple[str, float, str]]:
    return [
        ("norm_num", 0.5, "norm_num"),
        ("simp",     0.6, "simp"),
        ("ring_nf",  0.7, "ring_nf"),
        ("decide",   0.8, "unknown"),
    ]


def _sqrt_templates(terms: list[str]) -> list[tuple[str, float, str]]:
    out: list[tuple[str, float, str]] = [
        ("positivity", 0.5, "positivity"),
    ]
    for t in terms[:3]:
        if t.isdigit():
            continue
        out.append((f"have h := Real.sqrt_nonneg {t}", 0.6, "have"))
    return out


_DOMAIN_TEMPLATES = (
    ({"gcd", "lcm"},          _gcd_lcm_templates),
    ({"inequality"},          _real_ineq_templates),
    ({"sum", "range", "prod"},_finset_sum_templates),
    ({"sqrt"},                _sqrt_templates),
)


# --- D.2 / D.3 generic state-driven templates --------------------------------
#
# These fire regardless of the gcd/real-ineq/finset/sqrt domain switch
# above. They derive candidates from hypothesis NAMES extracted from
# the live goal_text (or theorem header at root), so they are generic
# across problem domains — no theorem-id mention, no problem-specific
# literals. Both generators return an empty list when the state lacks
# the structural hooks they consume, so they're zero-overhead on
# unrelated problems.

def _arithmetic_hyp_templates(
    features: ProofStateFeatures,
    goal_text: str | None,
) -> list[tuple[str, float, str]]:
    """`nlinarith [h₀, h₁, …]` and `nlinarith [sq_nonneg x, …]` driven
    by the hypothesis names and scalar real variables in scope.

    Fires when the GOAL is an inequality or equality involving a
    numeric scalar — `nlinarith` then has propositional hypotheses to
    chain through. Skipped when no propositional hypotheses are
    present (the hyp-spliced form would degenerate to bare `nlinarith`
    which the per-shape closer already emits).
    """
    text = goal_text or ""
    # Gate: at least one inequality / power / equality + a scalar
    # namespace. Avoids firing on pure-set / pure-Nat divisibility
    # problems where nlinarith is the wrong tool.
    scalar_ns = {"Real", "Rat", "Int"}
    has_arith_goal = (
        "inequality" in features.symbols
        or "power" in features.symbols
        or "equality" in features.symbols
    )
    if not has_arith_goal:
        return []
    if not (set(features.namespaces) & scalar_ns):
        return []
    props = _propositional_hypotheses(text)
    out: list[tuple[str, float, str]] = []
    if props:
        names = [n for n, _t in props][:5]
        out.append(
            (f"nlinarith [{', '.join(names)}]", 0.42, "nlinarith")
        )
        out.append(
            (f"linarith [{', '.join(names)}]", 0.48, "linarith")
        )
    scalars = _scalar_real_variables(text)
    if scalars:
        snip = ", ".join(f"sq_nonneg {v}" for v in scalars[:3])
        out.append((f"nlinarith [{snip}]", 0.45, "nlinarith"))
        if props:
            # Combine hypotheses with sq_nonneg hints — the canonical
            # nlinarith recipe for polynomial inequalities.
            names = [n for n, _t in props][:3]
            combined = ", ".join(
                list(f"sq_nonneg {v}" for v in scalars[:2]) + names
            )
            out.append((f"nlinarith [{combined}]", 0.40, "nlinarith"))
    return out


def _local_equality_rewrite_templates(
    _features: ProofStateFeatures,
    goal_text: str | None,
) -> list[tuple[str, float, str]]:
    """`rw [hX]` / `simp [hX]` for every equality hypothesis in scope.

    No `rw [hX] at *`: that form is brittle (rewrites everything,
    often loops) and the post-step state usually surfaces a better
    target for `rw [eqs] at <recent>` via the abstract-continuation
    instantiator. The bare `rw [hX]` and `simp [hX]` forms are cheap
    and useful at root for `(a = b)` hypotheses that match the goal
    LHS / RHS directly.
    """
    eqs = _equality_hypotheses(goal_text or "")
    if not eqs:
        return []
    out: list[tuple[str, float, str]] = []
    for h in eqs[:5]:
        out.append((f"rw [{h}]",   0.42, "rw"))
        out.append((f"simp [{h}]", 0.50, "simp"))
    if len(eqs) >= 2:
        # Joint rewrite — single Lean call versus one per equality.
        joined = ", ".join(eqs[:3])
        out.append((f"rw [{joined}]",   0.40, "rw"))
        out.append((f"simp [{joined}]", 0.48, "simp"))
    return out


_STATE_DRIVEN_TEMPLATES: tuple = (
    ("arithmetic_hyp_template", _arithmetic_hyp_templates),
    ("local_equality_rewrite",  _local_equality_rewrite_templates),
)


# NOTE: an earlier revision carried a `_rewrite_after_have_templates`
# helper that matched on the literal `hprod :` + `h₁`/`h₂` Nat.gcd /
# Nat.lcm pattern. That was a p100-specific shortcut and has been
# removed; the corpus-derived `abstract_continuation_prior` is the
# generic mechanism that replaces it (mined class transitions +
# goal_text-driven instantiation, no hard-coded hypothesis names).


def _has_any(features: ProofStateFeatures, needle: set[str]) -> bool:
    return any(s in needle for s in features.symbols)


# --- safety -----------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\?[A-Za-z0-9_]*|<FILL>|<TODO>|<PLACEHOLDER>", re.IGNORECASE)
_BARE_UNDERSCORE = re.compile(r"(?<![A-Za-z0-9_])_(?![A-Za-z0-9_])")


def _is_safe_for_execution(tactic: str) -> bool:
    """Reject anything that would slip a metavariable past Lean."""
    if not tactic:
        return False
    if _PLACEHOLDER_RE.search(tactic):
        return False
    if _BARE_UNDERSCORE.search(tactic):
        return False
    return True


# --- public API -------------------------------------------------------------

def generate_template_tactics(
    features: ProofStateFeatures,
    retrieved_premises: list[str],
    max_per_premise: int = 8,
    *,
    goal_text: str | None = None,
) -> list[tuple[str, float, dict]]:
    """Return a deterministic candidate list.

    Each element is `(tactic_text, heuristic_cost, metadata)`:
      - tactic_text never contains placeholders.
      - heuristic_cost is small (≤ ~1.0); lower = preferred. It is one
        input to move scoring, not the final priority.
      - metadata always contains: source, premise (or None), tactic_class,
        tags.
    """
    terms = _candidate_terms(goal_text, features)
    out: list[tuple[str, float, dict]] = []
    seen: set[str] = set()

    def emit(text: str, cost: float, premise: str | None, tags: tuple[str, ...]):
        text = text.strip()
        if not _is_safe_for_execution(text):
            return
        if text in seen:
            return
        seen.add(text)
        meta = {
            "source": "template",
            "premise": premise,
            "tactic_class": classify_tactic(text),
            "tags": tags,
        }
        out.append((text, cost, meta))

    # Domain-shape-driven FIRST so a high-confidence pattern like
    # `have h := Nat.gcd_mul_lcm n 40` is dedupe-winner over the
    # per-premise instantiation that would otherwise emit the same
    # text with premise_template tag (and a worse downstream score).
    for needle, fn in _DOMAIN_TEMPLATES:
        if _has_any(features, needle):
            for text, cost, _kind in fn(terms):
                emit(text, cost, None, ("domain_template",))

    # State-driven domain templates (D.2 / D.3): nlinarith-with-hyps and
    # local-equality rewrites. These read goal_text / theorem-header
    # binders directly and emit only when the state actually has the
    # relevant structural hooks, so they're zero-overhead otherwise.
    for tag, fn in _STATE_DRIVEN_TEMPLATES:
        for text, cost, _kind in fn(features, goal_text):
            emit(text, cost, None, ("domain_template", tag))

    # Premise-driven.
    for prem in retrieved_premises:
        for text, cost, _kind in _per_premise_templates(
            prem, terms=terms, max_per_premise=max_per_premise,
        ):
            emit(text, cost, prem, ("premise_template",))

    return out
