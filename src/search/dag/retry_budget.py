"""Failure-class retry budgets for the ARM prove step.

The lemma-proving loop in `proof_dag._abduce_theory` historically gave
every lemma exactly two attempts, regardless of why the first one failed.
The `p25_2021_legacyARM_0729` trace shows why that is the wrong
allocation — 19 failed attempts over 9 distinct lemmas:

* **12 attempts produced no proof at all** ("empty or forbidden proof":
  the model returned nothing, or returned `sorry`). The loop re-asked the
  identical question with `"empty or forbidden proof"` as its only
  feedback and burned a second flagship call to be told the same thing.
  These deserve ONE attempt, then escalation.
* **7 attempts died on mechanical, non-mathematical causes** — an
  unimported `le_of_pow_le_pow_left`, a dangling reference to a lemma the
  theory never proposed, an `unknown tactic`, and a `1/3` that elaborated
  as natural-number division. Each is cheap to fix given the error text,
  and each got only one or two shots.

So the budget must move in BOTH directions: fewer attempts for
non-engagement, more for static errors that the error message itself
tells you how to fix.

Every budget here is a small integer chosen to be defensible rather than
tuned — there is no data to tune against, and inventing knobs that can
only be fitted on the handful of affordable paid runs would be
overfitting. `DEFAULT_BUDGETS` is exposed so an ablation can override it.

Classification reuses `dag.obligations.classify_failure` (the generic,
already-tested classifier) and adds two signals it does not carry:
`unknown_tactic`, which lands in `other` there but is squarely a static
error here, and `empty_or_forbidden`, which is not a Lean diagnostic at
all — it is the absence of one.
"""
from __future__ import annotations

import re

from .obligations import classify_failure

#: Not a Lean diagnostic: the model returned no usable proof text, or one
#: containing a forbidden tactic (`sorry`/`admit`/`native_decide`).
EMPTY_OR_FORBIDDEN = "empty_or_forbidden"

#: `unknown tactic` is mechanical, but `classify_failure` files it under
#: `other` (its unknown-name regex is for identifiers and constants).
UNKNOWN_TACTIC = "unknown_tactic"
_UNKNOWN_TACTIC_RE = re.compile(r"unknown tactic", re.I)

#: A MECHANICAL RESIDUE: the proof got all the way to an arithmetic
#: leftover and an arithmetic decision procedure declined it. This is a
#: near-miss, not a failure of the mathematics.
#:
#: Measured on `p2020a2_recognize_v2`: `aux_negative_binomial_prefix` —
#: the single lemma standing between the run and a solved Putnam A2 —
#: produced two substantial inductions (1.9K and 2.2K of proof text) and
#: died on
#:     ⊢ 1 + ∑ i ∈ range r, n.choose (i+1) + n.choose (r+1)
#:     = ∑ i ∈ range r, n.choose (i+1) + 1 + n.choose (r+1)
#: — pure associativity — and then on `omega could not prove the goal`.
#: Both classified as `unsolved_goals`/`other` and got the mathematical
#: budget of 2, while a mistyped identifier gets 4. That is backwards:
#: the model was one rearrangement from a correct proof and the error
#: text says exactly what is left.
#:
#: Deliberately keyed to the FAILURE MESSAGES OF ARITHMETIC TACTICS, not
#: to goal shape: `omega`/`linarith`/`ring` are only reached once the
#: mathematical work is done, so their complaints are a reliable
#: end-of-proof signal and carry no problem-specific vocabulary.
MECHANICAL_RESIDUE = "mechanical_residue"
_RESIDUE_RE = re.compile(
    r"omega could not prove|linarith failed|nlinarith failed"
    r"|`?ring`? failed|ring_nf failed|`?norm_num`? failed"
    r"|simp made no progress", re.I)

#: Attempts per lemma before the budget mechanism is enabled — the exact
#: legacy behaviour (`for _pa in range(2)`), preserved as the default.
LEGACY_ATTEMPTS = 2

#: Absolute ceiling, so no combination of reclassification across attempts
#: can spin the prove loop.
HARD_CAP = 6

#: Attempts allowed per failure class.
#:   static / mechanical  -> more attempts: the error text says how to fix it
#:   mathematical         -> the legacy two: the model engaged and fell short
#:   non-engagement       -> one, then escalate; re-asking is pure waste
DEFAULT_BUDGETS: dict[str, int] = {
    EMPTY_OR_FORBIDDEN: 1,
    "timeout": 1,
    "unknown_identifier": 4,
    UNKNOWN_TACTIC: 4,
    "parse_error": 4,
    MECHANICAL_RESIDUE: 4,
    "unsolved_goals": 2,
    "type_mismatch": 2,
    "synth_failure": 2,
    "other": 2,
}

#: What the revision ledger tells the model when a lemma exhausts its
#: budget. The theory-revision prompt already receives this ledger, so
#: escalation rides the existing loop rather than adding a mechanism.
_HINTS: dict[str, str] = {
    EMPTY_OR_FORBIDDEN: (
        "no proof was produced for it (empty response or `sorry`) — it was "
        "never actually attempted. Do NOT restate it unchanged: SPLIT it "
        "into smaller lemmas, or drop this route"),
    "timeout": (
        "proof search timed out — split it into smaller steps rather than "
        "restating it"),
    "unknown_identifier": (
        "it referenced a name that does not resolve in this Mathlib pin — "
        "check the name exists, or state the fact you need as its own lemma"),
    UNKNOWN_TACTIC: (
        "the proof used a tactic Lean does not know — rewrite it with "
        "standard Mathlib tactics"),
    "parse_error": (
        "the proof did not parse — fix the syntax"),
    MECHANICAL_RESIDUE: (
        "the proof reached an arithmetic leftover that `omega`/`ring`/"
        "`linarith` would not close — that is a REARRANGEMENT away, not a "
        "new idea. Normalise the leftover explicitly (`ring_nf`, `ac_rfl`, "
        "`Nat.add_comm`/`add_assoc` rewrites) rather than restating the "
        "lemma"),
    "unsolved_goals": (
        "the proof was attempted but left goals open — either finish those "
        "goals or split them out as their own lemmas"),
    "type_mismatch": (
        "types did not line up — check the statement's coercions and "
        "numeric literals (a `1/3` over ℕ elaborates to 0)"),
    "synth_failure": (
        "an instance could not be synthesized — the statement may need "
        "extra typeclass hypotheses"),
}


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split on `sep` at bracket depth 0."""
    parts, cur, depth = [], [], 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "([{⟨":
            depth += 1
        elif ch in ")]}⟩":
            depth -= 1
        if depth == 0 and text.startswith(sep, i):
            parts.append("".join(cur))
            cur = []
            i += len(sep)
            continue
        cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return [" ".join(p.split()) for p in parts]


def is_ac_rearrangement(error: str | None) -> bool:
    """Is the residual goal the SAME TERMS in a different order?

    `⊢ 1 + S + c = S + 1 + c` is not a mathematical gap; it is
    associativity/commutativity, and the proof around it is finished. A
    lemma that fails this way is one `ring_nf`/`ac_rfl` from closing, so
    it deserves the mechanical budget rather than the mathematical one.

    Precise by construction: it fires ONLY when both sides split into the
    same multiset of top-level `+` operands, which cannot happen for a
    genuinely open goal. Splitting respects brackets, so the `+` inside
    `n.choose (i + 1)` does not create a spurious operand.
    """
    text = error or ""
    cut = text.rfind("⊢")
    if cut < 0:
        return False
    goal = text[cut + 1:].strip()
    sides = _split_top_level(goal, "=")
    if len(sides) != 2:
        return False
    lhs, rhs = (_split_top_level(s, "+") for s in sides)
    if len(lhs) < 2 or len(rhs) < 2:
        return False
    return sorted(lhs) == sorted(rhs)


def classify_prove_failure(error: str | None, *,
                           empty_or_forbidden: bool = False) -> list[str]:
    """Failure classes for one lemma-proof attempt.

    `empty_or_forbidden` is passed by the caller because it is not
    derivable from a Lean error blob — there is no compile to read.
    """
    if empty_or_forbidden:
        return [EMPTY_OR_FORBIDDEN]
    classes, _unknowns = classify_failure(error)
    text = error or ""
    if _UNKNOWN_TACTIC_RE.search(text):
        classes = [c for c in classes if c != "other"] + [UNKNOWN_TACTIC]
    if _RESIDUE_RE.search(text) or is_ac_rearrangement(text):
        classes = [c for c in classes if c != "other"] + [MECHANICAL_RESIDUE]
    return classes or ["other"]


def budget_for(classes: list[str],
               budgets: dict[str, int] | None = None) -> int:
    """Attempts allowed for a failure carrying `classes`.

    The MAXIMUM over classes, not the minimum: an error that is both
    `unknown_identifier` and `unsolved_goals` is usually one unresolved
    name causing downstream open goals, and is worth the static budget.
    """
    table = budgets if budgets is not None else DEFAULT_BUDGETS
    if not classes:
        return min(table.get("other", LEGACY_ATTEMPTS), HARD_CAP)
    return min(max(table.get(c, LEGACY_ATTEMPTS) for c in classes), HARD_CAP)


def ledger_hint(classes: list[str]) -> str:
    """Class-specific guidance appended to the PROOF FAILED ledger entry,
    or "" when no class has specific advice (caller keeps its generic
    wording)."""
    for c in classes:
        if c in _HINTS:
            return _HINTS[c]
    return ""
