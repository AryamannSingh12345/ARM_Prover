"""Theorem-statement features for the corpus miner.

Spec: PART 5 of the Mathlib-prior series.

`extract_state_features` is built around live REPL goals (Lean prints
hypotheses followed by `⊢ <target>`). Mathlib theorem statements have
the same essential structure but written differently:

    theorem foo (a : ℕ) (h : 0 < a) : a < a + 1 := by ...

We normalise the statement into an `⊢`-anchored pseudo-goal and reuse
the same extractor. That keeps a single source of truth for the feature
bag (no second namespace/symbol table to drift out of sync) and lets
priors mined from Mathlib share feature keys with priors mined from
solved JSONL logs.
"""
from __future__ import annotations

import re

from search.proof_prior import ProofStateFeatures
from search.state_features import extract_state_features


_THEOREM_HEAD_RE = re.compile(
    r"^\s*(?:theorem|lemma|example|def)\b\s+\S+\s*",
)


def _statement_to_pseudo_goal(statement_text: str) -> str:
    """Best-effort rewrite of a theorem statement into a `⊢`-anchored
    pseudo-goal string. The shape `(h₁ : T) ... : Target` becomes
    `h₁ : T\\n...\\n⊢ Target` so `extract_state_features` can split it
    along its usual hypothesis / target boundary.

    On failure (no `:` found, malformed signature) we hand back the raw
    statement — the extractor degrades gracefully to a target-only bag.
    """
    if not statement_text:
        return ""
    # Drop the leading `theorem foo` / `lemma foo` prefix.
    head = _THEOREM_HEAD_RE.sub("", statement_text, count=1)
    # Find the rightmost top-level `:` that introduces the target. We
    # walk left to right tracking paren / bracket / brace nesting; the
    # candidate is the LAST `:` at depth 0.
    target_colon = -1
    depth = 0
    i = 0
    while i < len(head):
        c = head[i]
        if c in "([{⟨":
            depth += 1
        elif c in ")]}⟩":
            if depth > 0:
                depth -= 1
        elif c == ":" and depth == 0:
            # Skip `:=` — that's the proof body marker, not the typing colon.
            if head[i:i + 2] == ":=":
                break
            target_colon = i
        i += 1
    if target_colon < 0:
        return head.strip()
    hypotheses_block = head[:target_colon]
    target_block = head[target_colon + 1:]
    # Strip a trailing `:= ...` if it slipped through (e.g. `Prop := by`).
    target_block = re.split(r":=\s*by\b|:=\s*sorry\b|:=\s*$", target_block,
                            maxsplit=1)[0]
    # Hypotheses come in `(name : type)` / `{name : type}` chunks. Render
    # each on its own line so the extractor's split-on-`⊢` works.
    hyp_pieces = re.findall(r"[({\[⟨]([^()\[\]{}⟨⟩]*?:[^()\[\]{}⟨⟩]*?)[)}\]⟩]",
                            hypotheses_block)
    if hyp_pieces:
        hyp_text = "\n".join(p.strip() for p in hyp_pieces)
        return f"{hyp_text}\n⊢ {target_block.strip()}"
    return f"⊢ {target_block.strip()}"


def extract_theorem_features(
    statement_text: str,
    proof_prefix: list[str] | None = None,
    retrieved_premises: list[str] | None = None,
) -> ProofStateFeatures:
    """Build a `ProofStateFeatures` from a Mathlib theorem statement.

    Wraps `extract_state_features` after rewriting the statement into a
    pseudo-goal. Returns an empty-but-valid bag for empty input."""
    pseudo = _statement_to_pseudo_goal(statement_text or "")
    return extract_state_features(
        pseudo,
        list(proof_prefix or []),
        retrieved_premises=list(retrieved_premises or []) or None,
    )
