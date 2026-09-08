"""Proof-body assembly — sketch → tactic block + line map (Phase 4, patch 3).

Third patch of the behaviour-preserving one-responsibility-per-patch
extraction of `proof_dag.py`: the assembly layer moves here — the
fallback-ladder wrapping (`first | (<orig>) | …`), tactic inlining, and
`assemble_proof_with_map`, which builds the body under `theorem T := by`
plus the segment line map that error attribution consumes. `proof_dag`
re-imports everything (facade); call sites unchanged. Duck-typed over the
sketch (`.setup` / `.haves` / `.closer`, have `.id`/`.type_text`/
`.tactic`) so this module needs no dataclass imports; segment ids come
from `.model` (consistency-tested against proof_dag's mirrors). No Lean,
no LLM.
"""
from __future__ import annotations

import re

from .model import SETUP_ID, CLOSER_ID

# Canned closers appended behind the LLM's tactic via `first | … | …`.
#
# This catches the common failure mode where the LLM picks the wrong
# arithmetic closer (e.g. `linarith` when the goal needs `nlinarith`,
# or `nlinarith` when `positivity` would have done it). It cannot
# rescue a leaf whose `type` is mathematically false — that's the
# right failure to surface.
DEFAULT_LEAF_FALLBACKS: tuple[str, ...] = (
    "nlinarith", "linarith", "omega", "norm_num",
    "positivity", "ring_nf", "ring", "field_simp",
    "simp_all", "aesop", "decide",
)


# Lines whose meaning depends on layout/structure: semicolon-joining them
# produces syntax errors (`unexpected token ';'` — observed on
# amc12a_2020_p9). Conservative: any hit disables inlining for the block.
#
# The first two alternatives test a line's START and its END. That is not
# enough on its own, because the SAME constructs are hazardous wherever
# they sit — and a model that emits its tactic already on one line puts
# them in the middle, where a start/end test cannot see them. Hence the
# third alternative, which is position-free:
#
#   `:= by`  opens a tactic block that greedily swallows every following
#            `; `-joined part, so the parts after it silently become part
#            of proving the `have` instead of the goal;
#   `; ·`    is a focus bullet after a semicolon, which is not valid Lean
#            at all — bullets are layout-structured.
#
# Measured on `putnam_1982_b4` (run `p1982b4_v1`, 7921 s, FAILED): the
# assembled body carried three `:= by` on one 451-char ladder line (char
# cols 80/154/226) and two `; ·` on another of 542 chars. Lean reported
# `unexpected token 'by'; expected '{' or tactic` — 14 times, across 3
# sketches and 9 abduction rounds, while the mathematics underneath
# changed completely each round. The repair loop cannot escape this: it
# reads an ASSEMBLER syntax error as a MODEL syntax error, asks for a new
# tactic, and re-flattens the answer into the same corruption.
#
# Failing the guard is cheap by design — `_wrap_with_fallback` then
# returns the model's tactic verbatim, which its own docstring already
# calls the better trade.
_NON_INLINABLE_LINE_RE = re.compile(
    r"^\s*(?:calc\b|·|\.\s|case\b|match\b|conv\b|next\b|\|)"
    r"|(?::=|\bby|=>|\bwith)\s*$"
    r"|:=\s*by\b|;\s*·"
)


def _inlinable(tactic: str) -> bool:
    """Whether a multi-line tactic block can be safely collapsed into a
    single `; `-joined line for use inside `first | (…) | …`."""
    return not any(
        _NON_INLINABLE_LINE_RE.search(line)
        for line in tactic.splitlines() if line.strip()
    )


def _inline_tactic_body(tactic: str) -> str:
    """Collapse a multi-line tactic into a single semicolon-joined line
    so it can be wrapped in `first | (…) | fallback | …`. Lean's
    `first` combinator expects single-tactic alternatives; parenthesised
    tactic blocks (`(t1; t2)`) count as one. Blank lines and pure-
    comment lines are skipped.

    `nlinarith [sq_nonneg (a-b), sq_nonneg (b-c)]` on one line stays
    one line. `have h := X; rw [h]; omega` becomes `have h := X; rw [h]; omega`.

    A line break INSIDE a bracket/paren/anonymous-constructor is a
    continuation, not a new tactic, and must be rejoined with a SPACE.
    Joining it with `;` puts a statement separator inside an argument
    list and the result does not parse:

        nlinarith [sq_nonneg (a - b),
          sq_nonneg (b - c)]
        -> nlinarith [sq_nonneg (a - b),; sq_nonneg (b - c)]

    MEASURED on `lrs_a0_unit_open_v1` (2026-08-26): that corruption is
    what Lean reports as `unknown tactic`, and it killed the run — three
    satisfaction-gate probes and the final verify all died at it, each
    time on a `<line>:43: unknown tactic` immediately after an
    `unsolved goals` on the line above. `_NON_INLINABLE_LINE_RE` does
    not catch it: no line starts with a bullet or ends with `by`/`:=`,
    so the block looks perfectly inlinable. Multi-line `simp only [...]`
    and `refine ⟨_, _, _⟩` are the common shapes, both frequent on
    Polynomial goals.
    """
    if not tactic:
        return ""
    out_parts: list[str] = []
    depth = 0
    for raw in tactic.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("--"):
            continue
        # Drop trailing inline comments.
        cidx = s.find(" --")
        if cidx >= 0:
            s = s[:cidx].rstrip()
        if not s:
            continue
        if out_parts and depth > 0:
            # Still inside a delimiter opened on an earlier line.
            out_parts[-1] = f"{out_parts[-1]} {s}"
        else:
            out_parts.append(s)
        depth = max(0, depth + _delimiter_delta(s))
    # Use `;` between parts; tactic-mode allows it.
    return "; ".join(out_parts)


_OPENERS = "([{⟨⦃"
_CLOSERS = ")]}⟩⦄"


def _delimiter_delta(line: str) -> int:
    """Net bracket depth a line contributes, ignoring string literals.

    Counts the delimiters Lean tactic arguments actually use, including
    the anonymous-constructor brackets `⟨⟩` that `refine`/`exact` lean
    on and the strict-implicit `⦃⦄`.
    """
    delta = 0
    in_str = False
    prev = ""
    for ch in line:
        if in_str:
            if ch == '"' and prev != "\\":
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in _OPENERS:
            delta += 1
        elif ch in _CLOSERS:
            delta -= 1
        prev = ch
    return delta


# Goals over continuous domains where `decide` can never succeed — it
# either fails Decidable-instance synthesis or grinds on kernel
# evaluation until the heartbeat/wall limit. Observed on putnam_1966_b5
# (EuclideanSpace goals): the ladder's `decide` burned the entire 600s
# verify budget, so the repair loop got a bare timeout instead of a
# line-mapped Lean error. Dropping it costs nothing: `decide` cannot
# close real/complex-valued propositions.
_UNDECIDABLE_GOAL_RE = re.compile(
    r"ℝ|ℂ|Real|Complex|EuclideanSpace|NNReal"
)


def _wrap_with_fallback(
    tactic: str,
    fallbacks: tuple[str, ...] = DEFAULT_LEAF_FALLBACKS,
    goal: str = "",
) -> str:
    """Wrap a tactic in `first | (<orig>) | <fallback1> | …`.

    Returns the LLM's tactic verbatim when no fallbacks are configured
    (so the diff vs the no-fallback path is opt-in), or when the tactic
    is structure-sensitive (calc / bullets / match / trailing `by`) —
    inlining those breaks syntax, and losing the LLM's tactic to gain a
    fallback ladder is a bad trade.

    When `goal` (the leaf's type text) mentions a continuous domain,
    `decide` is dropped from the ladder — see _UNDECIDABLE_GOAL_RE."""
    if not fallbacks:
        return tactic
    if goal and _UNDECIDABLE_GOAL_RE.search(goal):
        fallbacks = tuple(f for f in fallbacks if f != "decide")
        if not fallbacks:
            return tactic
    if tactic.strip() and not _inlinable(tactic):
        return tactic
    # EVERY alternative is parenthesised, fallbacks included. A ladder
    # entry containing a combinator — `interval_cases a <;> norm_num at *`
    # is one the LLM ladder generator actually produced — otherwise splices
    # in bare, and `<;>` then swallows the following `| decide`, so the
    # whole `first` block fails to parse. Lean reports that as
    # `unknown tactic` (or `unexpected token ';'`), pointing at the ladder
    # rather than at the tactic that caused it: 8 such failures in
    # putnam_easy8_sol_v1 and more in v2, diagnosed only once
    # --debug-dump-lean kept the assembled source. Parenthesising a simple
    # tactic is a no-op, so this is safe for every existing ladder.
    inlined = _inline_tactic_body(tactic)
    if not inlined:
        # Empty LLM tactic — fallbacks alone.
        alts = " | ".join(f"({f})" for f in fallbacks)
        return f"first | {alts}"
    alts = " | ".join([f"({inlined})", *(f"({f})" for f in fallbacks)])
    return f"first | {alts}"


def assemble_proof_with_map(
    sketch,
    *,
    leaf_fallbacks: tuple[str, ...] = (),
    closer_fallback: bool = False,
) -> tuple[str, list[tuple[str, int, int]]]:
    """Build the tactic body that goes under `theorem T := by`, plus a
    line map [(segment_id, start_line, end_line)] with 1-indexed
    inclusive line numbers WITHIN the body. Segment ids are have ids,
    plus CLOSER_ID for the closer block — this is what lets a Lean
    error at file line N be attributed to one leaf.

    When `leaf_fallbacks` is non-empty, each have's tactic is wrapped
    in `first | (<llm-tac>) | <fallback1> | …` so Lean tries the LLM's
    suggestion first and falls through to canned closers on failure.
    Same for the closer when `closer_fallback=True`.
    """
    lines: list[str] = []
    segments: list[tuple[str, int, int]] = []
    if getattr(sketch, "setup", None):
        start = len(lines) + 1
        for s in sketch.setup:
            for sl in s.splitlines():
                lines.append(f"  {sl.rstrip()}")
        segments.append((SETUP_ID, start, len(lines)))
    for h in sketch.haves:
        tac = h.tactic
        if leaf_fallbacks:
            tac = _wrap_with_fallback(tac, leaf_fallbacks, goal=h.type_text)
        start = len(lines) + 1
        # Multi-line tactic: indent every line under `by`.
        tac_lines = tac.splitlines() or [""]
        lines.append(f"  have {h.id} : {h.type_text} := by")
        for tl in tac_lines:
            lines.append(f"    {tl.rstrip()}")
        segments.append((h.id, start, len(lines)))
    # Closer: optional fallback wrap.
    closer = sketch.closer
    if closer_fallback and leaf_fallbacks:
        closer = _wrap_with_fallback(closer, leaf_fallbacks)
    closer_start = len(lines) + 1
    closer_lines = closer.splitlines() or [""]
    # A leading bare `by` is always spurious: this body is spliced under
    # the theorem's own `:= by`, so a closer that opens with `by` puts a
    # second one at body level. Observed twice on p1967a2_v3
    # (`007_verify.lean`, `014_verify.lean`), each costing a full compile.
    while closer_lines and closer_lines[0].strip() in ("by", "by\r"):
        closer_lines = closer_lines[1:]
    for cl in closer_lines:
        cl_stripped = cl.rstrip()
        if not cl_stripped:
            lines.append("")
            continue
        # EVERY line gets the base indent, exactly as the `have` bodies
        # above do — never the model's own indentation verbatim.
        #
        # The old form kept an already-indented line as-is, which silently
        # broke any closer containing a nested `have … := by`: the `have`
        # (unindented, so it got the base 2) and its body (indented 2 by
        # the model, so kept at 2) landed at the SAME column, and Lean read
        # the `by` block as empty —
        #     error: expected '{' or indented tactic sequence
        # Relative indentation is preserved either way; only the base
        # differs, and without it the block is not under `:= by` at all.
        #
        # Measured on p1967a2_v3 (6100 s, FAILED): six occurrences across
        # four satisfaction-gate probes. Not one lemma reached the kernel,
        # and the error feedback then drove sketch 3 to ABANDON a correct
        # generating-function architecture for 1x1-matrix trivia — an
        # assembler defect degrading the sketch policy, not just a round.
        lines.append("  " + cl_stripped)
    segments.append((CLOSER_ID, closer_start, len(lines)))
    return "\n".join(lines), segments


def assemble_proof(
    sketch,
    *,
    leaf_fallbacks: tuple[str, ...] = (),
    closer_fallback: bool = False,
) -> str:
    body, _ = assemble_proof_with_map(
        sketch, leaf_fallbacks=leaf_fallbacks, closer_fallback=closer_fallback,
    )
    return body
