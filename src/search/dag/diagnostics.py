"""Lean-diagnostic text parsing — pure, dependency-free (Phase 4 extraction).

Patches 1–2 of the behaviour-preserving one-responsibility-per-patch
extraction of `proof_dag.py`: the pure error/warning text helpers
(patch 1) and the error→segment attribution layer (`attribute_errors`,
`has_header_level_error`, patch 2) live here. `proof_dag` re-imports
them (facade), so every existing call site — in `proof_dag`,
`backend/repl_step.py`, and the test suite — keeps working unchanged.
No behaviour change; a code-organisation move only. No Lean, no LLM;
segment ids come from `.model` (consistency-tested against proof_dag's
mirrors).
"""
from __future__ import annotations

import re

from .model import CLOSER_ID, HEADER_ID

# A Lean error location: `<file>.lean:<line>:<col>: error[(name)]:`.
_LEAN_ERR_LOC_RE = re.compile(r"\.lean:(\d+):(\d+):\s*error(?:\([^)]*\))?:")

# Any Lean diagnostic marker (error/warning/info), used to slice output
# into blocks so warning spam can be dropped from LLM-bound text.
_DIAG_MARK_RE = re.compile(
    r"(?m)^[^\n]*?\.lean:\d+:\d+:\s*(error|warning|info)(?:\([^)]*\))?:")


def strip_warning_blocks(errors: str) -> str:
    """Drop warning/info diagnostic blocks from Lean output destined for
    LLM prompts, keeping error blocks and any leading non-diagnostic
    text (e.g. `timeout …`). Deprecation warnings (`Finset.toSet has
    been deprecated …` with multi-line Notes) dominated repair/prove
    prompts on b5_bare_v1 (+4.7K chars of noise in one block).
    Attribution (`parse_error_locations`) parses raw text and is
    unaffected — apply this only where text meets a prompt."""
    if not errors:
        return errors
    marks = list(_DIAG_MARK_RE.finditer(errors))
    if not marks:
        return errors
    out = [errors[:marks[0].start()]]
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(errors)
        if m.group(1) == "error":
            out.append(errors[m.start():end])
    return "".join(out).strip()


def parse_error_locations(errors: str) -> list[tuple[int, str]]:
    """Return [(file_line, message), ...] for each Lean error."""
    if not errors:
        return []
    out: list[tuple[int, str]] = []
    matches = list(_LEAN_ERR_LOC_RE.finditer(errors))
    for i, m in enumerate(matches):
        line = int(m.group(1))
        if i + 1 < len(matches):
            # Cut at the START OF THE LINE holding the next diagnostic, not
            # at the match itself: `_LEAN_ERR_LOC_RE` begins at `.lean:`,
            # i.e. part-way through the path, so slicing to `.start()` left
            # the next error's path prefix glued onto this message —
            # "unknown tactic\nC:\\Users\\...\\Try_71e7cfefd98f". That junk
            # reached every repair prompt, and `_DIAG_MARK_RE` below could
            # not remove it because its `.lean:` anchor had already been
            # consumed by the slice boundary.
            nl = errors.rfind("\n", 0, matches[i + 1].start())
            end = nl + 1 if nl != -1 else matches[i + 1].start()
        else:
            end = len(errors)
        msg = errors[m.end():end]
        # The slice runs to the next ERROR marker, so intervening
        # warning/info blocks (deprecation spam + multi-line Notes) get
        # glued onto the message — cut at the first diagnostic marker
        # of ANY kind (seen leaking into repair prompts on b5_bare_v3).
        cut = _DIAG_MARK_RE.search(msg)
        if cut:
            msg = msg[:cut.start()]
        out.append((line, msg.strip()))
    return out


# Environment-level failures no tactic repair can touch: missing oleans,
# unknown modules, broken imports. Attributing these to a segment sends
# the repair loop chasing ghosts (observed: 4 closer "repairs" against a
# missing-olean error on amc12a_2020_p25).
_ENV_ERROR_RE = re.compile(
    r"object file .* does not exist|unknown module|bad import|"
    r"unknown package|error: import", re.I,
)


def _prebody_kind(msg: str) -> str:
    """Classify an error located before body line 1:
    'env' (missing olean / unknown module), 'goals' (`unsolved goals`
    reported at the theorem line — the closer's job), or 'header'
    (the statement itself did not parse/elaborate)."""
    if _ENV_ERROR_RE.search(msg):
        return "env"
    if msg.lstrip().lower().startswith("unsolved goals"):
        return "goals"
    return "header"


def attribute_errors(
    errors: str,
    segments: list[tuple[str, int, int]],
    body_line_offset: int,
) -> dict[str, list[str]]:
    """Map Lean errors to segment ids. `body_line_offset` = number of
    file lines before body line 1 (imports + blank + header line(s)).

    Pre-body errors are split three ways: environment errors (missing
    olean / unknown module) are DROPPED — repair can't fix them, and an
    empty result correctly routes the caller to its no-repair path;
    `unsolved goals` at the theorem line goes to CLOSER_ID (the goal
    wasn't closed — regenerating the closer is the right repair); any
    other pre-body error goes to HEADER_ID — the statement itself is
    broken and NO sketch repair can fix it (callers must stop repairing,
    not chase it). In-body errors outside every segment go to CLOSER_ID
    as the conservative catch-all.
    """
    by_segment: dict[str, list[str]] = {}
    for file_line, msg in parse_error_locations(errors):
        body_line = file_line - body_line_offset
        if body_line < 1:
            kind = _prebody_kind(msg)
            if kind == "env":
                continue
            target = CLOSER_ID if kind == "goals" else HEADER_ID
            by_segment.setdefault(target, []).append(msg)
            continue
        target = CLOSER_ID
        for seg_id, start, end in segments:
            if start <= body_line <= end:
                target = seg_id
                break
        by_segment.setdefault(target, []).append(msg)
    return {k: _rank_messages(v) for k, v in by_segment.items()}


#: Diagnostics Lean emits as a COMPANION to a real failure, carrying no
#: information of their own. `unknown tactic` is the one that matters: it
#: is what Lean reports at the position after a nested `by …` block that
#: failed to elaborate inside a `first | …` alternative. The informative
#: error sits at a LOWER column on the same line, e.g.
#:
#:     20:59: error: unknown tactic
#:     20:55: error: unsolved goals
#:             ⊢ IsRelPrime 1 1
#:
#: Callers take `v[0]`, and the artifact is emitted first, so the segment
#: was reported as "unknown tactic" and `⊢ IsRelPrime 1 1` — the entire
#: content of the failure — was discarded before reaching the model. That
#: is what a repair loop was being asked to act on: in
#: putnam_easy8_sol_v1 one lemma burned four identical retries against it,
#: and the class fired eight times across three unrelated lemmas.
_UNINFORMATIVE_MSG_RE = re.compile(r"^\s*unknown tactic\s*$", re.I)


def _rank_messages(msgs: list[str]) -> list[str]:
    """Put informative messages first within a segment.

    Never drops anything: an uninformative message is merely demoted, and
    a segment whose ONLY message is uninformative keeps it (better a weak
    error than none). Order among informative messages is preserved, so
    this is a no-op wherever no artifact is present.
    """
    if len(msgs) < 2:
        return msgs
    good = [m for m in msgs if not _UNINFORMATIVE_MSG_RE.match(m)]
    if not good:
        return msgs
    weak = [m for m in msgs if _UNINFORMATIVE_MSG_RE.match(m)]
    return good + weak


def has_header_level_error(errors: str, body_line_offset: int) -> bool:
    """True when any Lean error sits before body line 1 and is neither
    an environment error nor `unsolved goals` — i.e. the theorem header
    itself failed to elaborate. Callers use this to stop the repair
    loop (a sketch cannot fix its own statement) or to cross-check a
    REPL verdict against the compile oracle."""
    return any(
        (line - body_line_offset) < 1 and _prebody_kind(msg) == "header"
        for line, msg in parse_error_locations(errors)
    )
