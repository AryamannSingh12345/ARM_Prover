"""Conservative Lean 4 declaration + tactic-script extractor.

Spec: PARTS 3+4 of the Mathlib-prior series.

The goal is to mine proof *moves* from a Lean source file without a real
parser. We accept some recall loss in exchange for soundness:

  - We only extract `theorem`, `lemma`, `example` (term `def`s are skipped
    by default — they rarely contain reusable tactic scripts).
  - We only emit a `proof_text` when the proof body is tactic-mode
    (`:= by ...`); term-mode proofs are returned with `proof_kind="term"`
    and an empty `proof_text` so the caller can skip them cleanly.
  - Tactic splitting is indentation-based; deeper-indented lines are
    treated as continuations of the previous tactic. Bullet markers
    (`·`, `case ...`), comments, and brace-only lines are skipped.

This is not a Lean parser. Anything ambiguous is dropped, not guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LeanDeclaration:
    declaration_name: str
    file_path: str
    module_guess: str
    declaration_kind: str    # "theorem" | "lemma" | "example" | "def"
    statement_text: str
    proof_text: str          # "" when proof_kind != "tactic"
    proof_kind: str          # "tactic" | "term" | "sorry" | "unknown"


# ---------------------------------------------------------------------------
# Comment + module helpers
# ---------------------------------------------------------------------------

_LINE_COMMENT_RE = re.compile(r"--[^\n]*")


def _strip_block_comments(text: str) -> str:
    """Strip non-nested `/- ... -/` blocks. Nested blocks are rare in
    Mathlib proof bodies; the few we hit get partially stripped, which
    only loses extra tokens — never widens a real declaration's bounds."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i:i + 2] == "/-":
            depth = 1
            j = i + 2
            while j < n and depth > 0:
                if text[j:j + 2] == "/-":
                    depth += 1
                    j += 2
                elif text[j:j + 2] == "-/":
                    depth -= 1
                    j += 2
                else:
                    j += 1
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _module_guess(file_path: str) -> str:
    """Derive a Mathlib-style module name from a path.

    `.../Mathlib/Data/Nat/GCD/Basic.lean` -> `Mathlib.Data.Nat.GCD.Basic`.
    Falls back to the bare stem when no `Mathlib` ancestor is present.
    """
    p = Path(file_path)
    parts = p.with_suffix("").parts
    if "Mathlib" in parts:
        idx = parts.index("Mathlib")
        return ".".join(parts[idx:])
    return p.stem


# ---------------------------------------------------------------------------
# Declaration scan
# ---------------------------------------------------------------------------

# A leading declaration line: optional `@[attrs]` on prior lines were
# already handled (we just key off the keyword). Modifiers before the
# keyword: `private`, `protected`, `noncomputable`. The name is allowed
# to contain unicode letters; we keep the character class permissive.
_MOD_PREFIX = r"(?:private\s+|protected\s+|noncomputable\s+|@\[[^\]]*\]\s+)*"
_DECL_LINE_RE = re.compile(
    rf"^(?P<indent>\s*){_MOD_PREFIX}"
    r"(?P<kind>theorem|lemma|example|def)\b"
    r"(?:\s+(?P<name>[^\s:({]+))?",
)
# Lines that re-anchor at top level (force end of previous decl body).
_TOP_LEVEL_ANCHORS = re.compile(
    r"^\s*("
    r"theorem|lemma|example|def|"
    r"instance|class|structure|inductive|abbrev|"
    r"namespace|end|section|"
    r"variable|universe|"
    r"@\[|"
    r"public\s+section|public\s+import|"
    r"open|import|set_option|attribute|"
    r"#check|#eval|#print"
    r")\b",
)


def _split_statement_proof(decl_text: str) -> tuple[str, str, str]:
    """Return (statement_text, proof_text, proof_kind).

    Recognises `:= by`, plain `by`, `:= sorry`, and term-mode `:=`.
    For tactic-mode, `proof_text` starts AFTER the `:= by` marker.
    """
    # Anchor on `:= by` first (most specific), then `:= sorry`, then `:=`,
    # then a bare `by` (some legacy lemmas).
    for marker, kind, take_after in (
        (":= by", "tactic", True),
        (":=  by", "tactic", True),
        (":= sorry", "sorry", False),
        (":=", "term", False),
    ):
        idx = decl_text.find(marker)
        if idx == -1:
            continue
        stmt = decl_text[:idx].rstrip()
        if take_after:
            body = decl_text[idx + len(marker):]
            # Drop one leading newline if present so indent math works.
            if body.startswith("\n"):
                body = body[1:]
            return stmt, body, kind
        return stmt, "", kind
    return decl_text.rstrip(), "", "unknown"


def _is_decl_continuation(line: str, decl_indent: int) -> bool:
    """A non-empty line is *inside* the current decl when:
      - it is indented strictly deeper than the decl's first line, OR
      - it does not start a new top-level form at <= decl_indent.
    Empty lines are inside by default; the caller stops when a new
    top-level form fires.
    """
    stripped = line.lstrip()
    if not stripped:
        return True
    this_indent = len(line) - len(stripped)
    if this_indent > decl_indent:
        return True
    return not _TOP_LEVEL_ANCHORS.match(line)


def extract_lean_declarations(text: str, file_path: str) -> list[LeanDeclaration]:
    """Return every `theorem|lemma|example|def` declaration in `text`."""
    text = _strip_block_comments(text)
    # Strip line comments line-by-line so column counts inside string
    # literals are unaffected (we never see Lean string literals in
    # declarations of interest).
    lines = [_LINE_COMMENT_RE.sub("", ln) for ln in text.splitlines()]
    decls: list[LeanDeclaration] = []
    module = _module_guess(file_path)
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        m = _DECL_LINE_RE.match(line)
        if not m or not m.group("name"):
            i += 1
            continue
        kind = m.group("kind")
        name = m.group("name")
        decl_indent = len(m.group("indent"))
        # Walk until next top-level anchor at <= decl_indent (exclusive).
        end = i + 1
        while end < n:
            if not _is_decl_continuation(lines[end], decl_indent):
                break
            end += 1
        decl_text = "\n".join(lines[i:end])
        stmt, proof, proof_kind = _split_statement_proof(decl_text)
        decls.append(LeanDeclaration(
            declaration_name=name,
            file_path=str(file_path),
            module_guess=module,
            declaration_kind=kind,
            statement_text=stmt,
            proof_text=proof,
            proof_kind=proof_kind,
        ))
        i = end
    return decls


# ---------------------------------------------------------------------------
# Tactic splitter
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(r"^[·•\-]\s*$|^case\b")
# Leading focused-bullet marker (`· simp [...]`) — we strip the `·` and
# keep the tactic that follows so classify_tactic sees `simp`, not the
# bullet character.
_BULLET_PREFIX_RE = re.compile(r"^[·•]\s+")
_BRACE_ONLY_RE = re.compile(r"^[{}⟨⟩()\[\]]+$")
_LINE_COMMENT_PREFIX_RE = re.compile(r"^\s*--")
# Tactic-line prefix allow-list: a line whose first token is one of these is
# always treated as a NEW tactic (even if its indent is deeper than the
# base — which happens after a `·` bullet). The list is the same set the
# spec calls out plus a few common-in-Mathlib extras.
_TACTIC_PREFIXES = frozenset({
    "intro", "intros", "constructor", "rcases", "obtain", "have", "suffices",
    "rw", "rewrite", "simp", "simp_all", "norm_num", "ring_nf", "ring",
    "field_simp", "linarith", "nlinarith", "omega", "exact", "apply",
    "aesop", "positivity", "use", "by_cases", "decide", "native_decide",
    "rfl", "trivial", "assumption", "contradiction", "show", "refine",
    "calc", "split", "cases", "induction", "subst", "convert",
    "exact_mod_cast", "push_cast", "norm_cast", "tauto", "polyrith",
})


def _is_tactic_starting_line(stripped: str) -> bool:
    """True iff the line's first token is in our allow-list."""
    head = stripped.split(maxsplit=1)[0] if stripped else ""
    head = head.rstrip(";")
    return head in _TACTIC_PREFIXES


def split_tactic_script(proof_text: str) -> list[str]:
    """Approximate tactic-by-tactic split of a `by`-block body.

    Strategy:
      1. Skip empty lines, bullet markers, brace-only lines, comments.
      2. Establish the base indent from the first eligible line.
      3. A new tactic starts on any line whose indent equals the base
         OR whose first token is in `_TACTIC_PREFIXES`. Deeper-indented
         lines whose first token is *not* a tactic prefix are appended
         to the previous tactic as a continuation.
      4. Trim each tactic.

    Returns an ordered list of one-line tactic strings. May be empty.
    """
    if not proof_text:
        return []
    raw = proof_text
    # Drop a leading `by` keyword if the caller passed the full RHS.
    s = raw.lstrip()
    if s.startswith("by"):
        # Eat `by` plus any spaces/newlines after.
        after = s[2:]
        # Keep going from the first newline (if `by` was on its own line)
        # or just continue (if `by` was followed by inline tactics).
        raw = after

    lines = raw.splitlines()
    base_indent: int | None = None
    candidates: list[tuple[int, str]] = []  # (indent, stripped_text)
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            continue
        if _LINE_COMMENT_PREFIX_RE.match(line):
            continue
        if _BULLET_RE.match(stripped):
            continue
        if _BRACE_ONLY_RE.match(stripped):
            continue
        # Strip a leading `· ` so the inner tactic classifies cleanly.
        stripped = _BULLET_PREFIX_RE.sub("", stripped)
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        if base_indent is None:
            base_indent = indent
        candidates.append((indent, stripped))

    if not candidates or base_indent is None:
        return []

    tactics: list[str] = []
    current: str | None = None
    for indent, stripped in candidates:
        starts_new = (
            indent <= base_indent
            or _is_tactic_starting_line(stripped)
            or current is None
        )
        if starts_new:
            if current is not None:
                tactics.append(current.strip())
            current = stripped
        else:
            current = (current or "") + " " + stripped
    if current is not None:
        tactics.append(current.strip())
    # Drop empty / single-symbol residue.
    return [t for t in tactics if len(t) >= 2 and not _BRACE_ONLY_RE.match(t)]
