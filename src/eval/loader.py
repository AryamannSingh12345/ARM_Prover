"""Generalized problem loader — one .lean file in, (statement, src) out.

Replaces the old hardcoded pattern (exactly one declaration, spelled
`theorem`, ending `:= by sorry`, everything above it discarded except
`open`/`set_option` lines). Generality rules:

  - Declaration keyword: theorem / lemma / def / instance / example /
    abbrev, with optional `noncomputable` / `private` / `protected`.
  - Proof placeholder: `:= by sorry`, `:= sorry`, or `:= by\\n  sorry`.
  - Multiple declarations: the LAST sorry-terminated declaration is the
    problem; everything before it is PRELUDE and is carried verbatim
    (minus import lines and comments) into prompts and verification —
    abbrevs, variables, universes, notation, opens, set_options all
    survive. This is what makes PutnamBench answer-construction files
    (an `abbrev putnam_x_solution := …` above the theorem) provable.
  - Comments (nested `/- -/` blocks and `--` lines) are stripped before
    matching, so prose containing the word "theorem" cannot corrupt the
    header. (Limitation: `--` inside a string literal is treated as a
    comment; statements with such literals are vanishingly rare.)

Returned `header` = prelude + declaration-without-placeholder; callers
append `:= by <proof>` and imports. `import` lines are stripped from the
prelude because import selection is the runner's job (LLM-resolved by
default).
"""
from __future__ import annotations

import re
from pathlib import Path

_DECL_START_RE = re.compile(
    r"(?m)^[ \t]*(?:@\[[^\]]*\][ \t\r\n]*)*"
    r"(?:noncomputable[ \t]+|private[ \t]+|protected[ \t]+)*"
    r"(?:theorem|lemma|def|abbrev|instance|example)\b")
_DOCSTRING_RE = re.compile(r"/--.*?-/", re.S)
_PLACEHOLDER_TAIL_RE = re.compile(r":=\s*(?:by\s+)?sorry\s*\Z", re.S)
_IMPORT_LINE_RE = re.compile(r"(?m)^[ \t]*import\s+\S+[ \t]*\r?\n?")


def strip_comments(s: str) -> str:
    """Remove Lean comments: nested `/- … -/` blocks and `--` lines."""
    out: list[str] = []
    i, depth, n = 0, 0, len(s)
    while i < n:
        if s.startswith("/-", i):
            depth += 1
            i += 2
            continue
        if depth:
            if s.startswith("-/", i):
                depth -= 1
                i += 2
            else:
                i += 1
            continue
        if s.startswith("--", i):
            j = s.find("\n", i)
            i = n if j == -1 else j
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


def extract_statement(src: str) -> tuple[str, str] | None:
    """Return (header, prelude) or None.

    `header` is prelude + declaration with the sorry-placeholder removed;
    `prelude` alone is also returned for callers that need the split.

    Semantic context: when the source carries docstrings (`/-- … -/`),
    the one nearest the problem declaration is re-attached to the header
    AS A COMMENT — legal Lean, invisible to the verifier, but it puts the
    informal statement of the problem in front of the sketching model.
    """
    doc = None
    docs = _DOCSTRING_RE.findall(src)
    if docs:
        doc = docs[-1].strip()
        if len(doc) > 1500:
            doc = doc[:1500] + " … -/"
        # `/--` doc comments are legal ONLY immediately before a
        # declaration. Re-attached at the top of the header they are
        # followed by set_option/open/prelude lines, which kills the
        # whole file at parse time ("unexpected token 'set_option';
        # expected 'lemma'" — every risingsea_seeded_v2 verify died
        # this way). Demote to a plain `/- … -/` block comment, which
        # is legal anywhere and identical for the prompting model.
        if doc.startswith("/--"):
            doc = "/-" + doc[3:]
    text = strip_comments(src)
    text = _IMPORT_LINE_RE.sub("", text)
    starts = list(_DECL_START_RE.finditer(text))
    for m in reversed(starts):
        decl = text[m.start():]
        tail = _PLACEHOLDER_TAIL_RE.search(decl)
        if tail is None:
            continue
        head = decl[:tail.start()].rstrip()
        if not head:
            continue
        prelude = text[:m.start()].strip("\n")
        prelude = "\n".join(
            ln.rstrip() for ln in prelude.splitlines() if ln.strip())
        parts = [p for p in (doc, prelude, head) if p]
        header = "\n\n".join(parts)
        return header, prelude
    return None


def load_problem(name: str, bench_dir: Path) -> tuple[str, str]:
    """Return (header, full_source). Raises ValueError when no
    sorry-terminated declaration is found."""
    path = Path(bench_dir) / f"{name}.lean"
    src = path.read_text(encoding="utf-8")
    got = extract_statement(src)
    if got is None:
        raise ValueError(f"no sorry-terminated declaration in {path}")
    header, _prelude = got
    return header, src
