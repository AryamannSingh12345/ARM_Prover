"""Tactic classification + premise extraction.

Spec: PART 5 of the proof-prior series.

`classify_tactic(tactic)` maps a Lean tactic string to a coarse class label.
The labels are used both as a feature (previous_tactic_class) and as a
provenance tag on mined moves.

`extract_used_premises(tactic)` recovers the dotted Mathlib names (and a
small allowlist of bare names) the tactic references. It does NOT
require an external node set — it returns *all* identifier-shaped citations
that look like Mathlib decls. The caller filters.
"""
from __future__ import annotations

import re


# Order matters: the first matching class wins. Patterns are word-anchored
# on the *leading* token so `apply foo` matches `apply`, not `exact`.
#
# `have` and `let` are handled out-of-band by `_classify_have_or_let` —
# the regex below cannot distinguish `have h : T := premise n k`
# (have_premise) from `have h : T := by ...` (have_local) without
# looking at what follows `:=`.
#
# `native_decide` MUST appear before `decide` so the prefix-shorter rule
# does not steal the match. Both are kept here as distinct classes; the
# runner's safety filters continue to reject
# `native_decide` as a forbidden keyword regardless of class label.
_CLASS_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("intro",           re.compile(r"^\s*(?:intro|intros)\b")),
    ("constructor",     re.compile(r"^\s*(?:constructor|refine\s*⟨)")),
    ("cases_or_rcases", re.compile(r"^\s*(?:cases|rcases|cases'|obtain)\b")),
    ("by_cases",        re.compile(r"^\s*by_cases\b")),
    ("rw",              re.compile(r"^\s*(?:rw|rewrite)\b")),
    ("simp",            re.compile(r"^\s*simp(?:_all|_rw)?\b")),
    ("norm_num",        re.compile(r"^\s*norm_num\b")),
    ("omega",           re.compile(r"^\s*omega\b")),
    ("linarith",        re.compile(r"^\s*linarith\b")),
    ("nlinarith",       re.compile(r"^\s*nlinarith\b")),
    ("ring_nf",         re.compile(r"^\s*(?:ring_nf|ring)\b")),
    ("field_simp",      re.compile(r"^\s*field_simp\b")),
    ("positivity",      re.compile(r"^\s*positivity\b")),
    ("aesop",           re.compile(r"^\s*aesop\b")),
    ("native_decide",   re.compile(r"^\s*native_decide\b")),
    ("decide",          re.compile(r"^\s*decide\b")),
    ("apply",           re.compile(r"^\s*apply\b")),
    ("exact",           re.compile(r"^\s*exact\b")),
    ("use",             re.compile(r"^\s*use\b")),
]

# `have <name> [: <type>] := <RHS>` / `let <name> [: <type>] := <RHS>`.
# Capture the FIRST identifier on the RHS. If that identifier is `by`,
# the RHS is a tactic block (have_local); otherwise the RHS starts with
# an external term — almost always a Mathlib lemma application
# (have_premise). A `have` without `:=` (e.g. `have h := ?_` is
# placeholder garbage that earlier safety gates strip; `have h : T := ?_`
# is similar) falls through to have_local.
_HAVE_LET_RE = re.compile(r"^\s*(have|let)\b")
_HAVE_RHS_RE = re.compile(r":=\s*([A-Za-z_][A-Za-z0-9_]*)")


def _classify_have_or_let(t: str) -> str:
    """Return have_premise / have_local for a leading have/let tactic."""
    m = _HAVE_RHS_RE.search(t)
    if m is None:
        return "have_local"
    first_rhs_ident = m.group(1)
    if first_rhs_ident == "by":
        return "have_local"
    return "have_premise"


# ---------------------------------------------------------------------------
# Tactic *shape* — coarser than `tactic_class` and structure-aware.
# ---------------------------------------------------------------------------
#
# The abstract-continuation prior keys on shape, not class. `rw` and
# `rw … at <hyp>` are both class=rw but very different shapes: the
# at-hyp form rewrites a hypothesis while the bare form rewrites the
# target. Distinguishing them lets the generic instantiator know
# whether to scan for a target hypothesis name.
#
# Shapes are a small fixed vocabulary so feature-key digests stay
# small.

_SHAPE_RW_AT_RE   = re.compile(r"^\s*(?:rw|rewrite)\s*\[[^\]]*\]\s+at\s+\S+")
_SHAPE_RW_BARE_RE = re.compile(r"^\s*(?:rw|rewrite)\s*\[")
_SHAPE_SIMP_AT_RE = re.compile(r"^\s*simp(?:_all|_rw)?\s*[^\n]*\s+at\s+\S+")
_SHAPE_SIMP_RE    = re.compile(r"^\s*simp(?:_all|_rw)?\b")
_SHAPE_HAVE_RE    = re.compile(r"^\s*have\b")


def tactic_shape(tactic: str) -> str:
    """Map `tactic` to a coarse structural shape label.

    The shape vocabulary is intentionally small and theorem-agnostic;
    its only consumers are the abstract-continuation prior (mining and
    generic instantiation)."""
    if not tactic:
        return "unknown"
    t = tactic.lstrip()
    t = re.sub(r"^[;<]+\s*", "", t)
    if _SHAPE_RW_AT_RE.match(t):
        return "rw_at_hyp"
    if _SHAPE_RW_BARE_RE.match(t):
        return "rw_bare"
    if _SHAPE_SIMP_AT_RE.match(t):
        return "simp_at_hyp"
    if _SHAPE_SIMP_RE.match(t):
        return "simp_shape"
    cls = classify_tactic_no_recurse(t)
    if cls in ("omega", "linarith", "nlinarith", "norm_num", "polyrith"):
        return "arithmetic_close"
    if cls in ("decide", "native_decide", "aesop"):
        return "decision_close"
    if cls in ("ring_nf", "field_simp"):
        return "algebra_close"
    if cls == "positivity":
        return "positivity_close"
    if cls == "exact":
        return "exact_shape"
    if cls == "apply":
        return "apply_shape"
    if cls == "use":
        return "use_shape"
    if cls == "intro":
        return "intro_shape"
    if cls == "constructor":
        return "constructor_shape"
    if cls == "by_cases":
        return "by_cases_shape"
    if cls == "cases_or_rcases":
        return "cases_shape"
    if _SHAPE_HAVE_RE.match(t):
        # have_premise / have_local both fold to a single shape — the
        # instantiator decides what to do based on local context.
        return "have_shape"
    return "other"


def classify_tactic_no_recurse(t: str) -> str:
    """Internal — avoid the have/let preamble shortcut inside
    `classify_tactic`, used by `tactic_shape` to avoid double work."""
    if _HAVE_LET_RE.match(t):
        return _classify_have_or_let(t)
    for cls, pat in _CLASS_RULES:
        if pat.match(t):
            return cls
    return "unknown"


def classify_tactic(tactic: str) -> str:
    """Map `tactic` to one of the coarse classes; default `unknown`.

    "obtain" maps to cases_or_rcases (it is the irrefutable variant)."""
    if not tactic:
        return "unknown"
    t = tactic.lstrip()
    # Drop a leading `;` / `<;>` combinator artefact, then re-check.
    t = re.sub(r"^[;<]+\s*", "", t)
    if _HAVE_LET_RE.match(t):
        return _classify_have_or_let(t)
    for cls, pat in _CLASS_RULES:
        if pat.match(t):
            return cls
    return "unknown"


# Dotted Mathlib citation (`Nat.gcd_mul_lcm`, `Real.sqrt_pos.2`, etc).
_DOTTED_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_'₀-₉]*)+")

# Bare names that the runner already treats as well-known Lean citations
# even when they are not dotted. Bare-name
# extraction is conservative — we extract only this small allowlist so we
# never pick up local hypothesis names.
_BARE_ALLOWLIST: frozenset[str] = frozenset({
    "sq_nonneg", "add_pos", "mul_pos", "ne_of_gt", "sub_nonneg", "abs_nonneg",
    "div_pos", "div_nonneg", "pow_pos", "pow_nonneg", "neg_nonpos",
    "le_refl", "lt_irrefl", "lt_or_gt_of_ne",
})

_BARE_RE = re.compile(r"(?<![A-Za-z0-9_.])([a-z][a-z0-9_]+)(?![A-Za-z0-9_.])")

# ---------------------------------------------------------------------------
# rw/simp bracket-list scanner (PART A fix)
# ---------------------------------------------------------------------------
#
# `extract_used_premises` historically only picked up dotted identifiers
# + a small bare allow-list. Mathlib's mining diagnostics surfaced rows
# like:
#     rw [← one_mul (lcm m n), ← h.gcd_eq_one, gcd_mul_lcm]
# where the only DOTTED hit (`h.gcd_eq_one`) is a local projection and
# the actual lemma (`gcd_mul_lcm`) is bare — so the extractor missed
# the reusable premise entirely. Fix: parse the rw/simp bracket list
# separately, strip direction markers + args, and skip local-projection-
# shaped names.

_BRACKET_TACTIC_RE = re.compile(
    r"\b(?:simp_all|simp\s+only|simp|rw|rewrite)\s*\["
)
# A name token inside an rw/simp item. Allow dotted ids and bare ids;
# downstream filters trim per item.
_ITEM_NAME_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_'₀-₉]*(?:\.[A-Za-z_][A-Za-z0-9_'₀-₉]*)*"
)
# Leading direction marker on an rw/simp item (`←` or ASCII `<-`).
_DIR_MARKER_RE = re.compile(r"^(?:←|<-)\s*")

# Local-hypothesis prefix: any dotted name whose FIRST segment is a
# short lowercase token (h, n, a, ih, ha, hb, h₀, h1, p, q, …) is a
# projection on a local variable, not a Mathlib citation. Bare names
# whose entire form is short-lowercase are also locals. We keep the
# bound loose (≤ 3 chars) because Mathlib lemma names almost always
# contain an underscore and are far longer.
_LOCAL_HYP_NAME_RE = re.compile(
    r"^(?:h|ih|hh|i)?[a-z][a-z0-9]?(?:[₀-₉]+|\d+)?$"
)


def _is_local_projection(name: str) -> bool:
    """True iff `name` looks like a local-hypothesis projection.

    Examples that return True:
        h, h₀, h1, ha, hb, ih, hp,
        h.gcd_eq_one, a.gcd, b.factorization, n.succ_pos
    Examples that return False:
        Nat.gcd_mul_lcm, Real.sqrt_pos, gcd_mul_lcm, sq_nonneg,
        factorization_mul, add_right_inj
    """
    if not name:
        return True
    first = name.split(".", 1)[0]
    if not first:
        return True
    # A real Mathlib namespace starts with an uppercase letter. Anything
    # else as the first segment is a local variable (Lean convention).
    if first[0].isupper():
        return False
    # Lowercase first segment: classify by length / shape. Mathlib lemma
    # bare-names always contain an underscore (`gcd_mul_lcm`,
    # `add_comm`), so anything ≤ 3 chars or matching the local-hyp
    # regex is treated as local.
    if len(first) <= 3:
        return True
    if _LOCAL_HYP_NAME_RE.match(first):
        return True
    return False


def _split_top_level_commas(content: str) -> list[str]:
    """Split `content` by commas at bracket-depth 0. Handles
    `()`, `[]`, `{}`, `⟨⟩` nesting."""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for c in content:
        if c in "([{⟨":
            depth += 1
            buf.append(c)
        elif c in ")]}⟩":
            depth -= 1
            buf.append(c)
        elif c == "," and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(c)
    if buf:
        out.append("".join(buf))
    return out


def _scan_bracket_lists(tactic: str) -> list[list[str]]:
    """Return one list of (raw, normalised-name) items per rw/simp
    bracket found in `tactic`. The outer list preserves bracket
    occurrence order; the inner list preserves item order inside that
    bracket. Names are already projection-suffix-stripped."""
    out: list[list[str]] = []
    for m in _BRACKET_TACTIC_RE.finditer(tactic):
        start = m.end()
        depth = 1
        i = start
        while i < len(tactic) and depth > 0:
            ch = tactic[i]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            i += 1
        # `i` is one past the matching `]` (or past EOF on unbalanced
        # source — defensive).
        end = i - 1 if i > start else len(tactic)
        content = tactic[start:end]
        items: list[str] = []
        for raw_item in _split_top_level_commas(content):
            item = raw_item.strip()
            if not item:
                continue
            # Strip leading ← / <- direction marker.
            item = _DIR_MARKER_RE.sub("", item).strip()
            # Take the first identifier-shaped token; drop args.
            nm = _ITEM_NAME_RE.match(item)
            if not nm:
                continue
            name = nm.group(0)
            for suffix in (".mp", ".mpr", ".symm", ".1", ".2"):
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            items.append(name)
        out.append(items)
    return out


def extract_rw_simp_lemmas(tactic: str) -> list[str]:
    """Return the non-local lemma names found inside every rw/simp
    bracket in `tactic`. Order preserved across brackets; deduped.

    This is the SAME data the per-bracket scan inside
    `extract_used_premises` uses, exposed as a separate API so the
    Mathlib miner can tag premise-prior rows with `rw_lemma`."""
    if not tactic:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for bracket in _scan_bracket_lists(tactic):
        for name in bracket:
            if _is_local_projection(name) or name in seen:
                continue
            seen.add(name)
            out.append(name)
    return out


def extract_used_premises(tactic: str) -> list[str]:
    """Return Mathlib decl names cited by `tactic`, in order of priority.

    Sources (each filters out local-hypothesis projections):
      1. rw / simp / simp_all / simp only / rewrite bracket lists.
         Lemmas inside `[...]` are extracted with direction markers and
         arguments stripped. This catches the bare-name case
         (`rw [gcd_mul_lcm]`) that the dotted-only scan missed.
      2. RHS of `:=` (if present). The applied theorem in
         `have h : T := Nat.gcd_mul_lcm n k` lives here; the dotted
         names in `T` are noise.
      3. LHS of `:=` (or the whole tactic when no `:=`).
      4. Bare `_BARE_ALLOWLIST` hits in tactic order.

    Order matters: callers treat `extract_used_premises(...)[0]` as the
    primary premise.
    """
    if not tactic:
        return []
    seen: set[str] = set()
    out: list[str] = []

    # 1. Bracket lists first — these are the surest signal of intent.
    for name in extract_rw_simp_lemmas(tactic):
        if name not in seen:
            seen.add(name)
            out.append(name)

    # 2/3. Dotted scan (RHS first when `:=` is present), with local-
    # projection filter applied to dotted hits too.
    if ":=" in tactic:
        head, _, tail = tactic.partition(":=")
        scan_order = (tail, head)
    else:
        scan_order = (tactic,)

    for chunk in scan_order:
        for m in _DOTTED_RE.finditer(chunk):
            name = m.group(0)
            for suffix in (".mp", ".mpr", ".symm", ".1", ".2"):
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            if not name or name in seen:
                continue
            if _is_local_projection(name):
                continue
            seen.add(name)
            out.append(name)
        # 4. Bare allow-list hits.
        for m in _BARE_RE.finditer(chunk):
            name = m.group(1)
            if name in _BARE_ALLOWLIST and name not in seen:
                seen.add(name)
                out.append(name)
    return out


# ---------------------------------------------------------------------------
# Namespace canonicalisation (PART B fix)
# ---------------------------------------------------------------------------

# Hand-curated map of bare lemma names → their Nat-qualified Mathlib
# form. Used by the Mathlib miner when the file path / declaration
# context indicates the active namespace is `Nat`. Conservative: only
# the names that we know are unambiguous wins. New entries should be
# checked against Mathlib first.
_NAT_BARE_TO_QUALIFIED: dict[str, str] = {
    "gcd_mul_lcm":      "Nat.gcd_mul_lcm",
    "gcd_eq_zero_iff":  "Nat.gcd_eq_zero_iff",
    "lcm_ne_zero":      "Nat.lcm_ne_zero",
    "gcd_comm":         "Nat.gcd_comm",
    "lcm_comm":         "Nat.lcm_comm",
    "gcd_assoc":        "Nat.gcd_assoc",
    "lcm_assoc":        "Nat.lcm_assoc",
}


def canonicalise_for_namespace(name: str, namespace: str | None) -> str:
    """Return `name` rewritten to its Nat-qualified form when applicable.

    - Only fires when `namespace == "Nat"` (case-sensitive).
    - Only rewrites bare names (no `.`).
    - Local projections (`h.gcd_eq_one`, `a.gcd`) are left alone — the
      dotted form prevents both branches from firing because the bare
      lookup misses.
    - Already-qualified names are left alone.
    """
    if not name or namespace != "Nat":
        return name
    if "." in name:
        return name
    return _NAT_BARE_TO_QUALIFIED.get(name, name)


# Pre-compiled word-bounded rewrite patterns so the miner can rewrite
# every occurrence of a canonicalisable bare name inside a tactic text
# in one regex pass per name. Word boundaries:
#   - lookbehind: not preceded by [A-Za-z0-9_.] so `foo.gcd_mul_lcm`
#     and `my_gcd_mul_lcm` are left alone.
#   - lookahead:  not followed by [A-Za-z0-9_]  so we don't bite into a
#     longer identifier (`gcd_mul_lcm_foo`).
_NAT_REWRITE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"(?<![A-Za-z0-9_.])"
                rf"{re.escape(bare)}"
                rf"(?![A-Za-z0-9_])"), qualified)
    for bare, qualified in _NAT_BARE_TO_QUALIFIED.items()
]


def canonicalise_tactic_text(tactic: str, namespace: str | None) -> str:
    """Rewrite bare Nat lemma names inside `tactic` to their qualified
    form when the active namespace is Nat. Used by the Mathlib miner to
    avoid emitting tactic templates like `rw [gcd_mul_lcm]` that won't
    resolve when replayed outside Mathlib/Data/Nat (PART E fix-up).

    Word-bounded so local projections (`h.gcd_eq_one`) and longer
    identifiers (`my_gcd_mul_lcm`) are untouched. Already-qualified
    citations (`Nat.gcd_mul_lcm`) are untouched.
    """
    if not tactic or namespace != "Nat":
        return tactic
    out = tactic
    for pat, qualified in _NAT_REWRITE_PATTERNS:
        out = pat.sub(qualified, out)
    return out
