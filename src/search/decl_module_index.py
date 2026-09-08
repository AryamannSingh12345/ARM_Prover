"""Declaration → Mathlib-module resolution from the local pin.

Environment-derived, not hand-curated: the premise graph
(`data/mathlib_graph/nodes.jsonl`, ~211K declarations extracted from
the pinned Mathlib) maps every declaration name to its source file, and
the Lean module name is just that path with separators as dots. The
import-refresh hook uses this to turn ``Unknown identifier `X` ``
directly into the import that provides `X` — deterministic, correct by
construction, no module names in code, and no LLM guessing (b5_bare_v4:
asked which module provides a betweenness lemma, the model answered
`Mathlib.Data.Fin.Basic`).

Hallucinated names (the other common case: `dist_add_dist_of_mem_segment`
does not exist in the pin) resolve to None — correctly adding NO import
— and the prove step's error feedback remains the fix for those.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_NODES = _ROOT / "data" / "mathlib_graph" / "nodes.jsonl"

# Lean's unknown-name diagnostics, with the name captured:
# `Unknown identifier `foo``, `Unknown constant `Foo.bar``.
UNKNOWN_NAME_RE = re.compile(
    r"[Uu]nknown (?:identifier|constant)[^`\n]*`([^`\s]+)`")

_INDEX: dict[str, str] | None = None
_BY_LAST: dict[str, set[str]] | None = None


def _module_of(file_path: str) -> str:
    mod = file_path.replace("\\", ".").replace("/", ".")
    return mod[:-5] if mod.endswith(".lean") else mod


def _load() -> bool:
    global _INDEX, _BY_LAST
    if _INDEX is not None:
        return True
    if not _NODES.exists():
        return False
    idx: dict[str, str] = {}
    by_last: dict[str, set[str]] = {}
    with _NODES.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            name, fp = r.get("name"), r.get("file_path")
            if not name or not fp:
                continue
            mod = _module_of(fp)
            idx.setdefault(name, mod)
            by_last.setdefault(name.rsplit(".", 1)[-1], set()).add(mod)
    _INDEX = idx
    _BY_LAST = by_last
    return True


def resolve_decl_module(name: str) -> str | None:
    """Module that declares `name`, or None.

    Tries the exact name; then progressively strips trailing dotted
    components (errors report accessor forms like
    `mem_segment_iff_wbtw.mp`); finally accepts a last-component match
    only when it is unambiguous (a single module declares it)."""
    if not name or not _load():
        return None
    assert _INDEX is not None and _BY_LAST is not None
    probe = name
    for _ in range(3):
        if probe in _INDEX:
            return _INDEX[probe]
        if "." not in probe:
            break
        probe = probe.rsplit(".", 1)[0]
    mods = _BY_LAST.get(name.rsplit(".", 1)[-1]) or set()
    if len(mods) == 1:
        return next(iter(mods))
    return None


def unknown_names(text: str) -> list[str]:
    """Deduped declaration names cited by unknown-name diagnostics in
    `text`, in order of first appearance."""
    return list(dict.fromkeys(UNKNOWN_NAME_RE.findall(text or "")))


def modules_for_premises(names: list[str] | None,
                         limit: int = 12) -> list[str]:
    """Modules providing the RETRIEVED PREMISES, deterministically.

    Import selection and premise retrieval were doing related work in
    isolation: imports were chosen from the STATEMENT alone (LLM guess,
    compile-gated) and premise retrieval ran AFTERWARDS, so the names
    BM25 had just surfaced never informed the import set. The refresh
    hook then had to rediscover them REACTIVELY, one failed compile at a
    time.

    MEASURED on putnam_1981_a1 (`p1981a1_dag_v1`): the LLM guess was
    `Polynomial.Basic` + `Finset.Interval` + `Deriv.Basic` for a 5-adic
    valuation problem, `Nat.Prime` was never imported at all, and
    `Nat.Factorization.Defs` arrived at t=13522s of a 14172s run — after
    the round that needed it had already failed. Meanwhile BM25 had
    retrieved the relevant number-theory names at t=388s.

    This closes the loop with the index that already exists: resolve
    each retrieved name to its module and hand the caller the set. No
    compiles, no LLM calls, no module names in code — the same
    environment-derived category as `apply_pin_renames`. A name the
    graph cannot resolve (hallucinated, or ambiguous by last component)
    yields nothing rather than a guess.

    `limit` caps the result because each import costs elaboration time;
    modules are returned in retrieval order, so the highest-ranked
    premises win. Returns [] when the graph is unavailable, which leaves
    the caller's existing behaviour untouched.
    """
    if not names:
        return []
    out: list[str] = []
    for n in names:
        mod = resolve_decl_module(n)
        if mod and mod not in out:
            out.append(mod)
            if len(out) >= limit:
                break
    return out
