"""Build a Mathlib declaration-use graph from source via regex extraction.

Two-pass:
  1) Walk every .lean file in the Mathlib source tree. Track namespace stack.
     Find every `theorem|lemma|def|abbrev|instance|structure|inductive|class|
     opaque|axiom NAME`. Emit one node per fully-qualified name.
  2) Second walk: for each declaration body, regex-extract every dotted identifier
     and intersect with the global node set. Emit an edge (from -> to) for each.

Outputs:
  data/mathlib_graph/nodes.jsonl   one JSON object per line {name,kind,file_path,signature?}
  data/mathlib_graph/edges.jsonl   one JSON object per line {from,to}

The regex extractor misses constants introduced by macros / `simp` lemma sets /
tactic shorthands. For a *scoring* signal (not proving), it's good enough.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---- Decl recognizer --------------------------------------------------------

DECL_KINDS = (
    "theorem", "lemma", "def", "abbrev", "instance",
    "structure", "inductive", "class", "opaque", "axiom",
)

# Optional modifiers before the kind keyword: @[...], protected, private,
# noncomputable, public, unsafe, partial.
DECL_RE = re.compile(
    r"(?m)^(?:\s*@\[[^\]]*\]\s*\n)*"
    r"(?:\s*(?:public|protected|private|noncomputable|unsafe|partial)\s+)*"
    rf"\s*(?P<kind>{'|'.join(DECL_KINDS)})\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_.']*)",
)

# Namespace open/close (single-line forms only; multi-line `namespace Foo.Bar`
# is fine because we match the entire line).
NS_OPEN_RE = re.compile(r"^\s*namespace\s+([A-Za-z_][A-Za-z0-9_.']*)\s*$", re.M)
NS_CLOSE_RE = re.compile(r"^\s*end(?:\s+([A-Za-z_][A-Za-z0-9_.']*))?\s*$", re.M)

# Identifier-like token used in the body to find references.
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


# ---- Comment + string stripping --------------------------------------------

LINE_COMMENT_RE = re.compile(r"--[^\n]*")
STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')


def strip_comments_and_strings(src: str) -> str:
    """Remove block comments (nested supported), line comments, strings.
    Replacement preserves newlines so line numbers stay roughly aligned."""
    out: list[str] = []
    i = 0
    n = len(src)
    depth = 0
    while i < n:
        if depth == 0 and src.startswith("/-", i):
            depth = 1
            i += 2
        elif depth > 0 and src.startswith("/-", i):
            depth += 1
            i += 2
        elif depth > 0 and src.startswith("-/", i):
            depth -= 1
            i += 2
        elif depth > 0:
            if src[i] == "\n":
                out.append("\n")
            i += 1
        else:
            out.append(src[i])
            i += 1
    cleaned = "".join(out)
    cleaned = LINE_COMMENT_RE.sub("", cleaned)
    cleaned = STRING_RE.sub('""', cleaned)
    return cleaned


# Names that look like decls in the regex but are noise / Lean keywords /
# conventional 1-2 char variable bindings the user reuses everywhere.
_NOISE_NAMES = {
    "lemma", "theorem", "def", "abbrev", "instance", "structure", "inductive",
    "class", "opaque", "axiom", "section", "namespace", "end", "where", "with",
    "do", "match", "fun", "let", "have", "show", "by", "from", "in", "open",
    "import", "module", "public", "protected", "private", "noncomputable",
    "unsafe", "partial",
}


def _is_noise(name: str) -> bool:
    if name in _NOISE_NAMES:
        return True
    # Single-letter unqualified -> almost certainly a parameter / mvar
    if "." not in name and len(name) <= 1:
        return True
    return False


# ---- Namespace tracking -----------------------------------------------------

def resolve_namespaces(src: str) -> list[tuple[int, list[str]]]:
    """Return [(char_offset, namespace_stack_at_offset)]. The list is sorted
    by offset; for any position p, find the last entry with offset <= p."""
    events: list[tuple[int, str, str]] = []
    for m in NS_OPEN_RE.finditer(src):
        events.append((m.start(), "open", m.group(1)))
    for m in NS_CLOSE_RE.finditer(src):
        events.append((m.start(), "close", m.group(1) or ""))
    events.sort(key=lambda e: e[0])
    stack: list[str] = []
    out: list[tuple[int, list[str]]] = [(0, list(stack))]
    for off, kind, name in events:
        if kind == "open":
            stack.append(name)
        else:
            # `end Foo`: pop while top doesn't match (be lenient; not all `end`s
            # close namespaces - some close sections).
            if name and stack and stack[-1] == name:
                stack.pop()
            elif not name and stack:
                stack.pop()
            # else: leave the stack alone (probably `end` of a section).
        out.append((off, list(stack)))
    return out


def ns_at(offset: int, table: list[tuple[int, list[str]]]) -> list[str]:
    lo, hi = 0, len(table) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if table[mid][0] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return table[lo][1]


def qualify(ns: list[str], name: str) -> str:
    if "." in name:
        # Authors sometimes write `theorem Foo.bar ...` even inside `namespace Foo`.
        # If the leading segment matches the namespace, treat as already qualified.
        return name
    return ".".join(ns + [name]) if ns else name


# ---- Phase 1.1 — collect nodes ---------------------------------------------

def scan_nodes(mathlib_root: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Returns (nodes_by_name, source_by_name)."""
    nodes: dict[str, dict] = {}
    src_text: dict[str, str] = {}  # name -> cleaned source (for phase-2 ref scan)
    files = sorted(mathlib_root.rglob("*.lean"))
    print(f"[phase 1.1] scanning {len(files)} files for decls", flush=True)
    t0 = time.time()
    for i, fp in enumerate(files):
        try:
            raw = fp.read_text(encoding="utf-8")
        except OSError:
            continue
        clean = strip_comments_and_strings(raw)
        ns_table = resolve_namespaces(clean)

        # Collect decl positions to slice bodies (until the next decl / EOF).
        decl_hits: list[tuple[int, str, str]] = []  # (start, kind, qualified_name)
        for m in DECL_RE.finditer(clean):
            kind = m.group("kind")
            name = m.group("name")
            qname = qualify(ns_at(m.start(), ns_table), name)
            decl_hits.append((m.start(), kind, qname))

        decl_hits.sort(key=lambda x: x[0])
        for idx, (start, kind, qname) in enumerate(decl_hits):
            if _is_noise(qname):
                continue
            end = decl_hits[idx + 1][0] if idx + 1 < len(decl_hits) else len(clean)
            body = clean[start:end]
            if qname not in nodes:
                nodes[qname] = {
                    "name": qname, "kind": kind,
                    "file_path": str(fp.relative_to(mathlib_root.parent)),
                }
                src_text[qname] = body
            else:
                # Duplicate (e.g. extending an instance); keep first.
                src_text[qname] = src_text[qname] + "\n" + body
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)}  nodes={len(nodes)}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"[phase 1.1] done. {len(nodes)} nodes in {time.time()-t0:.1f}s", flush=True)
    return nodes, src_text


# ---- Phase 1.2 — collect edges ---------------------------------------------

def scan_edges(nodes: dict[str, dict], src_text: dict[str, str]) -> list[tuple[str, str]]:
    """Edges: only exact-qualified matches. Tail-fallback (matching bare
    identifier `bar` to `Foo.bar`) is disabled because it generates massive
    false positives from local variables sharing names with global decls.
    Precision-over-recall is the right call here — this is a structural prior
    for a scoring head, not a proof oracle."""
    node_set = set(nodes.keys())
    edges: list[tuple[str, str]] = []
    print(f"[phase 1.2] scanning bodies for qualified refs across {len(src_text)} decls",
          flush=True)
    t0 = time.time()
    for i, (from_name, body) in enumerate(src_text.items()):
        seen: set[str] = set()
        for m in IDENT_RE.finditer(body):
            tok = m.group(0)
            if "." not in tok:
                continue   # bare names too noisy; skip
            if tok == from_name or tok in seen:
                continue
            if tok in node_set:
                seen.add(tok)
                edges.append((from_name, tok))
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(src_text)}  edges={len(edges)}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"[phase 1.2] done. {len(edges)} edges in {time.time()-t0:.1f}s", flush=True)
    return edges


# ---- Main ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mathlib-root", default=str(
        Path(__file__).resolve().parents[1] / "lean" / ".lake" / "packages"
        / "mathlib" / "Mathlib"
    ))
    ap.add_argument("--out-dir", default=str(
        Path(__file__).resolve().parents[1] / "data" / "mathlib_graph"
    ))
    args = ap.parse_args()
    root = Path(args.mathlib_root)
    if not root.exists():
        print(f"ERROR: mathlib root not found: {root}", file=sys.stderr)
        return 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    nodes, src_text = scan_nodes(root)
    edges = scan_edges(nodes, src_text)

    nodes_path = out / "nodes.jsonl"
    edges_path = out / "edges.jsonl"
    with nodes_path.open("w", encoding="utf-8") as f:
        for v in nodes.values():
            f.write(json.dumps(v) + "\n")
    with edges_path.open("w", encoding="utf-8") as f:
        for a, b in edges:
            f.write(json.dumps({"from": a, "to": b}) + "\n")
    print(f"\nwrote {nodes_path} ({len(nodes)} nodes)")
    print(f"wrote {edges_path} ({len(edges)} edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
