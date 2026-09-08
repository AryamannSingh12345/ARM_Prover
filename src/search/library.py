"""Build a self-contained Lean library, kernel-gated, run-scoped, disposable.

The prover can already invent and prove general declarations — this
session alone produced `aux_powerSum_recurrence {R} [CommRing R]`,
`aux_nat_recurrence_unique`, `aux_choose_prefix_pascal`, all
kernel-verified and none problem-specific. What it could not do is BUILD A
LIBRARY: ARM invents lemmas only in service of a stuck leaf, so "what does
this domain need?" is never asked, and anything proved inside a theory
that later abandons evaporates.

That gap is why `order5_lrs_a0_unit` was unreachable: Mathlib has 21
`LinearRecurrence` declarations and no power-sum representation, so no
amount of search on the target could work. The missing prerequisite had to
be built first.

This module builds it. Given an informal SPEC of a concept, it plans an
ordered list of declarations, writes each one with every previously
VERIFIED declaration in scope, and admits it only if Lean accepts it.

Sandbox contract — the reason this is safe to run
-------------------------------------------------
1. NOTHING is written outside the caller's output directory. Deleting that
   one directory is complete cleanup.
2. The persistent cross-run bank (`results/invented_lemmas.lean`) is NEVER
   touched. A library build cannot pollute it, so a failed or junk build
   costs nothing but disk.
3. Mathlib and `lean/` are never modified. Verification goes through the
   ordinary `verify_fn`, which compiles a temp file like every other
   check.
4. Declarations are admitted ONLY on a real compile with no
   `sorry`/`admit`/`native_decide`. The library is as trustworthy as the
   kernel and no more.
5. Everything is in memory until `emit()` is called, so an abandoned build
   leaves no trace at all.

This module owns no Lean and no LLM access: it takes the same callables
the rest of the pipeline uses.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

_FORBIDDEN_TAC_RE = re.compile(r"\b(?:sorry|admit|native_decide)\b")
_DECL_HEAD_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*"
    r"(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(?P<kind>theorem|lemma|def|abbrev|structure|class|instance|example)\b"
    r"\s*(?P<name>[A-Za-z_][A-Za-z0-9_'!?₀-₉.]*)?")

#: Declaration kinds a library may contain.
DECL_KINDS = ("def", "abbrev", "structure", "class", "instance",
              "lemma", "theorem")


PLAN_SYSTEM = """\
You are a Lean 4 + Mathlib expert planning a small, self-contained
LIBRARY: the ordered list of declarations that a concept needs before
anything interesting can be stated or proved about it.

You are NOT proving a theorem. You are designing an API.

RULES
- Order matters: every declaration may use only the ones BEFORE it.
- Start with the objects (`def`, `abbrev`, `structure`, `class`), then the
  basic facts, then the results that depend on them.
- Prefer many small declarations over few large ones. Each is proved
  separately and a large one that fails costs the whole build.
- State each declaration in FULL Lean syntax, including binders and
  typeclass assumptions, but WITHOUT its proof.
- Generalise: state facts for an arbitrary `CommRing R` / `Type*` rather
  than a specific instance, unless the concept genuinely needs one.
- Do NOT restate something Mathlib already has. If a step is already in
  Mathlib, skip it and note the name in `rationale`.

OUTPUT — reply with exactly this JSON and NOTHING else:

{
  "decls": [
    {"name": "<lean identifier>",
     "kind": "def|abbrev|structure|class|instance|lemma|theorem",
     "statement": "<full Lean declaration WITHOUT `:= proof`>",
     "rationale": "<one line: why this is needed>"}
  ]
}\
"""


BUILD_SYSTEM = """\
You are a Lean 4 + Mathlib expert completing ONE declaration of a library.

You are given the declarations already VERIFIED and in scope, and the next
declaration to write. Produce that declaration COMPLETE and compilable.

RULES
- Reproduce the given statement exactly, then supply its body: `:= …` for
  a definition, `:= by …` for a proof.
- You may use anything from Mathlib and any declaration already in scope.
- NEVER use `sorry`, `admit` or `native_decide`. The declaration is
  compiled and will be rejected.
- If the statement as given cannot be proved, say so in `problem` and
  return your best attempt anyway — a rejected attempt with its Lean error
  is more useful than nothing.

OUTPUT — reply with exactly this JSON and NOTHING else:

{"declaration": "<the complete Lean declaration>", "problem": "<empty, or what is wrong with the statement>"}\
"""


def plan_prompt(spec: str, imports: str, max_decls: int,
                known: list[str] | None = None) -> str:
    out = [f"Design a Lean 4 library for this concept:\n\n{spec}\n",
           f"Available imports:\n{imports}\n",
           f"At most {max_decls} declarations."]
    if known:
        out.append("\nAlready in scope (do not restate):\n"
                   + "\n".join(f"- {k}" for k in known[:40]))
    return "\n".join(out)


def build_prompt(decl_statement: str, in_scope: list[str],
                 prev_error: str | None = None,
                 prev_attempt: str | None = None) -> str:
    out = ["Declaration to complete:\n", "```lean", decl_statement, "```"]
    if in_scope:
        out += ["", "Already verified and in scope:", "```lean",
                "\n\n".join(in_scope[-12:]), "```"]
    if prev_error:
        out += ["", "Your previous attempt was REJECTED by Lean:", "```",
                prev_error[:1200], "```"]
        if prev_attempt:
            out += ["", "It was:", "```lean", prev_attempt[:2000], "```",
                    "", "Fix it with the smallest change that works."]
    return "\n".join(out)


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


def decl_head(source: str) -> tuple[str | None, str | None]:
    """(kind, name) of a Lean declaration, or (None, None)."""
    m = _DECL_HEAD_RE.match(source or "")
    if not m:
        return None, None
    return m.group("kind"), m.group("name")


def parse_plan(raw: str, max_decls: int) -> tuple[list[dict] | None, str | None]:
    obj = _extract_json_object(raw or "")
    if obj is None:
        return None, "no JSON object in response"
    raw_decls = obj.get("decls")
    if not isinstance(raw_decls, list) or not raw_decls:
        return None, "no decls"
    out: list[dict] = []
    seen: set[str] = set()
    for d in raw_decls[:max_decls]:
        if not isinstance(d, dict):
            continue
        stmt = str(d.get("statement") or "").strip()
        if not stmt:
            continue
        kind, name = decl_head(stmt)
        name = str(d.get("name") or name or "").strip()
        if not name or name in seen:
            continue
        if kind is None or kind not in DECL_KINDS:
            continue
        seen.add(name)
        out.append({"name": name, "kind": kind, "statement": stmt,
                    "rationale": str(d.get("rationale") or "")})
    if not out:
        return None, "no usable decls"
    return out, None


@dataclass(slots=True)
class LibraryDecl:
    name: str
    kind: str
    source: str
    attempts: int = 0
    rationale: str = ""
    error: str | None = None          # last Lean error, when unverified

    @property
    def verified(self) -> bool:
        return self.error is None


@dataclass
class LibraryResult:
    """A built library. In memory until `emit()`."""
    spec: str
    imports: str
    name: str = "Library"
    decls: list[LibraryDecl] = field(default_factory=list)   # verified
    failed: list[LibraryDecl] = field(default_factory=list)
    planned: int = 0
    elapsed_s: float = 0.0

    def to_lean(self) -> str:
        """The library as one compilable file."""
        head = [self.imports.rstrip(), "",
                "/-!", f"# {self.name}", "",
                *(f"{ln}" for ln in self.spec.strip().splitlines()),
                "",
                "Machine-generated. Every declaration below was accepted by",
                "the Lean kernel with no `sorry`, `admit` or",
                "`native_decide`. Declarations the build could NOT verify",
                "are listed in the manifest, not here.",
                "-/", ""]
        return "\n".join(head) + "\n\n".join(d.source for d in self.decls) + "\n"

    def to_manifest(self) -> dict:
        return {
            "name": self.name,
            "spec": self.spec,
            "imports": self.imports,
            "planned": self.planned,
            "verified": len(self.decls),
            "failed": len(self.failed),
            "elapsed_s": round(self.elapsed_s, 1),
            "decls": [asdict(d) for d in self.decls],
            "failed_decls": [asdict(d) for d in self.failed],
        }

    def emit(self, out_dir: str | Path) -> list[Path]:
        """Write the library and its manifest under `out_dir`.

        This is the ONLY method that touches the filesystem, and it writes
        nowhere else — so `rm -rf out_dir` is complete cleanup and the
        persistent cross-run bank is untouched by construction.
        """
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        lean = d / f"{self.name}.lean"
        man = d / "manifest.json"
        lean.write_text(self.to_lean(), encoding="utf-8")
        man.write_text(json.dumps(self.to_manifest(), indent=2,
                                  ensure_ascii=False), encoding="utf-8")
        return [lean, man]


def build_library(
    spec: str,
    *,
    llm_call: Callable[[str, str], str],
    verify_fn: Callable[[str, str], dict],
    imports: str = "import Mathlib",
    name: str = "Library",
    max_decls: int = 10,
    attempts_per_decl: int = 3,
    known: list[str] | None = None,
    trace: Callable[..., None] | None = None,
) -> LibraryResult:
    """Plan a library, then build it declaration by declaration.

    `verify_fn(header, body)` is the same contract the rest of the
    pipeline uses. A declaration is compiled as a COMPLETE unit — the
    header carries the imports plus every previously verified declaration,
    and the body is empty — so `def`s, `structure`s and lemmas all go
    through one gate.

    A declaration that will not verify is recorded in `failed` and the
    build CONTINUES: a library is inherently partial, and 8 of 12 useful
    declarations beats nothing. Later declarations simply do not get the
    failed one in scope.
    """
    tr = trace or (lambda kind, **kw: None)
    t0 = time.time()
    res = LibraryResult(spec=spec, imports=imports, name=name)

    try:
        raw = llm_call(PLAN_SYSTEM,
                       plan_prompt(spec, imports, max_decls, known))
    except Exception as e:
        tr("library", stage="plan_failed", detail=f"{type(e).__name__}: {e}")
        res.elapsed_s = time.time() - t0
        return res

    plan, perr = parse_plan(raw, max_decls)
    if plan is None:
        tr("library", stage="plan_unparseable", detail=perr)
        res.elapsed_s = time.time() - t0
        return res
    res.planned = len(plan)
    tr("library", stage="planned", n=len(plan),
       names=[d["name"] for d in plan])

    for item in plan:
        prev_err: str | None = None
        prev_src: str | None = None
        rec = LibraryDecl(name=item["name"], kind=item["kind"],
                          source=item["statement"],
                          rationale=item["rationale"],
                          error="not attempted")
        for attempt in range(1, max(int(attempts_per_decl), 1) + 1):
            rec.attempts = attempt
            try:
                raw_d = llm_call(
                    BUILD_SYSTEM,
                    build_prompt(item["statement"],
                                 [d.source for d in res.decls],
                                 prev_err, prev_src))
            except Exception as e:
                rec.error = f"llm error: {type(e).__name__}"
                break
            obj = _extract_json_object(raw_d)
            src = str((obj or {}).get("declaration") or "").strip()
            if not src:
                prev_err = "your reply contained no `declaration`"
                rec.error = prev_err
                continue
            if _FORBIDDEN_TAC_RE.search(src):
                prev_err = ("the declaration used sorry/admit/"
                            "native_decide, which is never accepted")
                prev_src = src
                rec.error = prev_err
                continue
            # Environment integrity: every verified declaration is carried
            # into the header of every LATER one, so one `instance` or
            # `axiom` here poisons the whole library and anything built on
            # it. See src/search/decl_safety.py.
            from search.decl_safety import unsafe_declaration
            _why = unsafe_declaration(src)
            if _why:
                prev_err = _why
                prev_src = src
                rec.error = prev_err
                continue

            header = "\n\n".join(
                [imports.rstrip()]
                + [d.source for d in res.decls]
                + [src])
            try:
                gate = verify_fn(header, "")
            except Exception as e:
                rec.error = f"verifier error: {type(e).__name__}"
                break
            if gate.get("ok"):
                rec.source = src
                rec.error = None
                kind, nm = decl_head(src)
                if kind:
                    rec.kind = kind
                if nm:
                    rec.name = nm
                res.decls.append(rec)
                # NB `decl_kind`, not `kind`: the trace callable's first
                # positional parameter is itself named `kind`.
                tr("library", stage="VERIFIED", name=rec.name,
                   decl_kind=rec.kind, attempt=attempt)
                break
            prev_err = (gate.get("errors") or "?")[:1200]
            prev_src = src
            rec.source = src
            rec.error = prev_err
            tr("library", stage="rejected", name=rec.name, attempt=attempt,
               detail=prev_err[:200])
        if rec.error is not None:
            res.failed.append(rec)
            tr("library", stage="gave_up", name=rec.name,
               attempts=rec.attempts)

    res.elapsed_s = time.time() - t0
    tr("library", stage="done", verified=len(res.decls),
       failed=len(res.failed), elapsed_s=round(res.elapsed_s, 1))
    return res
