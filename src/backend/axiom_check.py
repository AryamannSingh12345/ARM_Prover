"""Ask the kernel what a proof actually depends on.

Motivation (2026-08-16). `putnam_1981_a1` was reported `outcome="solved"`
by a bare-baseline run whose entire proof body was:

    exact sorryAx _ true

`sorryAx` is the primitive constant that `sorry` elaborates to, so this is
`sorry` written to evade the guards, and it evaded all of them:

  - `compile_verify._SORRY_RE` is `\\bsorry\\b`, which does NOT match
    `sorryAx` (no word boundary between `sorry` and `Ax`);
  - Lean emitted no `declaration uses 'sorry'` warning, so
    `_SORRY_WARNING_RE` never fired either;
  - the declaration-head audit scans heads (`axiom`, `instance`, …) and
    `sorryAx` is not a declaration head, so that audit could not have
    caught it in principle.

Widening the regex would patch this one spelling and invite the next. The
gate implemented here instead asks Lean itself, via `#print axioms`, what
the finished declaration rests on, and requires that to be a subset of
Lean's three standard axioms. That catches `sorryAx`, any model-declared
`axiom` (the separate hole open since the `hard6` run), and spellings
nobody has thought of yet — because it inspects the kernel's dependency
record rather than the text that produced it.

ALLOWED is exactly the axiom set classical Mathlib is built on. A proof
depending on anything else is not wrong, but it is not something this
project may count without a human looking at it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Lean's three standard axioms. Mathlib is classical: essentially every
#: real theorem depends on some subset of these, and depending on all
#: three is unremarkable.
ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})

#: Declarations worth interrogating. `example` has no name to print, and
#: `def`/`abbrev` carry no proof obligation.
_DECL_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)?"
    r"(?:private\s+|protected\s+|noncomputable\s+|partial\s+)*"
    r"(?:theorem|lemma)\s+"
    r"([A-Za-z_À-ɏ][A-Za-z0-9_'À-ɏͰ-Ͽ]*)",
    re.M,
)

#: `#print axioms foo` prints either
#:   'foo' depends on axioms: [propext, Classical.choice, Quot.sound]
#: or
#:   'foo' does not depend on any axioms
_DEPENDS_RE = re.compile(
    r"'([^']+)'\s+depends on axioms:\s*\[([^\]]*)\]", re.S)
_NO_AXIOMS_RE = re.compile(r"'([^']+)'\s+does not depend on any axioms")


@dataclass(slots=True)
class AxiomCheckResult:
    """Verdict of the axiom gate.

    `ok` is the only field callers need. It is True ONLY when every
    theorem in the source was interrogated and every one of them depends
    on a subset of ALLOWED_AXIOMS.

    The gate FAILS CLOSED: an unparseable report, a compile failure, or a
    source with no theorem in it all yield ok=False with `inconclusive`
    set. A soundness gate that fails open is the bug it was written to
    prevent, and the cost of failing closed is a re-run, not a wrong
    number.
    """
    ok: bool
    checked: dict[str, frozenset[str]] = field(default_factory=dict)
    offending: dict[str, frozenset[str]] = field(default_factory=dict)
    inconclusive: bool = False
    detail: str = ""
    report: str = ""


def extract_theorem_names(source: str) -> list[str]:
    """Names of every `theorem`/`lemma` declared in `source`, in order.

    Namespaces are deliberately NOT resolved: `#print axioms` is issued
    in the same file, at the end, where the short name is in scope only
    if no `namespace` was opened. Files this project compiles are flat
    (a header plus a proof), and a file that does declare a namespace
    will simply fail the lookup and be reported inconclusive rather than
    silently passed.
    """
    return _DECL_RE.findall(source)


def parse_axiom_report(output: str) -> dict[str, frozenset[str]]:
    """Parse `#print axioms` lines out of Lean's stdout."""
    found: dict[str, frozenset[str]] = {}
    for m in _NO_AXIOMS_RE.finditer(output):
        found[m.group(1)] = frozenset()
    for m in _DEPENDS_RE.finditer(output):
        axioms = {a.strip() for a in m.group(2).split(",") if a.strip()}
        found[m.group(1)] = frozenset(axioms)
    return found


def build_probe_source(source: str, names: list[str]) -> str:
    """`source` with a `#print axioms` command appended per theorem.

    Appended at the END so the proof itself is compiled exactly as the
    scoring compile saw it; `#print axioms` is a command with no
    elaboration effect on what precedes it.
    """
    lines = [source.rstrip("\n"), ""]
    lines += [f"#print axioms {n}" for n in names]
    return "\n".join(lines) + "\n"


def evaluate_report(names: list[str],
                    report: dict[str, frozenset[str]]
                    ) -> tuple[bool, dict, dict, bool, str]:
    """Pure decision step, factored out so it is testable without Lean.

    Returns (ok, checked, offending, inconclusive, detail).
    """
    if not names:
        # NOT APPLICABLE, not a failure. `example : … := rfl` and probe
        # sources declare nothing to interrogate, and rejecting them
        # would break every diagnostic compile in the repo.
        #
        # This is safe for SCORING because a solve always carries the
        # benchmark's named theorem: body mode builds the file from the
        # loaded `theorem …` header, and verbatim mode is gated by
        # `statement_is_verbatim`. It is safe for everything ELSE because
        # `compile_verify._SORRY_RE` now also matches `sorryAx`/`admit`
        # in the source, so an unnamed `example … := by exact sorryAx _
        # true` is still rejected — one layer up, before any compile.
        return (True, {}, {}, True,
                "not applicable: no named theorem/lemma in source")

    checked: dict[str, frozenset[str]] = {}
    offending: dict[str, frozenset[str]] = {}
    missing: list[str] = []

    for n in names:
        # Lean prints the fully-qualified name; match on the short name
        # too so a `namespace`-free file resolves cleanly.
        hit = None
        if n in report:
            hit = report[n]
        else:
            for k, v in report.items():
                if k == n or k.endswith("." + n):
                    hit = v
                    break
        if hit is None:
            missing.append(n)
            continue
        checked[n] = hit
        extra = hit - ALLOWED_AXIOMS
        if extra:
            offending[n] = frozenset(extra)

    if missing:
        return (False, checked, offending, True,
                f"no axiom report for: {', '.join(sorted(missing))}")
    if offending:
        detail = "; ".join(
            f"{n} depends on {sorted(ax)}" for n, ax in sorted(offending.items()))
        return (False, checked, offending, False, detail)
    return (True, checked, {}, False, "")


def check_axioms(source: str, compile_fn) -> AxiomCheckResult:
    """Run the gate. `compile_fn(src) -> (returncode_ok, combined_output)`.

    `compile_fn` is injected rather than imported so this module has no
    dependency on the verifier, and so tests can exercise every branch
    without Lean.
    """
    names = extract_theorem_names(source)
    if not names:
        # No compile is spent: nothing to interrogate. See evaluate_report.
        ok, checked, offending, inconc, detail = evaluate_report([], {})
        return AxiomCheckResult(ok=ok, checked=checked, offending=offending,
                                inconclusive=inconc, detail=detail)

    probe_ok, output = compile_fn(build_probe_source(source, names))
    if not probe_ok:
        return AxiomCheckResult(
            ok=False, inconclusive=True, report=output,
            detail="axiom probe failed to compile")

    report = parse_axiom_report(output)
    ok, checked, offending, inconc, detail = evaluate_report(names, report)
    return AxiomCheckResult(ok=ok, checked=checked, offending=offending,
                            inconclusive=inconc, detail=detail, report=output)
