"""LLM-driven import resolution — no hardcoded Mathlib knowledge.

The rule table in `import_inference.py` only knows the miniF2F-shaped
world; statement-heavy benchmarks (ProofNet, PutnamBench) open namespaces
the rules have never heard of. Here the policy model itself is asked which
Mathlib modules a STATEMENT needs, and `resolve_imports_llm` closes the
loop: compile the bare statement as a gate, feed any header-level error
back to the model, retry. Nothing in this module names a Mathlib module —
the only fixed strings are syntax (the `import` keyword) and the
whole-library fallback `import Mathlib`, which is Lean's own spelling of
"everything".

Escalation ladder:  llm  ->  llm + error feedback (xN)  ->  import Mathlib
-> "unresolved" (statement genuinely broken against this pin).
"""
from __future__ import annotations

import re
from typing import Callable

_IMPORT_LINE_RE = re.compile(r"^import\s+[A-Za-z][\w.]*$")

LLM_IMPORT_SYSTEM = (
    "You are a Lean 4 / Mathlib 4 expert. You will be shown a theorem "
    "statement, possibly preceded by `open` declarations. Reply with ONLY "
    "the Lean import lines needed for the STATEMENT ITSELF to elaborate "
    "(every opened namespace, every type and identifier mentioned). "
    "Prefer specific `import Mathlib.<Module>` lines over `import Mathlib`. "
    "One import per line. No prose, no code fences, no comments."
)


def llm_import_prompt(statement_text: str, prev_imports: str | None = None,
                      lean_error: str | None = None) -> tuple[str, str]:
    """Build (system, user) for an import-suggestion call. When a previous
    attempt failed to elaborate, the exact Lean error is included so the
    model can correct its own import list."""
    user = f"Statement:\n```lean\n{statement_text}\n```"
    if prev_imports and lean_error:
        user += (
            "\n\nYour previous import list was:\n```\n" + prev_imports +
            "\n```\nLean failed to elaborate the statement with this error:\n"
            "```\n" + lean_error[:1500] + "\n```\n"
            "Reply with a corrected, complete import list (again: import "
            "lines only)."
        )
    return LLM_IMPORT_SYSTEM, user


def parse_llm_imports(raw: str, cap: int = 15) -> str | None:
    """Validate the model's reply into a newline-joined import block.

    Purely syntactic: keep lines matching `import <dotted.ident>`, dedupe,
    cap the count. Returns None when nothing valid survives."""
    out: list[str] = []
    seen: set[str] = set()
    for line in (raw or "").splitlines():
        line = line.strip().strip("`").strip()
        if _IMPORT_LINE_RE.match(line) and line not in seen:
            seen.add(line)
            out.append(line)
        if len(out) >= cap:
            break
    return "\n".join(out) if out else None


def resolve_imports_llm(statement_text: str,
                        llm_call: Callable[[str, str], str],
                        compile_check: Callable[[str], str | None],
                        max_rounds: int = 3) -> tuple[str, str]:
    """Returns (imports, source_tag).

    `llm_call(system, user) -> raw text`; `compile_check(imports)` compiles
    `imports + statement := by sorry` and returns None on success (sorry
    warning allowed) or the Lean error text on failure.

    source_tag is one of: "llm", "llm_fixN", "mathlib_fallback",
    "unresolved".
    """
    imports: str | None = None
    error: str | None = None
    for round_no in range(max_rounds):
        sys_p, user_p = llm_import_prompt(
            statement_text,
            prev_imports=imports if error else None,
            lean_error=error)
        try:
            raw = llm_call(sys_p, user_p)
        except Exception:
            break
        cand = parse_llm_imports(raw)
        if cand is None:
            continue
        imports = cand
        error = compile_check(imports)
        if error is None:
            return imports, ("llm" if round_no == 0 else f"llm_fix{round_no}")
    # Last resort: the whole library. Slow but universally correct.
    if compile_check("import Mathlib") is None:
        return "import Mathlib", "mathlib_fallback"
    # Statement is genuinely broken against this pin — return the best
    # attempt; the caller records the failure.
    return (imports or "import Mathlib"), "unresolved"
