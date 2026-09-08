"""Pick a minimal `import` set from a theorem statement.

Spec: PART D of the candidate-selection / clean-eval fix-up.

Goal: when clean evaluation is on (`--clean-eval-no-theorem-id-imports`)
we cannot peek at the theorem ID to pick a per-problem import profile.
Falling back to `import Mathlib` blows past the REPL startup budget on
this Windows host (~600s vs ~180s for the minimal p100 imports). This
module derives a Mathlib import set from the same `ProofStateFeatures`
the prior already extracts, plus a small hand-curated rule table.

The rules are intentionally conservative:
  - Each rule maps a feature-bag predicate to a tuple of import lines.
  - Multiple matched rules are unioned (order-preserving).
  - `import Mathlib.Tactic` is only appended when an analytical /
    inequality-shape goal is detected — bare arithmetic / decidability
    problems don't need it and the smaller import is much faster.
  - When no rule fires, the caller is responsible for the
    `import Mathlib` fallback (so the JSONL row can distinguish
    `import_source=inferred` from `import_source=fallback_mathlib`).

The output never depends on the theorem name.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from search.proof_prior import ProofStateFeatures
from search.state_features import extract_state_features


@dataclass(slots=True)
class ImportInference:
    """Result of `infer_imports_from_header`."""
    imports: str          # newline-joined `import …` lines
    matched_rules: list[str] = field(default_factory=list)
    # True iff at least one rule matched; False means the caller should
    # decide between this (still-valid) suggestion and a Mathlib-wide
    # fallback. By convention we return DEFAULT_IMPORTS in that case.
    matched: bool = False


DEFAULT_IMPORTS: str = "import Mathlib"


def _has_any(values, needle: set[str]) -> bool:
    return any(v in needle for v in values)


# Header-substring probes used when a needed symbol isn't yet in the
# feature bag (e.g. `Irrational`, `floor`, `filter`). Kept here rather
# than extended into state_features to avoid bloating the feature bag
# for non-import use cases. Predicate signature: (header_text) -> bool.
_HEADER_HAS_IRRATIONAL_RE = re.compile(
    r"\bIrrational\b|\birrational\b"
)
_HEADER_HAS_REAL_POW_RE = re.compile(
    r"Real\.rpow|\^\s*\(\s*\d|ℝ[^=]*\^|\(\s*\d+\s*:\s*ℝ\s*\)\s*\^"
)
_HEADER_HAS_FILTER_RE = re.compile(
    r"Finset\.filter|\bfilter\b"
)
_HEADER_HAS_FLOOR_RE = re.compile(
    r"⌊|⌋|\bfloor\b|Int\.floor|Nat\.floor"
)


def _header_has(header: str, pat: re.Pattern[str]) -> bool:
    if not header:
        return False
    return bool(pat.search(header))


def _rules() -> list[tuple[str, callable, tuple[str, ...]]]:
    """Each rule = (rule_name, predicate, import-lines).

    Predicates take `(features, header_text)`. Import lines are emitted
    in declaration order; downstream dedup preserves the first match.
    A rule fires if its predicate returns True.

    Ordering note: more specific rules come first so that, e.g., a
    `Real.sqrt` header matches `real_sqrt` and not just
    `real_analytical_inequality`. Multiple rules can fire and contribute
    to the union — that is by design: `Finset` + inequality should pull
    in both BigOperators and Tactic.
    """
    return [
        (
            "nat_gcd_lcm",
            lambda f, h: (_has_any(f.symbols, {"gcd", "lcm"})
                          and "Nat" in f.namespaces),
            ("import Mathlib.Data.Nat.GCD.Basic",),
        ),
        (
            "real_sqrt",
            # Trigger on EITHER the feature path or a literal `√` in the
            # header text. `√` is the Lean notation that
            # `state_features` doesn't decompose into `sqrt`.
            lambda f, h: (
                ("sqrt" in f.symbols and "Real" in f.namespaces)
                or "√" in (h or "")
            ),
            ("import Mathlib.Analysis.SpecialFunctions.Sqrt",
             "import Mathlib.Tactic"),
        ),
        (
            "factorial",
            lambda f, h: "factorial" in f.symbols,
            ("import Mathlib.Data.Nat.Factorial.Basic",
             "import Mathlib.Tactic"),
        ),
        (
            # PART B fix-up: include Finset.filter / `filter` so theorems
            # like `Finset.prod (Finset.filter …) …` pull the right deps
            # even when the symbol bag missed `sum`/`prod`.
            "finset_big_ops",
            lambda f, h: (
                (_has_any(f.symbols, {"sum", "prod", "range"})
                 and "Finset" in f.namespaces)
                or _header_has(h, _HEADER_HAS_FILTER_RE)
                or "∑" in (h or "")
                or "∏" in (h or "")
            ),
            # NOTE: `Mathlib.Algebra.BigOperators.Basic` does NOT exist in
            # the pinned Mathlib (ab9605ce) — it was split into
            # BigOperators.Group.*; emitting it fails every verify at
            # line 1 with a missing-olean error (burned amc12a_2020_p25
            # in run smoke10_dag_v2_opus48).
            ("import Mathlib.Data.Finset.Basic",
             "import Mathlib.Algebra.BigOperators.Group.Finset.Basic",
             "import Mathlib.Tactic"),
        ),
        (
            # PART B (new): real arithmetic shapes — ℝ + (inequality or
            # power) — pull Real.Basic + Tactic. The strong `Real.Basic`
            # add is what `real_analytical_inequality` was missing, so
            # nlinarith-shaped goals now compile without falling back to
            # full Mathlib.
            "real_arithmetic",
            lambda f, h: (
                _has_any(f.symbols, {"inequality", "power", "equality"})
                and "Real" in f.namespaces
            ),
            ("import Mathlib.Data.Real.Basic",
             "import Mathlib.Tactic"),
        ),
        (
            # Complex arithmetic: ℂ + (inequality OR equality OR power)
            # → Complex.Basic + Tactic. Distinct module from Real because
            # Complex.Basic isn't pulled by Real.Basic. Without this
            # rule a `(f z : ℂ)` linear-system theorem falls back to
            # full `import Mathlib` and hits the 300s import budget.
            "complex_arithmetic",
            lambda f, h: (
                _has_any(f.symbols, {"inequality", "power", "equality"})
                and "Complex" in f.namespaces
            ),
            ("import Mathlib.Data.Complex.Basic",
             "import Mathlib.Tactic"),
        ),
        (
            # Real powers / irrationality — distinct module because
            # rpow / Irrational live under SpecialFunctions.Pow.Real,
            # not Real.Basic.
            "real_pow_or_irrational",
            lambda f, h: (
                _header_has(h, _HEADER_HAS_IRRATIONAL_RE)
                or _header_has(h, _HEADER_HAS_REAL_POW_RE)
                or ("power" in f.symbols
                    and _has_any(f.namespaces, {"Real", "Complex"}))
            ),
            ("import Mathlib.Analysis.SpecialFunctions.Pow.Real",
             "import Mathlib.Tactic"),
        ),
        (
            # Kept for back-compat: the old rule still fires for goals
            # that don't trip the more specific `real_arithmetic` /
            # `real_pow_or_irrational` checks. The import set is a
            # subset of `real_arithmetic`, so the union is identical
            # when both fire.
            "real_analytical_inequality",
            lambda f, h: (
                _has_any(f.symbols, {"inequality", "power"})
                and _has_any(f.namespaces, {"Real", "Complex"})
            ),
            ("import Mathlib.Tactic",),
        ),
        (
            "floor_or_ceil",
            lambda f, h: _header_has(h, _HEADER_HAS_FLOOR_RE),
            ("import Mathlib.Algebra.Order.Floor",
             "import Mathlib.Data.Real.Basic",
             "import Mathlib.Tactic"),
        ),
        (
            "nat_modulo",
            lambda f, h: "modulo" in f.symbols and "Nat" in f.namespaces,
            ("import Mathlib.Data.Nat.Basic", "import Mathlib.Tactic"),
        ),
        # NOTE (2026-07-26): deliberately NO geometry rule here. A rule
        # keyed to segment/Wbtw/Collinear was added after b5_bare_v1 and
        # removed the same day as benchmark-tuning: the general
        # mechanism (LLM import resolution + the error-driven ARM
        # import refresh in run_dag/proof_dag) must handle proof-layer
        # import gaps without problem-shaped module lists in code.
    ]


def infer_imports_from_features(
    features: ProofStateFeatures,
    header_text: str = "",
) -> ImportInference:
    """Apply the rule table to a feature bag and return the union of
    matching import lines. Returns DEFAULT_IMPORTS with matched=False
    when no rule fires.

    `header_text` lets header-substring rules (e.g. `√`, `Irrational`,
    `⌊`) trigger even when the canonical feature bag misses them.
    Default empty string keeps the old call-site signature working —
    those callers only get the feature-bag rules, never the substring
    ones. Passing the header is strongly recommended for production.
    """
    matched_names: list[str] = []
    emitted: list[str] = []
    seen: set[str] = set()
    for name, pred, lines in _rules():
        try:
            if not pred(features, header_text):
                continue
        except Exception:
            continue
        matched_names.append(name)
        for line in lines:
            if line not in seen:
                seen.add(line)
                emitted.append(line)
    if not emitted:
        return ImportInference(imports=DEFAULT_IMPORTS, matched=False)
    return ImportInference(
        imports="\n".join(emitted),
        matched_rules=matched_names,
        matched=True,
    )


def infer_imports_from_header(header_text: str) -> ImportInference:
    """Convenience wrapper that extracts features from a theorem header.

    `header_text` is the same string `load_header` produces for the
    runner — `theorem <name> ... := by sorry`. We extract features as
    if it were a goal and apply the rule table."""
    features = extract_state_features(header_text or "", proof_prefix=[])
    return infer_imports_from_features(features, header_text or "")


# --------------------------------------------------------------------------
# LLM-driven import resolution (no hardcoded module knowledge).
#
# The rule table above only knows the miniF2F-shaped world; statement-heavy
# benchmarks (ProofNet, PutnamBench) open namespaces the rules have never
# heard of. `llm_suggest_imports` asks the policy model itself which
# Mathlib modules a STATEMENT needs, and `resolve_imports_llm` closes the
# loop: compile the bare statement as a gate, feed any header-level error
# back to the model, retry. Nothing here names a Mathlib module.
# --------------------------------------------------------------------------

_IMPORT_LINE_RE = re.compile(r"^import\s+[A-Za-z][\w.]*$")

_LLM_IMPORT_SYSTEM = (
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
    return _LLM_IMPORT_SYSTEM, user


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


def resolve_imports_llm(statement_text: str, llm_call, compile_check,
                        max_rounds: int = 3) -> tuple[str, str]:
    """Returns (imports, source_tag).

    `llm_call(system, user) -> raw text`; `compile_check(imports) ->
    error_text | None` compiles `imports + statement := by sorry` and
    returns None on success (sorry warning allowed) or the error text.
    Escalation: llm -> llm+error-fix (max_rounds total) -> `import Mathlib`.
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
    err = compile_check("import Mathlib")
    if err is None:
        return "import Mathlib", "mathlib_fallback"
    # Statement is genuinely broken against this pin — return the best
    # attempt; the caller records the failure.
    return (imports or "import Mathlib"), "unresolved"
