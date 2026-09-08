"""Environment-integrity guard for model-supplied declarations.

THE KERNEL IS THE ORACLE — but only if the statement still means what we
wrote. Every abduction path (`proof_dag._parse_theory`'s `defs`,
`reframe`'s `objects`, `library.build_library`'s declarations) splices
model-authored declarations into the file BEFORE the goal is elaborated.
A declaration that mutates the ambient environment therefore changes what
the goal SAYS, and the kernel will faithfully verify the new, weaker
statement.

Observed live on `frankl_open_v1` (2026-08-08), which reported SOLVED on
Frankl's open conjecture:

    instance (priority := 10000) aux_instLENat : LE ℕ where
      le _ _ := True

    intro A h_nonempty h_union; exact ⟨0, True.intro⟩

`≤` on ℕ was redefined to `True` at a priority beating Mathlib's, so
`2 * card ≥ A.card` elaborated to `True` and `True.intro` closed it. The
witness `0` was never shown to be abundant. Re-verification in a fresh
session does NOT catch this: the fresh compile includes the poisoned
instance.

`sorry`/`admit`/`native_decide` scanning does not catch it either — the
proof contains none of them. The defence has to be at the point where a
declaration enters the file.

What is refused, and why each one is sufficient on its own to fake a
proof:

* `instance` / `local instance` / `scoped instance` — override an
  existing typeclass instance (the observed exploit);
* `axiom` — assume anything at all, including `False`;
* `attribute` / `@[...]` targeting existing names, `export`, `open ... in`
  — change elaboration or simp behaviour for declarations we did not
  write;
* `macro` / `notation` / `syntax` / `macro_rules` / `elab` — change what
  the surface syntax of the GOAL parses to;
* `set_option` — can switch off the checks the goal relies on;
* `deriving instance` — same reach as `instance`;
* `unsafe` / `opaque` / `implemented_by` / `extern` — escape hatches.

Legitimate abduction never needs any of these: a theory introduces
DEFINITIONS and LEMMAS about its own new objects. If a future round
genuinely needs an instance for a newly-defined structure, that is a
deliberate design change, not something a model should be able to do by
emitting a string.
"""
from __future__ import annotations

import re

#: Declaration heads that mutate the ambient environment.
_BANNED = (
    "instance", "axiom", "attribute", "macro_rules", "macro", "notation",
    "syntax", "elab", "set_option", "export", "deriving", "opaque",
    "implemented_by", "extern", "unsafe", "partial",
)

#: Head-of-line match: these are only dangerous as DECLARATIONS. The word
#: "instance" inside a statement (`Nonempty (Invertible M)`) is harmless,
#: so anchoring to line starts avoids rejecting honest mathematics.
_BANNED_RE = re.compile(
    r"(?m)^\s*(?:@\[[^\]]*\]\s*)*"
    r"(?:private\s+|protected\s+|noncomputable\s+|scoped\s+|local\s+)*"
    r"(?P<head>" + "|".join(_BANNED) + r")\b")

#: An attribute applied to something we did not define reaches back into
#: the existing environment (`@[simp] theorem Nat.foo ...` re-tagging, or
#: `@[instance]`). Attributes on the declaration's OWN new name are fine,
#: so only the dangerous attribute names are refused.
_BANNED_ATTR_RE = re.compile(
    r"@\[[^\]]*\b(instance|default_instance|simp\s+high|implemented_by"
    r"|extern|reducible_pattern)\b[^\]]*\]")


def unsafe_declaration(src: str) -> str | None:
    """The reason `src` may not be spliced, or None if it is safe.

    Refuses declarations that change the MEANING of the goal rather than
    adding to what is known. See the module docstring for why each head
    is on its own sufficient to manufacture a proof.
    """
    text = src or ""
    m = _BANNED_RE.search(text)
    if m:
        return (f"declaration head `{m.group('head')}` may not be introduced: "
                f"it mutates the ambient environment, so the goal would no "
                f"longer mean what the problem states")
    m2 = _BANNED_ATTR_RE.search(text)
    if m2:
        return (f"attribute `{m2.group(0)}` may not be introduced: it reaches "
                f"into the existing environment")
    return None


def first_unsafe(decls) -> tuple[str, str] | None:
    """First `(declaration, reason)` that must be refused, or None."""
    for d in decls or ():
        why = unsafe_declaration(d)
        if why:
            return (d, why)
    return None
