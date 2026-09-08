"""Run-scoped store of kernel-accepted invented lemmas.

The ARM theory loop proves lemmas one at a time and commits a theory only
when EVERY lemma in it proved (`proof_dag.attempt_dag_proof`, the
`any_failed` branch). Until now the cache of accepted proofs lived inside
a single `_abduce_theory` call, so:

* a theory that abandoned on `revision rounds exhausted` discarded every
  lemma it had already proved, and
* the next sketch attempt — and the leaf-stuck / closer-stuck / bare-timeout
  trigger sites — each started from an empty cache.

Measured cost on `amc12a_2021_p25` (run `p25_2021_legacyARM_0729`): six
lemmas kernel-accepted, two theories abandoned, all six discarded, and
later proposals re-derived the same cube-root arithmetic from scratch.

This module is that cache, hoisted to **per-problem** scope. It is
deliberately:

* **in-memory only** — nothing is written to disk, so a run cannot mutate
  the persistent cross-run bank (`results/invented_lemmas.lean`) or any
  other repo state. The store dies with the problem.
* **per-problem, not per-process** — reuse across problems in one eval run
  would be cross-problem transfer and would contaminate a matched-budget
  ablation. The owner (`run_dag`) constructs one per problem.

Soundness. A cache hit can never manufacture a proof. Every lemma stored
here was accepted by `verify_fn` (a real Lean compile) when it was
proved, and any theory that reuses it is still re-verified by the normal
repair/verify loop and ultimately by a fresh-session compile. A stale hit
can therefore only cost a wasted verify, never produce a false solve —
the kernel remains the only oracle.

Import guard. Import sets mutate DURING a problem: `refresh_imports_call`
rewrites the verify environment whenever the theory reaches for API the
statement never forced. A lemma proved under one import set is not
automatically meaningful under another, so each record carries the
imports it was proved under and a lookup hits only when the current
import set is a superset. That is a practical, not a formal, criterion
(adding imports can in principle perturb elaboration via ambiguous names
or instance selection) — it is safe here precisely because of the
soundness argument above: the worst case is a failed re-verify.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WS_RE = re.compile(r"\s+")
_IMPORT_RE = re.compile(r"^\s*import\s+(\S+)", re.M)


def statement_key(statement: str) -> str:
    """Cache key for a lemma statement.

    Whitespace-stripped, and the declared NAME is deliberately part of the
    key: a stored declaration is reused verbatim, and the leaf tactics of
    the proposing theory call it by name. A revision that renames a lemma
    therefore misses and re-proves — conservative, and identical to the
    key the in-call `proved_bank` used before this module existed.
    """
    return _WS_RE.sub("", statement or "")


def imports_of(prelude: str) -> frozenset[str]:
    """The set of imported module names in a header prelude."""
    return frozenset(_IMPORT_RE.findall(prelude or ""))


_DECL_NAME_RE = re.compile(
    r"\s*(?:@\[[^\]]*\]\s*)*"
    r"(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(?:def|abbrev|structure|inductive|instance)\s+"
    r"([A-Za-z_][A-Za-z0-9_'!?₀-₉.]*)")


def decl_name(src: str) -> str:
    """The declared name of an auxiliary `def`/`abbrev`/… block, or "".

    Used to deduplicate spliced declarations by IDENTITY rather than by
    source text: two bodies under one name are a header error, not two
    declarations.
    """
    m = _DECL_NAME_RE.match(src or "")
    return m.group(1) if m else ""


@dataclass(slots=True)
class StoredLemma:
    key: str
    name: str
    statement: str
    decl: str                    # "<statement> := by\n  <proof>"
    imports: frozenset[str]      # imports in scope when it was proved
    sketch_attempt: int
    theory_round: int
    #: Auxiliary `def`/`abbrev`/`structure` declarations the lemma's
    #: STATEMENT refers to. A theory's defs reach the header only when the
    #: theory COMMITS, so a lemma proved inside an ABANDONED theory is
    #: meaningless on its own: reusing it alone yields `Unknown
    #: identifier`, and under `autoImplicit` the stray name becomes a
    #: metavariable ("Function expected at aux_weighted … ?m.1"). That is
    #: exactly how --store-feedback killed putnam_2020_a2_v1 at t=3059s,
    #: after 7 lemmas had been proved. Anything splicing stored lemmas
    #: must splice `required_defs()` first.
    defs: tuple[str, ...] = ()
    hits: int = 0


class LemmaStore:
    """Per-problem bank of kernel-accepted lemma declarations.

    `put` records a lemma the verifier accepted; `get` returns it only if
    the current import set still covers the one it was proved under.
    """

    __slots__ = ("_by_key", "_hits", "_misses", "_import_rejects")

    def __init__(self) -> None:
        self._by_key: dict[str, StoredLemma] = {}
        self._hits = 0
        self._misses = 0
        self._import_rejects = 0

    def get(self, statement: str, prelude: str) -> StoredLemma | None:
        rec = self._by_key.get(statement_key(statement))
        if rec is None:
            self._misses += 1
            return None
        if not rec.imports <= imports_of(prelude):
            # Proved under imports no longer in scope — treat as absent
            # rather than hand back a declaration that may not elaborate.
            self._import_rejects += 1
            self._misses += 1
            return None
        rec.hits += 1
        self._hits += 1
        return rec

    def put(self, statement: str, name: str, decl: str, prelude: str, *,
            defs: tuple[str, ...] | list[str] = (),
            sketch_attempt: int = 0, theory_round: int = 0) -> StoredLemma:
        key = statement_key(statement)
        rec = self._by_key.get(key)
        if rec is not None:
            return rec          # first proof wins; never overwrite
        rec = StoredLemma(
            key=key, name=name, statement=statement, decl=decl,
            defs=tuple(defs),
            imports=imports_of(prelude),
            sketch_attempt=sketch_attempt, theory_round=theory_round,
        )
        self._by_key[key] = rec
        return rec

    def records(self) -> list[StoredLemma]:
        return list(self._by_key.values())

    def required_defs(self) -> list[str]:
        """Every auxiliary decl any stored lemma needs, deduplicated BY
        DECLARATION NAME and in first-seen order. A caller splicing stored
        lemmas MUST splice these first, or the lemmas reference identifiers
        that do not exist.

        Dedup is by NAME, not by source text. A theory that revises a def
        across rounds produces two records carrying the same name with
        different bodies, and text-level dedup lets BOTH through — Lean
        then reports "has already been declared", a header error, which
        stops the repair loop by design. MEASURED on p1965a2_dag_v1: the
        theory rewrote

            def aux_chooseDeviationTerm (n r : N) : Z := (... ) ^ 2
            def aux_chooseDeviationTerm (n r : N) : Z := ((... : Z) ^ (2 : N))

        across rounds 0 and 1; both were spliced into sketch attempt 3's
        header at t=40799s and the attempt died on its first compile with
        three proved lemmas in hand.

        First-seen wins, matching `put`'s "first proof wins" policy. See
        `incompatible_names()` for the records this can invalidate.
        """
        out: list[str] = []
        seen: set[str] = set()
        for r in self._by_key.values():
            for d in r.defs:
                n = decl_name(d)
                if not n:
                    if d not in out:
                        out.append(d)
                    continue
                if n in seen:
                    continue
                seen.add(n)
                out.append(d)
        return out

    def incompatible_names(self) -> set[str]:
        """Names of stored lemmas that must NOT be surfaced.

        A lemma proved under one version of a def is not valid under a
        different version of the same-named def. Since `required_defs()`
        keeps only the first-seen body per name, any record needing a
        DIFFERENT body for that name would be spliced into a header where
        its def means something else — sound only by accident. Such
        records are reported here so callers can drop them.
        """
        chosen: dict[str, str] = {}
        for d in self.required_defs():
            n = decl_name(d)
            if n:
                chosen[n] = d
        bad: set[str] = set()
        for r in self._by_key.values():
            for d in r.defs:
                n = decl_name(d)
                if n and chosen.get(n, d) != d:
                    bad.add(r.name)
                    break
        return bad

    def names(self) -> list[str]:
        return [r.name for r in self._by_key.values()]

    def __len__(self) -> int:
        return len(self._by_key)

    def stats(self) -> dict:
        """Telemetry for the run row: how much re-proving this saved."""
        return {
            "size": len(self._by_key),
            "hits": self._hits,
            "misses": self._misses,
            "import_rejects": self._import_rejects,
            "names": self.names(),
        }
