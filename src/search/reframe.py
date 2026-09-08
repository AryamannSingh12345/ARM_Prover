"""Theory invention: reframe a problem into a domain where it is routine.

Some problems resist until a new *perspective* arrives — the classical
example being solvability by radicals, which stayed open until a group was
attached to each polynomial and the question was transported into group
theory, where it became a computation.

That move has three parts, and none of them is "invent a definition":

1. **objects** — new defs/structures that constitute the new domain;
2. **a bridge** — statements connecting the problem's objects to the new
   ones, so a fact there means something here;
3. **leverage** — the theory's own theorems must be genuinely EASIER than
   the target, or the reframing has bought nothing.

`proof_dag._abduce_theory` already invents defs and lemmas, but always
*about the original objects* and always *in service of one stuck leaf*.
It has no notion of a bridge and no leverage test, so it cannot reframe a
problem — it can only decompose it. Across the runs measured here it
produced correct decompositions on every hard problem and never once
changed the frame.

Economics. The expensive step is proving; the cheap step is checking that
a theory WOULD suffice. So the order here is deliberately:

    propose  ->  leverage gate (free)  ->  sufficiency probe (1 compile)
             ->  prove the pieces (many compiles)

A theory that cannot close the target even with every one of its own
theorems granted is discarded before a single proof is attempted. This is
`_satisfied` lifted from leaf level to whole-problem level.

Soundness is unchanged: the probe tolerates `sorry` and NEVER counts as a
solve; only a real compile of the assembled proof does. A reframing can
therefore save work or waste a compile, never manufacture a result.

This module owns no Lean and no LLM access.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable

_FORBIDDEN_TAC_RE = re.compile(r"\b(?:sorry|admit|native_decide)\b")
_DECL_NAME_RE = re.compile(
    r"^\s*(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(?:lemma|theorem)\s+([A-Za-z_][A-Za-z0-9_'!?₀-₉.]*)")


REFRAME_SYSTEM = """\
You are a research mathematician working in Lean 4 with Mathlib.

You are given a theorem that resists direct proof. Your task is NOT to
decompose it into smaller steps in the same language — that has already
been tried. Your task is to CHANGE THE FRAME: introduce a different
mathematical setting in which this problem becomes routine.

This is the move that unlocks problems rather than grinding them: attach
new objects to the problem, transport the question into the world of those
objects, and answer it there.

A REFRAME HAS THREE PARTS
1. OBJECTS — the new setting. Definitions, structures, classes: the things
   your theory is about. They may be entirely new, or a specialisation of
   something Mathlib already has.
2. BRIDGE — statements tying the problem's data to your objects. Without
   these your theory is about nothing, and a fact proved in it says
   nothing about the target.
3. THEOREMS — the facts IN your setting that carry the argument.

THE TEST YOUR REFRAME MUST PASS
Granting every bridge and theorem for free, `derivation` must close the
ORIGINAL goal. This is checked mechanically before anything is proved, and
a reframe that fails it is sent back to you.

LEVERAGE — the point of the exercise
Your theorems must be genuinely EASIER than the original goal. A theory
whose main theorem is as hard as the target has moved the difficulty, not
reduced it, and is worthless. Prefer settings where the answer becomes a
computation, a finiteness argument, an invariant, or a known Mathlib
result.

PREFER EXISTING THEORY. If Mathlib already has the right setting, USE IT
and say so — an unnecessary reinvention is a worse answer than a citation.
Invent only what is genuinely absent.

OUTPUT — reply with exactly this JSON and NOTHING else:

{
  "name": "<short name for the perspective>",
  "rationale": "<why this frame makes the problem easy — the key idea, one or two sentences>",
  "objects": ["<complete Lean def/structure/class declaration>", "..."],
  "bridges": ["<Lean lemma statement, NO proof>", "..."],
  "theorems": ["<Lean lemma statement, NO proof>", "..."],
  "derivation": "<Lean tactic block closing the ORIGINAL goal, using the bridges and theorems by name>"
}

RULES
- `objects` are complete (they have bodies). `bridges` and `theorems` are
  STATEMENTS ONLY — they will be proved separately.
- Every bridge/theorem must be a named `lemma` or `theorem`.
- NEVER use `sorry`, `admit` or `native_decide` anywhere.
- If you genuinely cannot find a better frame, say so in `rationale` and
  return empty `objects`/`bridges`/`theorems`. An honest "no reframe" is a
  useful answer; a frame that does not close the goal is not.\
"""


def reframe_prompt(target: str, failures: list[str] | None = None,
                   premises: list[str] | None = None) -> str:
    out = ["This theorem has resisted direct proof:", "",
           "```lean", target.strip(), "```", "",
           "Propose a change of frame that makes it routine."]
    if premises:
        out += ["", "Mathlib names retrieved for this problem (the right "
                    "existing setting may be among them):",
                ", ".join(premises[:40])]
    if failures:
        out += ["", "PREVIOUS REFRAMES WERE REJECTED:"]
        for i, f in enumerate(failures[-3:], 1):
            out += ["", f"--- rejection {i} ---", f[:1000]]
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


@dataclass
class Theory:
    name: str = ""
    rationale: str = ""
    objects: list[str] = field(default_factory=list)
    bridges: list[str] = field(default_factory=list)
    theorems: list[str] = field(default_factory=list)
    derivation: str = ""

    @property
    def is_empty(self) -> bool:
        """The model declined to reframe."""
        return not (self.objects or self.bridges or self.theorems)

    @property
    def claims(self) -> list[str]:
        """Bridges and theorems — everything that needs proving."""
        return list(self.bridges) + list(self.theorems)

    def to_dict(self) -> dict:
        return asdict(self)


def claim_name(stmt: str) -> str:
    """The declared name of a bridge/theorem statement, or "".

    The derivation cites its claims BY NAME, so the name recorded for a
    claim must be the one actually declared in its statement — a
    bookkeeping name invented elsewhere would silently break the
    reference check in the post-proving minimise step.
    """
    m = _DECL_NAME_RE.match(stmt or "")
    return m.group(1) if m else ""


def theory_as_proposal(t: Theory, leaf_ids: Iterable[str],
                       extra_defs: Iterable[str] | None = None,
                       extra_claims: Iterable[str] | None = None) -> str:
    """Render a reframing in the ARM theory-JSON format.

    A reframing IS an ARM proposal with a wider vocabulary: objects are
    `defs`, bridges and theorems are `lemmas`, and the derivation is the
    tactic for the stuck leaf. Emitting that shape lets a reframe enter
    `_abduce_theory`'s existing prove-and-commit path — circularity
    guard, quality gate, satisfaction probe, per-lemma kernel proving,
    all-or-nothing commit — instead of duplicating any of it. The
    conversion is deliberately the ONLY new code path: everything that
    decides whether a theory is admissible stays in one place.

    `extra_defs` / `extra_claims` carry a support library built for the
    new objects. They are kept SEPARATE by kind because the two fields
    are treated differently downstream and the split is not cosmetic:
    `defs` are spliced verbatim and are matched by a regex accepting only
    `def|abbrev|instance|structure|inductive`, so a proved LEMMA placed
    there is silently dropped. Library lemmas therefore arrive as claims
    (already registered in the proved-lemma bank, so the prove step
    reuses them for zero compiles) and library definitions as defs.
    """
    objects = list(t.objects) + [d for d in (extra_defs or []) if d.strip()]
    claims = list(t.claims) + [c for c in (extra_claims or []) if c.strip()]
    return json.dumps({
        "defs": objects,
        "lemmas": [{"name": claim_name(c), "statement": c}
                   for c in claims],
        "leaf_tactics": {lid: t.derivation for lid in leaf_ids},
    })


def parse_reframe(raw: str) -> tuple[Theory | None, str | None]:
    obj = _extract_json_object(raw or "")
    if obj is None:
        return None, "no JSON object in response"

    def _strlist(key: str) -> list[str]:
        v = obj.get(key) or []
        if isinstance(v, str):
            v = [v]
        return [str(x).strip() for x in v if str(x).strip()] \
            if isinstance(v, list) else []

    t = Theory(
        name=str(obj.get("name") or "").strip(),
        rationale=str(obj.get("rationale") or "").strip(),
        objects=_strlist("objects"),
        bridges=_strlist("bridges"),
        theorems=_strlist("theorems"),
        derivation=re.sub(r"^\s*by\b", "",
                          str(obj.get("derivation") or "")).strip(),
    )
    if t.is_empty:
        return t, None                       # an honest "no reframe"
    blob = "\n".join(t.objects + t.claims + [t.derivation])
    if _FORBIDDEN_TAC_RE.search(blob):
        return None, "forbidden tactic (sorry/admit/native_decide)"
    # Environment integrity: `objects` are spliced VERBATIM before the goal
    # elaborates, so an `instance`/`axiom`/`notation` among them changes
    # what the goal means. See src/search/decl_safety.py for the live
    # exploit this prevents.
    from search.decl_safety import first_unsafe
    bad = first_unsafe(t.objects)
    if bad:
        return None, f"unsafe object declaration: {bad[1]}"
    if not t.derivation:
        return None, "no derivation: nothing connects the theory to the goal"
    for c in t.claims:
        if not _DECL_NAME_RE.match(c):
            return None, f"claim is not a named lemma/theorem: {c[:80]}"
    return t, None


def leverage_ok(t: Theory, target_goal: str,
                difficulty_fn: Callable[[str], float],
                factor: float = 0.9) -> tuple[bool, str]:
    """Reject a theory whose hardest claim is no easier than the target.

    A reframing that moves the difficulty instead of reducing it has
    bought nothing, and this costs no compile to detect. Same criterion
    (and same proxy) as the ARM decomposition-quality gate.
    """
    if not t.claims:
        return True, ""
    try:
        goal_d = difficulty_fn(target_goal)
        worst = max((difficulty_fn(c), c) for c in t.claims)
    except Exception:
        return True, ""                      # cannot judge -> do not block
    if goal_d <= 0:
        return True, ""
    if worst[0] >= factor * goal_d:
        return False, (
            f"no leverage: hardest claim scores {worst[0]:.2f} against a "
            f"target of {goal_d:.2f} — the frame moves the difficulty "
            f"rather than reducing it. Offending claim: {worst[1][:120]}")
    return True, ""


def assemble_theory_header(prelude: str, t: Theory,
                           stub: bool = True) -> str:
    """Header carrying the theory: objects, then claims.

    With `stub=True` every claim is `:= by sorry` — that is the
    sufficiency probe, which asks whether the theory WOULD close the goal
    before any of it is proved.
    """
    parts = [prelude.rstrip("\n")] if prelude.strip() else []
    parts += [o.strip() for o in t.objects if o.strip()]
    for c in t.claims:
        parts.append(f"{c.strip()} := by sorry" if stub else c.strip())
    return "\n\n".join(parts)


@dataclass
class ReframeResult:
    theory: Theory | None = None
    sufficient: bool = False
    rounds_used: int = 0
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"theory": self.theory.to_dict() if self.theory else None,
                "sufficient": self.sufficient,
                "rounds_used": self.rounds_used,
                "log": list(self.log)}


def propose_reframe(
    target_header: str,
    target_goal: str,
    *,
    llm_call: Callable[[str, str], str],
    probe_fn: Callable[[str, str], dict],
    prelude: str = "",
    rounds: int = 2,
    premises: list[str] | None = None,
    difficulty_fn: Callable[[str], float] | None = None,
    leverage_factor: float = 0.9,
    trace: Callable[..., None] | None = None,
) -> ReframeResult:
    """Find a reframing that WOULD close the target, cheapest checks first.

    Returns a theory only when the sufficiency probe passed, i.e. granting
    every bridge and theorem, the derivation closes the goal. Proving them
    is the caller's job — and the point is that the caller now knows the
    work is worth doing.
    """
    tr = trace or (lambda kind, **kw: None)
    res = ReframeResult()
    failures: list[str] = []

    for rnd in range(1, max(int(rounds), 1) + 1):
        res.rounds_used = rnd
        try:
            raw = llm_call(REFRAME_SYSTEM,
                           reframe_prompt(target_header, failures, premises))
        except Exception as e:
            res.log.append(f"round {rnd}: llm error: {type(e).__name__}")
            tr("reframe", round=rnd, stage="llm_error")
            break

        t, err = parse_reframe(raw)
        if t is None:
            res.log.append(f"round {rnd}: unparseable ({err})")
            tr("reframe", round=rnd, stage="unparseable", detail=err)
            failures.append(f"Your reply was rejected: {err}")
            continue
        if t.is_empty:
            res.theory = t
            res.log.append(f"round {rnd}: model found no better frame")
            tr("reframe", round=rnd, stage="declined",
               detail=t.rationale[:200])
            break

        if difficulty_fn is not None:
            ok, why = leverage_ok(t, target_goal, difficulty_fn,
                                  leverage_factor)
            if not ok:
                res.log.append(f"round {rnd}: {why[:120]}")
                tr("reframe", round=rnd, stage="no_leverage", detail=why[:300])
                failures.append(why)
                continue                     # zero compiles spent

        header = assemble_theory_header(prelude, t, stub=True)
        tr("reframe", round=rnd, stage="probing", name=t.name,
           n_objects=len(t.objects), n_claims=len(t.claims))
        try:
            gate = probe_fn(header + "\n\n" + target_header,
                            "  " + t.derivation.replace("\n", "\n  "))
        except Exception as e:
            res.log.append(f"round {rnd}: probe error: {type(e).__name__}")
            tr("reframe", round=rnd, stage="probe_error")
            break
        if gate.get("ok"):
            res.theory = t
            res.sufficient = True
            res.log.append(f"round {rnd}: SUFFICIENT — `{t.name}` closes the "
                           f"goal given its {len(t.claims)} claim(s)")
            tr("reframe", round=rnd, stage="SUFFICIENT", name=t.name,
               detail=t.rationale[:200])
            return res
        errs = (gate.get("errors") or "?")[:1200]
        res.theory = t
        res.log.append(f"round {rnd}: insufficient")
        tr("reframe", round=rnd, stage="insufficient", detail=errs[:300])
        failures.append(
            "Granting every bridge and theorem, your derivation did NOT "
            "close the goal. Lean reported:\n" + errs)

    return res
