"""Blueprint-first sketching: plan the STATEMENTS, then prove them.

The legacy sketch contract (`proof_dag.SKETCH_SYSTEM`) asks the model for
every `have` **together with its tactic** and the closer, in one JSON
object. That forces a commitment to tactics at the moment the model knows
least — before any Lean feedback and before anyone has checked whether the
decomposition is even logically sufficient.

Two costs are visible in the traces.

*The tactics are largely wasted output.* In `lrs_sol_legacyARM_store_v3`
the attempt-1 sketch (107s) proposed nine haves whose tactics were
`aesop`/`omega`/`simp_all`, and the closer was `aesop` in two of three
sketch attempts. Everything of value in that 8h22m run came from theory
mode — which asks for lemma STATEMENTS ONLY — and theory mode does not get
a turn until a leaf has been broken twice, i.e. after 1,500-4,500s of
failed compiles.

*Failure is unattributable.* When an assembled sketch fails to verify, the
pipeline cannot distinguish "the tactics are bad" from "these claims do
not imply the goal". It owns the instrument that separates them — the
sorry-stubbed sufficiency probe — but uses it only inside ARM, never on
the initial sketch.

This module supplies the other contract: statements only, ordered, each
with its dependencies. Tactics are left empty, which the assembler already
handles — `assembly._wrap_with_fallback` turns an empty tactic into
`first | <ladder>` — so a blueprint costs no assembly change and still
compiles in ONE verify per round. Per-lemma discharge is deliberately NOT
adopted here: the amortisation of a single assembled compile is what makes
the DAG affordable (theory mode's per-lemma proving is exactly why the LRS
run spent 2,046-5,259s per lemma).

Generality: nothing here names a benchmark, a problem, or a mathematical
domain. The statement is opaque text passed through to the policy.

Anchors: the workflow imitated is the Lean community's *blueprint*
practice (Massot's `leanblueprint`; the PFR project), where the dependency
graph of statements is fixed before any tactic is written; the
draft/sketch/prove decomposition of Jiang et al.; and POETRY-style
recursive discharge, which here remains reactive (`decompose_depth`)
rather than eager.

This module holds NO Lean and NO LLM access, and does not import
`proof_dag` — it returns plain data that the caller turns into a `Sketch`.
"""
from __future__ import annotations

import json
import re
from typing import Callable

_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*$")
_FORBIDDEN_TAC_RE = re.compile(r"\b(?:sorry|admit|native_decide)\b")


BLUEPRINT_SYSTEM = """\
You are a Lean 4 + Mathlib expert. You are writing the BLUEPRINT of a
proof: the ordered list of intermediate claims that, once each is proved,
together discharge the goal.

DO NOT WRITE TACTICS FOR THE CLAIMS. Your job here is to get the
mathematical decomposition right. Tactics come later, with Lean feedback;
committing to them now is guesswork and is not wanted.

WHAT MAKES A GOOD BLUEPRINT
- Each claim is a syntactically valid Lean 4 + Mathlib proposition.
- The claims are ordered: every claim may use the theorem's hypotheses,
  anything introduced in `setup`, and any EARLIER claim by its id.
- `depends` lists exactly the earlier claim ids a claim relies on.
- The final `closer` is a tactic block that closes the ORIGINAL goal from
  the theorem's hypotheses plus your claims. This is the one place a
  tactic IS required, because it is what makes the blueprint checkable.
- Prefer claims that are individually much easier than the goal. A claim
  restating the goal, or as hard as it, is worthless.
- State the claims you actually need. A blueprint that omits a step is
  worse than one with an extra step, because the missing step is where
  the proof will silently fail.

SELF-CHECK BEFORE ANSWERING
Ask: if every one of my claims were handed to me as a proved fact, would
`closer` really close the goal? If not, add the missing claim. This is
checked mechanically and a blueprint that fails it is sent back to you.

OUTPUT — reply with exactly this JSON object and NOTHING else:

{
  "setup": ["<optional tactic lines run before any claim>"],
  "claims": [
    {"id": "<lean identifier>", "type": "<Lean proposition>", "depends": ["<earlier id>", "..."]}
  ],
  "closer": "<tactic block closing the original goal>"
}

RULES
- `id` must be a valid Lean identifier, unique, and referenced in
  `depends` only by claims that come LATER.
- NEVER use `sorry`, `admit` or `native_decide` anywhere.
- Omit `setup` when the proof needs no context first. Use it for `intro`,
  `obtain`, `by_contra`, induction case openers.
- No `tactic` field on claims. If you include one it is ignored.\
"""


SEQUENTIAL_SYSTEM = """\
You are a Lean 4 + Mathlib expert building the BLUEPRINT of a proof ONE
CLAIM AT A TIME.

You will be shown the goal and the claims established so far. Give the
SINGLE next intermediate claim that makes most progress, or declare the
blueprint complete.

DO NOT WRITE A TACTIC FOR THE CLAIM. Decompose; do not prove.

Answer with exactly this JSON and nothing else:

{
  "done": false,
  "claim": {"id": "<lean identifier>", "type": "<Lean proposition>", "depends": ["<earlier id>", "..."]},
  "closer": ""
}

or, when the established claims suffice:

{
  "done": true,
  "claim": null,
  "closer": "<tactic block closing the original goal from the hypotheses and the established claims>"
}

RULES
- One claim per reply. It may use the theorem's hypotheses, the `setup`
  bindings, and any established claim by id.
- Declare `done` as soon as the established claims genuinely suffice —
  padding the blueprint wastes effort. But do NOT declare done while a
  step is still missing.
- NEVER use `sorry`, `admit` or `native_decide`.\
"""


def blueprint_user_prompt(statement: str,
                          failures: list[str] | None = None,
                          premises: list[str] | None = None) -> str:
    """User turn for one-shot blueprint generation."""
    out = ["Produce the blueprint for this Lean 4 theorem.", "",
           "```lean", statement.strip(), "```"]
    if premises:
        out += ["", "Mathlib declarations that may be relevant (names "
                    "only; verify signatures yourself):",
                ", ".join(premises[:40])]
    if failures:
        out += ["", "YOUR PREVIOUS BLUEPRINT WAS REJECTED. Fix it:"]
        for i, f in enumerate(failures[-3:], 1):
            out += ["", f"--- rejection {i} ---", f[:1200]]
    return "\n".join(out)


def sequential_user_prompt(statement: str,
                           established: list[dict],
                           premises: list[str] | None = None,
                           failures: list[str] | None = None) -> str:
    """User turn for one step of sequential (conditioned) generation."""
    out = ["Goal:", "```lean", statement.strip(), "```", ""]
    if established:
        out.append("Claims established so far (usable by id):")
        for c in established:
            out.append(f"  {c['id']} : {c['type']}")
    else:
        out.append("No claims established yet — this is the first.")
    if premises:
        out += ["", "Possibly relevant Mathlib names:",
                ", ".join(premises[:40])]
    if failures:
        out += ["", "The previous reply was rejected:"]
        for f in failures[-2:]:
            out.append(f"  {f[:400]}")
    out += ["", "Give the single next claim, or declare the blueprint "
                "complete with a closer."]
    return "\n".join(out)


def _extract_json_object(text: str) -> dict | None:
    """First balanced top-level JSON object, tolerating fences and prose."""
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


def _norm_claim(raw: dict) -> dict | None:
    if not isinstance(raw, dict):
        return None
    cid = str(raw.get("id") or "").strip()
    typ = str(raw.get("type") or "").strip()
    if not cid or not typ or not _ID_RE.match(cid):
        return None
    dep = raw.get("depends") or []
    if isinstance(dep, str):
        dep = [dep]
    if not isinstance(dep, list):
        return None
    return {"id": cid, "type": typ,
            "depends": [str(d).strip() for d in dep if str(d).strip()]}


def parse_blueprint(raw: str) -> tuple[dict | None, str | None]:
    """Parse a one-shot blueprint response.

    Returns ({"setup": [...], "claims": [...], "closer": str}, None) or
    (None, reason). Any `tactic` field on a claim is dropped: tactics are
    not part of this contract.
    """
    obj = _extract_json_object(raw or "")
    if obj is None:
        return None, "no JSON object in response"
    claims_raw = obj.get("claims")
    if not isinstance(claims_raw, list) or not claims_raw:
        return None, "no claims"
    claims = []
    for c in claims_raw:
        n = _norm_claim(c)
        if n is None:
            return None, f"malformed claim: {str(c)[:120]}"
        claims.append(n)
    closer = str(obj.get("closer") or "").strip()
    if not closer:
        return None, "no closer"
    setup = obj.get("setup") or []
    if isinstance(setup, str):
        setup = [setup]
    if not isinstance(setup, list):
        return None, "setup is not a list"
    setup = [str(s) for s in setup if str(s).strip()]
    blob = closer + "\n" + "\n".join(setup)
    if _FORBIDDEN_TAC_RE.search(blob):
        return None, "forbidden tactic (sorry/admit/native_decide)"
    return {"setup": setup, "claims": claims, "closer": closer}, None


def parse_sequential_step(raw: str) -> tuple[dict | None, str | None]:
    """Parse one step of sequential generation.

    Returns ({"done": bool, "claim": dict|None, "closer": str}, None).
    """
    obj = _extract_json_object(raw or "")
    if obj is None:
        return None, "no JSON object in response"
    done = bool(obj.get("done"))
    closer = str(obj.get("closer") or "").strip()
    if done:
        if not closer:
            return None, "done=true but no closer"
        if _FORBIDDEN_TAC_RE.search(closer):
            return None, "forbidden tactic in closer"
        return {"done": True, "claim": None, "closer": closer}, None
    claim = _norm_claim(obj.get("claim") or {})
    if claim is None:
        return None, "done=false but no usable claim"
    return {"done": False, "claim": claim, "closer": ""}, None


def validate_references(claims: list[dict]) -> str | None:
    """Reject dangling and forward references, and duplicate ids.

    Cheap replacement for what sequential conditioning is often argued to
    buy. The dangling-reference slip is real and recurrent: the ARM loop
    cited `aux_ratio_lt` (p25) and `aux_nonzero_trailing_coefficient_of_no_zero`
    (LRS) — neither of which its own proposal contained — at a cost of one
    full compile each. Catching it here costs nothing.
    """
    seen: set[str] = set()
    for c in claims:
        if c["id"] in seen:
            return f"duplicate claim id `{c['id']}`"
        for d in c["depends"]:
            if d not in seen:
                return (f"claim `{c['id']}` depends on `{d}`, which is not "
                        f"an earlier claim")
        seen.add(c["id"])
    return None


def generate_blueprint(
    statement: str,
    *,
    llm_call: Callable[[str, str], str],
    premises: list[str] | None = None,
    sequential: bool = False,
    max_claims: int = 12,
    max_steps: int = 24,
    trace: Callable[..., None] | None = None,
) -> tuple[dict | None, str | None]:
    """Produce a validated blueprint, one-shot or sequentially.

    Sequential mode issues one call per claim, each conditioned on the
    claims established so far, until the model declares completion. It
    buys adaptivity — a later claim can supply what an earlier one turned
    out to need — at the cost of k calls instead of 1, and is an ablation
    lever rather than the default.
    """
    tr = trace or (lambda kind, **kw: None)

    if not sequential:
        try:
            raw = llm_call(BLUEPRINT_SYSTEM,
                           blueprint_user_prompt(statement, None, premises))
        except Exception as e:
            return None, f"llm error: {type(e).__name__}: {e}"
        bp, err = parse_blueprint(raw)
        if bp is None:
            return None, err
        verr = validate_references(bp["claims"])
        if verr:
            return None, verr
        tr("blueprint", stage="generated", mode="one_shot",
           n_claims=len(bp["claims"]))
        return bp, None

    established: list[dict] = []
    failures: list[str] = []
    for step in range(max_steps):
        try:
            raw = llm_call(
                SEQUENTIAL_SYSTEM,
                sequential_user_prompt(statement, established, premises,
                                       failures))
        except Exception as e:
            return None, f"llm error: {type(e).__name__}: {e}"
        stepobj, err = parse_sequential_step(raw)
        if stepobj is None:
            failures.append(err or "?")
            tr("blueprint", stage="step_rejected", step=step, detail=err)
            if len(failures) >= 3:
                return None, f"sequential generation failed: {err}"
            continue
        if stepobj["done"]:
            if not established:
                return None, "model declared done with no claims"
            tr("blueprint", stage="generated", mode="sequential",
               n_claims=len(established))
            return {"setup": [], "claims": established,
                    "closer": stepobj["closer"]}, None
        claim = stepobj["claim"]
        trial = established + [claim]
        verr = validate_references(trial)
        if verr:
            failures.append(verr)
            tr("blueprint", stage="step_rejected", step=step, detail=verr)
            if len(failures) >= 3:
                return None, verr
            continue
        established = trial
        failures = []
        tr("blueprint", stage="claim_added", step=step, id=claim["id"])
        if len(established) >= max_claims:
            return None, (f"sequential generation exceeded {max_claims} "
                          f"claims without completing")
    return None, f"sequential generation exceeded {max_steps} steps"
