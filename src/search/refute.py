"""Pre-flight refutation search: try to prove the goal FALSE before proving it.

Motivation, from this repo's own traces. Run `lrs_sol_legacyARM_store_v3`
spent 8h22m, 37 Lean compiles and $4.58 attempting `order5_lrs_a0_unit`,
whose conclusion is false — a witness with `a₀ = 2¹⁵` exists. Every
"content" lemma the ARM loop invented for it (`aux_negative_constant_
modular_obstruction` and friends) was a false statement, so the kernel
correctly refused all of them and the run could not have succeeded. A
cheap refutation pass in front of the prover turns that class of run from
an expensive failure into a fast, publishable NEGATIVE result.

Soundness — the single invariant this module rests on:

    THE MODEL PROPOSES, THE KERNEL DECIDES.

A refutation is believed only when Lean accepts a proof of
``¬ (∀ binders, goal)`` in a fresh compile, with no `sorry`/`admit`/
`native_decide`. An LLM asked to find a counterexample will confidently
invent one whether or not any exists (this repo has the scar: b5_bare_v1's
ARM proposed a false "wishful" lemma that passed the usefulness gate and
was rejected twice by the kernel). So nothing here is trusted on the
model's say-so, and a failed refutation NEVER weakens or short-circuits
the ordinary proof attempt — the pass can only (a) refute, with a
kernel-checked proof, or (b) stay silent.

Generality. Nothing in this module mentions any problem, benchmark or
mathematical domain. The goal is handled as an opaque proposition: the
statement is telescoped to a closed `∀`-proposition by
`goal_fingerprint.statement_to_prop`, negated, and handed back to the
same verifier every other part of the pipeline uses.

Method notes and where they come from
-------------------------------------
The strategy checklist in `REFUTE_SYSTEM` is distilled from the standard
counterexample-search literature rather than invented:

* Lakatos, *Proofs and Refutations* (1976) — counterexamples drive
  statement repair ("monster-barring", "lemma-incorporation"). ARM's
  revise-on-failure loop is already Lakatosian; a refutation pass makes
  it literal, and the failed-witness feedback is exactly
  lemma-incorporation.
* Claessen & Hughes, *QuickCheck* (ICFP 2000) — random property testing
  plus SHRINKING to a minimal counterexample. Hence the instruction to
  report the smallest/simplest witness found, not the first.
* Blanchette & Nipkow, *Nitpick* (ITP 2010) — counterexamples for HOL by
  finite model finding. Hence "try the smallest finite model first".
* Bulwahn, *Isabelle Quickcheck* narrowing (2012) — symbolic narrowing
  beats random sampling when inputs are constrained by hypotheses.
  Hence "solve the hypotheses first, then look for a violating instance"
  rather than sampling blindly.
* Johansson et al., *Hipster / QuickSpec* — theory exploration filters
  conjectures by testing BEFORE proving. That is precisely this pass's
  role relative to the sketcher.
* Ireland & Bundy, *productive use of failure*; Clarke et al., CEGAR
  (2000) — a counterexample is not just a verdict, it is the most
  informative feedback available for the next proposal.

Prompt design follows from the same sources plus the failure modes seen
in this repo's traces:

1. Explicit FRAME INVERSION. The policy has usually just been reasoning
   in a proving frame; the prompt says plainly that the task is now to
   refute.
2. An explicit "no counterexample" verdict. Without an approved way to
   decline, a model asked to refute will fabricate. This escape hatch is
   the main anti-hallucination device and it ends the search early
   instead of burning the remaining attempts (the same waste pattern the
   prove-step retry budget was built to stop).
3. The dominant LLM error for this task, stated as a hard rule: a witness
   must satisfy EVERY hypothesis. An "counterexample" that violates a
   hypothesis refutes nothing.
4. Lean-checkable output demanded, prose refused — the kernel is the
   oracle, so an unproved witness is worthless.
5. Failure feedback: each retry receives the previous attempt's Lean
   errors, so attempts are not independent samples.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

# Reject a "refutation" that leans on an escape hatch. Mirrors
# proof_dag._FORBIDDEN_TAC_RE; kept local so this module has no import
# cycle with proof_dag.
_FORBIDDEN_TAC_RE = re.compile(r"\b(?:sorry|admit|native_decide)\b")

#: Name given to the refutation theorem in the gate compile.
REFUTATION_DECL = "refutation_witness"


REFUTE_SYSTEM = """\
You are a proof assistant expert working in Lean 4 with Mathlib.

YOUR TASK IS TO REFUTE, NOT TO PROVE.

You are given a proposition P. Your job is to decide whether P is FALSE,
and if it is, to produce a Lean 4 proof of `¬ P`. You are NOT being asked
to prove P. Read the statement fresh, with the working assumption that it
may well be wrong.

WHY THIS MATTERS: a prover can spend hours failing to prove a statement
that is simply false. Finding the counterexample ends the problem
immediately and is a complete, valuable result.

THE ONE RULE YOU MUST NOT BREAK
A counterexample must satisfy EVERY hypothesis of the statement. An
instance that violates a hypothesis refutes nothing at all. This is the
single most common mistake on this task — before you answer, re-read each
hypothesis and confirm your witness satisfies it.

HONESTY REQUIREMENT
If you believe the statement is TRUE, say so: answer with
`"verdict": "no_counterexample"` and explain briefly why you think it
holds. That is a correct and useful answer. Do NOT invent a
counterexample you cannot prove — your proof is checked by the Lean
kernel and a bogus witness costs a multi-minute compile and tells us
nothing. A confident wrong answer is worse than an honest "I think this
is true".

WHERE COUNTEREXAMPLES HIDE — work through these in order:
1. DEGENERATE AND BOUNDARY INSTANCES. Empty sets, singletons, zero, one,
   the empty function, equal arguments, collinear/coincident points,
   identical elements. Statements are usually right in general position
   and wrong at the boundary.
2. SMALLEST FINITE MODELS FIRST. If the statement quantifies over a
   structure, try the smallest instances that satisfy the hypotheses
   before anything large.
3. STRICT vs NON-STRICT. `<` claimed where only `≤` holds is a classic;
   look for the equality case.
4. OFF-BY-ONE AND INDEX EDGES. n = 0, n = 1, the first index at which a
   recurrence or induction actually applies, empty ranges.
5. MISSING HYPOTHESES. Ask what the author forgot to assume — positivity,
   non-degeneracy, coprimality, non-triviality, injectivity.
6. VACUOUS OR EXTREME PARAMETERS. Values that make a hypothesis trivially
   true while breaking the conclusion.
7. CONSTRUCTED WITNESSES. If no small instance works, build one: define
   the sequence, polynomial, set or function that breaks the statement.
   You may introduce as many auxiliary definitions and lemmas as you need.

SHRINK YOUR ANSWER. If you find a counterexample, look for a smaller or
simpler one before answering. Minimal witnesses are far easier to prove
and far easier to check.

OUTPUT — reply with exactly this JSON object and NOTHING else:

{
  "verdict": "refuted" | "no_counterexample",
  "witness": "<one-line informal description of the counterexample, or why you think P is true>",
  "decls": ["<complete Lean declaration>", "..."],
  "proof": "<Lean 4 tactic block proving ¬ P, WITHOUT a leading `by`>",
  "rationale": "<brief: which hypothesis-satisfying instance breaks the conclusion, and why>"
}

RULES FOR THE LEAN OUTPUT
- `decls` holds any `def`/`lemma`/`theorem`/`structure` your witness
  needs, each a complete standalone declaration, in dependency order.
  Use `noncomputable def` where Lean requires it. Leave `decls` empty
  when the proof needs nothing extra.
- `proof` is a tactic block proving `¬ P` — it will be placed after
  `theorem {decl} : ¬ (P) := by`. Do not restate the theorem.
- NEVER use `sorry`, `admit`, or `native_decide`. A proof containing any
  of them is discarded outright.
- Everything you write is compiled by Lean against Mathlib. Prefer
  `decide`, `norm_num`, `omega` and explicit term proofs for concrete
  witnesses — they are reliable and fast.
- When `"verdict"` is `"no_counterexample"`, set `decls` to `[]` and
  `proof` to `""`.\
"""


#: Sent after a decline, before the search is allowed to end.
#:
#: Rationale from `lrs_refute_v1`: the policy declined on attempt 1 after
#: 104s — "standard locally zero constructions fail the order-five
#: dominant-root, simplicity, or nondegeneracy assumptions" — on a
#: statement that IS false. Its reasoning was sound (it correctly rejected
#: candidates for violating hypotheses) but its search was shallow: it
#: looked for a counterexample to FIND rather than one to BUILD. Ending the
#: search on that first decline spent 1 of 5 attempts. A decline now costs
#: one attempt and earns this push-back; only a SECOND CONSECUTIVE decline
#: ends the search, which keeps the escape hatch honest while actually
#: using the budget.
PUSHBACK_NOTE = """\
YOU DECLINED, AND THAT IS NOT YET ACCEPTED. Try harder before declining
again.

The most common reason a real counterexample is missed is looking for one
to FIND when the answer has to be BUILT. Small or standard instances
usually fail the hypotheses; that is evidence you need a construction, not
evidence the statement is true.

Work through these before answering:
- CONSTRUCT, don't search. Define the object — a sequence, function, set,
  polynomial, structure — whose properties make every hypothesis hold BY
  CONSTRUCTION, then check the conclusion fails.
- Build from a simpler object. Take something whose behaviour you control
  and transform it, so the hypotheses are inherited rather than verified
  case by case.
- Use a parametric family. A free parameter often lets you satisfy an
  awkward hypothesis and break the conclusion at the same time.
- Give the witness extra structure — symmetry, a closed form, a recurrence,
  a group action. Structure is what makes the hypotheses checkable at all.
- Re-read each hypothesis and ask what it is REALLY excluding. A hypothesis
  that looks restrictive is often satisfied by a whole family you have not
  considered.

If, after genuinely attempting to construct a witness, you still believe
the statement is true, decline again. An honest decline is an acceptable
answer; a fabricated counterexample is not.\
"""

#: Added from the second decline onward. Asks for a committed attempt and
#: lets the kernel adjudicate, rather than another refusal — a construction
#: Lean rejects is more informative than a repeated "I think it's true",
#: and costs one compile. The right to decline is preserved: this asks for
#: effort, never for a witness the model does not believe in.
ESCALATION_NOTE = """\
THIS IS DECLINE {n} OF AT MOST {m}.

Repeating that you cannot find one is no longer useful. Pick the single
most promising construction you considered and WRITE IT OUT in full — the
definitions, the witness, and a proof attempt — even if you are not
certain it works. Lean will adjudicate it, and a construction the kernel
rejects tells us far more than another refusal.

Do not invent a witness you believe is wrong: that wastes a compile and
teaches nothing. But if you have any candidate you have not yet committed
to paper, commit to it now. Only decline again if you have genuinely
exhausted the constructions you can think of.\
"""


def refute_user_prompt(prop: str,
                       failures: list[str] | None = None,
                       premises: list[str] | None = None,
                       declines: int = 0,
                       max_declines: int = 0) -> str:
    """Build the user turn: the proposition, prior failed attempts, and
    optional retrieved premise names.

    `declines` is how many times the model has already refused. 1 appends
    `PUSHBACK_NOTE`; 2 or more additionally appends `ESCALATION_NOTE`,
    which asks for a committed construction and lets the kernel judge it.
    """
    out = [
        "Decide whether the following Lean 4 proposition is FALSE, and if "
        "so prove its negation.",
        "",
        "PROPOSITION P:",
        "```lean",
        prop.strip(),
        "```",
    ]
    if premises:
        out += ["",
                "Mathlib declarations that may be relevant (names only; "
                "verify signatures yourself):",
                ", ".join(premises[:40])]
    if failures:
        out += ["",
                "PREVIOUS ATTEMPTS AT REFUTATION FAILED. Read the Lean "
                "errors and do not repeat the same idea — either fix the "
                "witness, choose a different one, or conclude the "
                "statement is true."]
        for i, f in enumerate(failures[-3:], 1):
            out += [f"", f"--- failed attempt {i} ---", f[:1200]]
    if declines >= 1:
        out += ["", PUSHBACK_NOTE]
        if declines >= 2:
            out += ["", ESCALATION_NOTE.format(n=declines,
                                               m=max_declines or declines)]
    else:
        out += ["",
                "Remember: your witness must satisfy EVERY hypothesis, and "
                "answering \"no_counterexample\" is a fully acceptable "
                "result if that is what you believe."]
    return "\n".join(out)


def _extract_json_object(text: str) -> dict | None:
    """First balanced top-level JSON object in the response, tolerating
    prose and code fences around it."""
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


def parse_refutation(raw: str) -> tuple[dict | None, str | None]:
    """Parse a model response into a normalized dict, or (None, reason).

    Normalized keys: verdict, witness, decls (list[str]), proof, rationale.
    """
    obj = _extract_json_object(raw or "")
    if obj is None:
        return None, "no JSON object in response"
    verdict = str(obj.get("verdict") or "").strip().lower()
    if verdict not in ("refuted", "no_counterexample"):
        return None, f"bad verdict: {verdict!r}"
    decls = obj.get("decls") or []
    if isinstance(decls, str):
        decls = [decls]
    if not isinstance(decls, list):
        return None, "decls is not a list"
    decls = [str(d) for d in decls if str(d).strip()]
    proof = str(obj.get("proof") or "")
    # A leading `by` would produce `by by` after assembly — the same slip
    # the ARM prove step had to guard against.
    proof = re.sub(r"^\s*by\b", "", proof).strip()
    if verdict == "refuted" and not proof:
        return None, "verdict=refuted but no proof"
    return ({"verdict": verdict,
             "witness": str(obj.get("witness") or ""),
             "decls": decls,
             "proof": proof,
             "rationale": str(obj.get("rationale") or "")}, None)


def build_refutation_header(prelude: str, decls: list[str], prop: str,
                            *, decl_name: str = REFUTATION_DECL) -> str:
    """`prelude + decls + "theorem <name> : ¬ (prop)"` — the header handed
    to the verifier. The caller appends `:= by <proof>`."""
    parts = [prelude.rstrip("\n")] if prelude.strip() else []
    parts += [d.strip() for d in decls if d.strip()]
    parts.append(f"theorem {decl_name} : ¬ ({prop.strip()})")
    return "\n\n".join(parts)


@dataclass(slots=True)
class RefutationResult:
    """Outcome of the pre-flight pass.

    `refuted` is True ONLY when Lean accepted a proof of `¬ P`. Everything
    else — model declined, unparseable, forbidden tactic, compile failure,
    LLM error — leaves it False and the pipeline proceeds exactly as if
    this pass had not run.
    """
    refuted: bool = False
    proof: str | None = None
    decls: list[str] = field(default_factory=list)
    witness: str = ""
    rationale: str = ""
    attempts_used: int = 0
    #: One entry per attempt: "attempt N: <what happened>".
    log: list[str] = field(default_factory=list)
    #: How many times the model declined to produce a counterexample. A
    #: single decline earns a push-back; two consecutive end the search.
    declines: int = 0
    #: The full `theorem … : ¬ (P) := by …` text, when refuted.
    refutation_source: str | None = None

    def to_dict(self) -> dict:
        return {
            "refuted": self.refuted,
            "witness": self.witness,
            "rationale": self.rationale,
            "attempts_used": self.attempts_used,
            "log": list(self.log),
            "declines": self.declines,
            "decls": list(self.decls),
            "refutation_source": self.refutation_source,
        }


def attempt_refutation(
    prop: str,
    *,
    llm_call: Callable[[str, str], str],
    verify_fn: Callable[[str, str], dict],
    prelude: str = "",
    attempts: int = 5,
    max_declines: int | None = None,
    premises: list[str] | None = None,
    trace: Callable[..., None] | None = None,
) -> RefutationResult:
    """Try up to `attempts` times to obtain a KERNEL-VERIFIED proof of
    `¬ prop`.

    `llm_call(system, user) -> raw_text` and
    `verify_fn(header, proof_block) -> {"ok", "errors", ...}` are the same
    callables the rest of the pipeline uses, so this module needs no LLM
    or Lean access of its own.

    A `no_counterexample` verdict costs one attempt and earns escalating
    push-back rather than ending the search: `PUSHBACK_NOTE` on the first
    decline (search → construct), plus `ESCALATION_NOTE` from the second
    (commit to a construction and let the kernel adjudicate).

    `max_declines` caps how many refusals are tolerated before the search
    stops; None means the full `attempts` budget, i.e. a reluctant model
    never short-circuits it. Lowering it trades coverage for wall time: a
    decline costs one LLM call, whereas a committed-but-wrong construction
    costs a full compile.

    The right to decline is never removed — a model with no approved way
    to refuse will fabricate, and fabrications cost compiles. History: the
    first version stopped on the first decline and used 1 of 5 attempts on
    `lrs_refute_v1`, a statement that is in fact false.
    """
    tr = trace or (lambda kind, **kw: None)
    res = RefutationResult()
    failures: list[str] = []
    budget = max(int(attempts), 0)
    cap = budget if max_declines is None else max(int(max_declines), 1)

    for i in range(1, budget + 1):
        res.attempts_used = i
        try:
            raw = llm_call(REFUTE_SYSTEM,
                           refute_user_prompt(prop, failures, premises,
                                              declines=res.declines,
                                              max_declines=cap))
        except Exception as e:                       # LLM/transport error
            res.log.append(f"attempt {i}: llm error: {type(e).__name__}")
            tr("refute", attempt=i, stage="llm_error", detail=str(e)[:200])
            break

        parsed, perr = parse_refutation(raw)
        if parsed is None:
            res.log.append(f"attempt {i}: unparseable ({perr})")
            tr("refute", attempt=i, stage="parse_failed", detail=perr)
            failures.append(f"Your reply could not be parsed: {perr}")
            continue

        if parsed["verdict"] == "no_counterexample":
            res.witness = parsed["witness"]
            res.rationale = parsed["rationale"]
            res.declines += 1
            if res.declines >= cap:
                res.log.append(
                    f"attempt {i}: declined ({res.declines}/{cap}); "
                    f"decline budget exhausted, search ends")
                tr("refute", attempt=i, stage="no_counterexample_final",
                   declines=res.declines, detail=parsed["witness"][:200])
                break
            res.log.append(
                f"attempt {i}: declined ({res.declines}/{cap}); pushing "
                f"back for a CONSTRUCTED witness")
            tr("refute", attempt=i, stage="no_counterexample_pushback",
               declines=res.declines, detail=parsed["witness"][:200])
            continue

        proof = parsed["proof"]
        if _FORBIDDEN_TAC_RE.search(proof) or any(
                _FORBIDDEN_TAC_RE.search(d) for d in parsed["decls"]):
            res.log.append(f"attempt {i}: forbidden tactic in refutation")
            tr("refute", attempt=i, stage="forbidden_tactic")
            failures.append(
                "Your refutation used `sorry`/`admit`/`native_decide`, "
                "which is never accepted. Produce a real proof or answer "
                "no_counterexample.")
            continue

        header = build_refutation_header(prelude, parsed["decls"], prop)
        body = "  " + proof.replace("\n", "\n  ")
        tr("refute", attempt=i, stage="gate", detail=parsed["witness"][:200])
        try:
            gate = verify_fn(header, body)
        except Exception as e:                       # verifier blew up
            res.log.append(f"attempt {i}: verifier error: {type(e).__name__}")
            tr("refute", attempt=i, stage="verify_error", detail=str(e)[:200])
            break

        if gate.get("ok"):
            res.refuted = True
            res.proof = proof
            res.decls = parsed["decls"]
            res.witness = parsed["witness"]
            res.rationale = parsed["rationale"]
            res.refutation_source = f"{header} := by\n{body}\n"
            res.log.append(f"attempt {i}: REFUTED (kernel-verified)")
            tr("refute", attempt=i, stage="REFUTED",
               detail=parsed["witness"][:200])
            return res

        err = (gate.get("errors") or "?")[:1500]
        res.log.append(f"attempt {i}: kernel rejected the refutation")
        tr("refute", attempt=i, stage="rejected", detail=err[:300])
        failures.append(
            f"Claimed counterexample: {parsed['witness']}\n"
            f"Lean rejected your proof of ¬P:\n{err}")

    return res
