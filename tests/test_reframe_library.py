"""Stage 3: a reframing builds a support library for its new objects.

A reframing introduces objects that NOTHING is yet known about, so its
bridges can be unprovable purely for lack of elementary facts — not
because the frame was wrong. `--reframe-build-library` constructs those
facts first, each individually kernel-verified.

The properties that make it safe and affordable, pinned here:

* OFF by default and OFF means untouched — no library call, no compile,
  and the reframe proceeds exactly as it would have;
* results are split BY KIND. `defs` are spliced verbatim and matched by a
  regex accepting only def/abbrev/instance/structure/inductive, so a
  proved LEMMA placed there is SILENTLY DROPPED. Library lemmas therefore
  come back as claims;
* a proved library lemma is registered in the proved-lemma bank, so the
  prove step reuses it for ZERO compiles — otherwise the library would be
  paid for twice;
* the library verifier compiles a COMPLETE source. It must not go through
  `verify_proof`, which appends `:= by <body>` and would turn every
  finished declaration into a syntax error;
* any failure returns nothing rather than breaking the reframe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.library import BUILD_SYSTEM, PLAN_SYSTEM            # noqa: E402
from search.proof_dag import (                                  # noqa: E402
    PROVE_LEMMA_SYSTEM, THEORY_SYSTEM, attempt_dag_proof,
)
from search.reframe import REFRAME_SYSTEM, theory_as_proposal, Theory  # noqa: E402

# A deliberately heavy goal: the leverage gate compares claim difficulty
# against the TARGET's, so the target must be the harder thing.
HEADER = ("theorem t (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) "
          ": (a^2 + b^2 + c^2 + d^2) * (a + b + c + d) ≥ "
          "4 * (a*b*c*d) ^ (1/2) * (a + b + c)")

# The stuck leaf must be GENUINELY HARD. The leverage gate compares each
# claim against the target's difficulty, so against an easy leaf every
# reframing is correctly rejected as moving the difficulty rather than
# reducing it — you do not reframe a goal that is already easy. Measured:
# this scores 38.25, `a^2 + b^2 ≥ 2*a*b` scores 0.85, and the reframe's
# claim scores 2.90.
HARD_LEAF = ("∀ ε > 0, ∃ N, ∀ n ≥ N, |(∑ i ∈ Finset.range n, f i / (i+1)^2) "
             "- L| < ε ∧ (∏ j ∈ Finset.range n, (1 + a j)) "
             "≤ Real.exp (∑ j ∈ Finset.range n, a j)")

SKETCH = json.dumps({
    "setup": [],
    "haves": [{"id": "h1", "type": HARD_LEAF,
               "tactic": "positivity", "depends": []}],
    "closer": "exact h1",
})

# Never admissible: restates nothing useful and never proves.
BAD_THEORY = json.dumps({
    "defs": [],
    "lemmas": [{"name": "aux_no", "statement": "lemma aux_no : True"}],
    "leaf_tactics": {"h1": "trivial"},
})

REFRAME = json.dumps({
    "name": "Frame",
    "rationale": "work in the new domain",
    "objects": ["def W (x : ℝ) : ℝ := x"],
    "bridges": ["lemma w_ok (x : ℝ) : W x = x"],
    "theorems": [],
    "derivation": "simpa using w_ok a",
})

LIB_PLAN = json.dumps({"decls": [
    {"name": "w_def", "kind": "def",
     "statement": "def w_twice (x : ℝ) : ℝ := x + x", "rationale": "r"},
    {"name": "w_lem", "kind": "lemma",
     "statement": "lemma w_self (x : ℝ) : W x = x", "rationale": "r"},
]})
LIB_DEF = json.dumps({"declaration": "def w_twice (x : ℝ) : ℝ := x + x"})
LIB_LEM = json.dumps({"declaration": "lemma w_self (x : ℝ) : W x = x := rfl"})

TIMEOUT_ERR = "f.lean:5:4: error: timeout: heartbeat exceeded"


def _fixture(lib_replies=(LIB_DEF, LIB_LEM), lib_ok=True):
    c = {"theory": 0, "reframe": 0, "plan": 0, "build": 0,
         "prove": 0, "probe": 0, "lib_compile": 0,
         "proved_stmts": []}

    def llm(system, user):
        if system == THEORY_SYSTEM:
            c["theory"] += 1
            return BAD_THEORY
        if system == REFRAME_SYSTEM:
            c["reframe"] += 1
            return REFRAME
        if system == PLAN_SYSTEM:
            c["plan"] += 1
            return LIB_PLAN
        if system == BUILD_SYSTEM:
            r = lib_replies[min(c["build"], len(lib_replies) - 1)]
            c["build"] += 1
            return r
        if system == PROVE_LEMMA_SYSTEM:
            c["prove"] += 1
            c["proved_stmts"].append(user)
            return json.dumps({"proof": "GOODPROOF"})
        return SKETCH

    def verify(header, body):
        if "theorem t" not in header:            # standalone lemma gate
            ok = "GOODPROOF" in body
            return {"ok": ok, "errors": None if ok else "e: nope",
                    "body_line_offset": 1}
        ok = "w_ok" in header and ":= by sorry" not in header
        return {"ok": ok, "errors": None if ok else TIMEOUT_ERR,
                "body_line_offset": 3}

    def probe(header, body):
        # Only the REFRAMED theory satisfies the leaf. Without this the
        # in-language theory commits on round 0 and the reframe round is
        # never reached — which is correct behaviour, just not what these
        # tests are about.
        c["probe"] += 1
        ok = "w_ok" in header
        return {"ok": ok,
                "errors": None if ok else "f.lean:1:0: error: unsolved goals",
                "body_line_offset": 3}

    def lib_compile(source):
        c["lib_compile"] += 1
        return {"ok": lib_ok, "errors": None if lib_ok else "e: nope"}

    return llm, verify, probe, lib_compile, c


def _run(llm, verify, probe, lib_compile, *, build_library: bool,
         reframe: bool = True):
    return attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify,
        sketch_attempts=1, repair_rounds=3,
        leaf_fallbacks=(), decompose_depth=0,
        abduce_lemmas=True, abduce_mode="theory",
        probe_fn=probe, abduce_theory_rounds=1,
        abduce_theory_trigger="always",
        reframe_on_abandon=reframe,
        reframe_build_library=build_library,
        reframe_library_decls=2,
        reframe_library_attempts=1,
        library_compile_fn=lib_compile,
    )


# ---- the flag is genuinely off by default ----------------------------------

def test_library_is_not_built_when_the_flag_is_off():
    llm, verify, probe, lib, c = _fixture()
    _run(llm, verify, probe, lib, build_library=False)
    assert c["reframe"] >= 1, "reframe itself should still run"
    assert c["plan"] == 0 and c["build"] == 0
    assert c["lib_compile"] == 0


def test_no_library_call_when_reframe_itself_is_off():
    llm, verify, probe, lib, c = _fixture()
    _run(llm, verify, probe, lib, build_library=True, reframe=False)
    assert c["reframe"] == 0
    assert c["lib_compile"] == 0


# ---- the library is actually built and used --------------------------------

def test_library_is_built_after_a_sufficient_reframe():
    llm, verify, probe, lib, c = _fixture()
    _run(llm, verify, probe, lib, build_library=True)
    assert c["reframe"] >= 1
    assert c["plan"] == 1, "the library planner should be asked once"
    assert c["lib_compile"] >= 1, "declarations must reach the kernel"


def test_library_runs_only_after_in_language_rounds_failed():
    """Abduction is cheaper; building a domain is the escalation."""
    llm, verify, probe, lib, c = _fixture()
    _run(llm, verify, probe, lib, build_library=True)
    assert c["theory"] >= 1
    assert c["plan"] <= 1


def test_a_proved_library_lemma_is_not_proved_again():
    """The bank hit is what stops the library being paid for twice."""
    llm, verify, probe, lib, c = _fixture()
    _run(llm, verify, probe, lib, build_library=True)
    asked = "\n".join(c["proved_stmts"])
    assert "w_self" not in asked, (
        "w_self was kernel-verified by the library; re-proving it wastes "
        "the compile the library already spent")


# ---- failures are survivable ------------------------------------------------

def test_library_compile_failure_does_not_break_the_reframe():
    llm, verify, probe, lib, c = _fixture(lib_ok=False)
    _run(llm, verify, probe, lib, build_library=True)
    assert c["reframe"] >= 1                 # the reframe still happened


def test_library_exception_does_not_break_the_reframe():
    llm, verify, probe, _lib, c = _fixture()

    def boom(_source):
        raise RuntimeError("lake died")
    _run(llm, verify, probe, boom, build_library=True)
    assert c["reframe"] >= 1


def test_missing_compile_fn_disables_the_library():
    """The flag alone must not be enough — without a compiler there is
    nothing to gate declarations with."""
    llm, verify, probe, _lib, c = _fixture()
    attempt_dag_proof(
        HEADER, sketch_llm_call=llm, verify_fn=verify,
        sketch_attempts=1, repair_rounds=3, leaf_fallbacks=(),
        decompose_depth=0, abduce_lemmas=True, abduce_mode="theory",
        probe_fn=probe, abduce_theory_rounds=1,
        abduce_theory_trigger="always", reframe_on_abandon=True,
        reframe_build_library=True, library_compile_fn=None,
    )
    assert c["plan"] == 0 and c["build"] == 0


# ---- kind splitting ---------------------------------------------------------

def test_proved_lemmas_must_not_be_routed_into_defs():
    """`_parse_theory`'s def regex accepts only
    def/abbrev/instance/structure/inductive — a lemma put there vanishes
    without an error, and its claims would then reference nothing."""
    import search.proof_dag as pd
    raw = theory_as_proposal(
        Theory(objects=["def F : ℕ := 0"], bridges=["lemma b : True"],
               derivation="trivial"),
        ["h1"],
        extra_defs=["lemma sneaky : True := trivial"],
        extra_claims=["lemma good (n : ℕ) : n + 0 = n"])
    parsed, err = pd._parse_theory(raw)
    assert err is None
    defs, lemmas, _ = parsed
    assert "lemma sneaky : True := trivial" not in defs   # dropped, as warned
    assert "good" in [n for n, _ in lemmas]               # claims survive
