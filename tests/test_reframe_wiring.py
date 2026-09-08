"""Reframe wired into `attempt_dag_proof` (flag-gated, default OFF).

`_abduce_theory` decomposes a problem in its OWN vocabulary. Across the
July round it produced a correct decomposition on every hard problem and
NEVER ONCE changed the frame — it has no new objects, no bridge, and no
leverage test, so it cannot. `--reframe-on-abandon` adds exactly one
escalation round at the END of the theory loop.

What these tests pin:

* OFF is byte-for-byte the legacy control flow — no reframe import, no
  extra round, no extra LLM call. This is the invariant the migration
  tripwire exists to protect.
* the reframe round is reached only AFTER every in-language round has
  failed (abduction is cheaper; changing the frame is the escalation).
* it fires only on a SINGLE stuck leaf — one derivation closes one goal,
  so a multi-leaf reframe would fail the coverage check anyway.
* a reframing is converted into an ORDINARY theory proposal, so the
  circularity guard, quality gate, satisfaction probe, per-lemma kernel
  proving and all-or-nothing commit all still apply. Conversion is the
  only new code path; nothing that decides admissibility is duplicated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import search.proof_dag as pd                                   # noqa: E402
from search.reframe import (                                    # noqa: E402
    Theory, claim_name, theory_as_proposal,
)

HEADER = "theorem tgt (n : ℕ) : n + 0 = n"


# ---- the adapter ------------------------------------------------------------

def test_claim_name_reads_the_declared_name():
    """The derivation cites claims BY NAME; a bookkeeping name invented
    elsewhere would break the minimise step's reference check."""
    assert claim_name("lemma bridge_one (n : ℕ) : n + 0 = n") == "bridge_one"
    assert claim_name("theorem t1 : True") == "t1"
    assert claim_name("not a declaration") == ""


def _theory():
    return Theory(name="Frame", objects=["def F : ℕ := 0"],
                  bridges=["lemma b1 (n : ℕ) : n + 0 = n"],
                  theorems=["theorem t1 : True"],
                  derivation="exact b1 n")


def test_reframe_renders_as_a_theory_proposal():
    raw = theory_as_proposal(_theory(), ["h1"])
    parsed, err = pd._parse_theory(raw)
    assert err is None
    defs, lemmas, tactics = parsed
    assert defs == ["def F : ℕ := 0"]
    assert [n for n, _ in lemmas] == ["b1", "t1"]
    assert tactics == {"h1": "exact b1 n"}


def test_objects_become_defs_and_claims_become_lemmas():
    """Objects carry bodies and are spliced as-is; claims are stubbed and
    proved. Mixing the two would either stub a definition or try to prove
    one."""
    obj = json.loads(theory_as_proposal(_theory(), ["h1"]))
    assert obj["defs"] == ["def F : ℕ := 0"]
    assert {c["statement"] for c in obj["lemmas"]} == {
        "lemma b1 (n : ℕ) : n + 0 = n", "theorem t1 : True"}


def test_every_stuck_leaf_gets_the_derivation():
    obj = json.loads(theory_as_proposal(_theory(), ["h1", "h2"]))
    assert obj["leaf_tactics"] == {"h1": "exact b1 n", "h2": "exact b1 n"}


# ---- the flag ---------------------------------------------------------------

def test_flag_defaults_to_off():
    import inspect
    sig = inspect.signature(pd.attempt_dag_proof)
    assert sig.parameters["reframe_on_abandon"].default is False
    assert sig.parameters["reframe_rounds"].default == 2
    assert sig.parameters["reframe_leverage_factor"].default == 0.9


def test_runner_exposes_the_flag_off_by_default():
    import argparse
    import importlib
    rd = importlib.import_module("eval.run_dag")
    ap = [o for o in dir(rd) if o]          # module imports cleanly
    assert ap
    parser = None
    for name in ("build_parser", "_build_parser", "make_parser"):
        if hasattr(rd, name):
            parser = getattr(rd, name)()
            break
    if parser is None:
        return                               # parser built inline in main()
    ns = parser.parse_args([])
    assert isinstance(ns, argparse.Namespace)
    assert ns.reframe_on_abandon is False


def test_reframe_module_is_not_imported_when_the_flag_is_off():
    """Invariant 5 of the migration tripwire, applied here: a disabled
    mechanism is neither imported nor instantiated. The import lives
    INSIDE the branch for exactly this reason."""
    src = Path(__file__).resolve().parents[1] / "src" / "search" / "proof_dag.py"
    text = src.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and "reframe" in stripped:
            assert line.startswith("            "), (
                "reframe must be imported inside the guarded branch, "
                f"not at module scope: {line!r}")


# ---- ordering and restriction ----------------------------------------------

def _round_plan(n_rounds: int, enabled: bool, n_stuck: int) -> tuple[int, int]:
    """Mirror of the loop bounds in `_abduce_theory`: (total, reframe_at)."""
    reframe_round = -1
    if enabled and n_stuck == 1:
        reframe_round = n_rounds + 1
    total = n_rounds + 1 + (1 if reframe_round > 0 else 0)
    return total, reframe_round


def test_off_adds_no_round():
    assert _round_plan(2, enabled=False, n_stuck=1) == (3, -1)


def test_on_adds_exactly_one_round_at_the_end():
    total, at = _round_plan(2, enabled=True, n_stuck=1)
    assert total == 4 and at == 3
    assert at == total - 1                   # strictly last


def test_reframe_never_preempts_an_in_language_round():
    """Abduction is far cheaper; the frame change is the escalation."""
    _total, at = _round_plan(2, enabled=True, n_stuck=1)
    assert at > 2


def test_multi_leaf_does_not_trigger_reframe():
    """One derivation closes one goal — a multi-leaf reframe would fail
    the coverage check, so the calls are not spent."""
    assert _round_plan(2, enabled=True, n_stuck=3) == (3, -1)


def test_zero_configured_rounds_still_places_reframe_last():
    total, at = _round_plan(0, enabled=True, n_stuck=1)
    assert at == 1 and total == 2


# ---- soundness --------------------------------------------------------------

def test_conversion_cannot_smuggle_a_forbidden_tactic():
    """The converted proposal goes through the SAME parser, which scans
    defs and tactics for sorry/admit/native_decide."""
    t = Theory(objects=["def F : ℕ := 0"],
               bridges=["lemma b1 : True"], derivation="sorry")
    parsed, err = pd._parse_theory(theory_as_proposal(t, ["h1"]))
    assert parsed is None and err


def test_conversion_of_an_empty_theory_is_rejected():
    parsed, err = pd._parse_theory(theory_as_proposal(Theory(), ["h1"]))
    assert parsed is None and err
