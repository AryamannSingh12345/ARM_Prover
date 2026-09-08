"""The bare-model baseline must measure the MODEL, not the harness.

Two defects motivated `--response-mode verbatim`, both found by reading the
code rather than a run:

1. `extract_proof` RE-INDENTS column-0 lines. An unindented line under `by`
   is a Lean syntax error, so the harness was silently repairing the model's
   proof into compiling shape and scoring the repair.
2. The default whole-proof prompt requires one tactic per line and asks the
   model to PREFER one-liner closers — which forbids the `have … := by`
   structure a competition proof is built from.

A third risk is created BY the fix: once the model writes the whole file it
writes the statement too, and the kernel will faithfully verify a weakened
theorem. `statement_is_verbatim` is the gate for that, and it must refuse
before anything is compiled.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.run_minif2f import (  # noqa: E402
    build_prompt, evaluate_response, extract_proof, statement_is_verbatim,
    unwrap_fence,
)
from policy.prompts import (  # noqa: E402
    WHOLE_FILE_SYSTEM_VERBATIM, WHOLE_PROOF_SYSTEM, WHOLE_PROOF_SYSTEM_BARE,
)

HEADER = (
    "theorem putnam_2001_a1\n"
    "(S : Type*)\n"
    "[Mul S]\n"
    "(hS : ∀ a b : S, (a * b) * a = b)\n"
    ": ∀ a b : S, a * (b * a) = b"
)

FILE = (
    "import Mathlib\n"
    f"{HEADER} := by\n"
    "intro a b\n"
    "exact hS a b\n"
)


def _args(**kw):
    base = dict(response_mode="verbatim", prompt_style="bare",
                verify_timeout=60)
    base.update(kw)
    return types.SimpleNamespace(**base)


# --------------------------------------------------------------- unwrap

def test_fence_unwrap_is_the_only_edit():
    """Indentation the model chose must survive, including column 0."""
    out = unwrap_fence(f"```lean4\n{FILE}```")
    assert out.strip() == FILE.strip()
    assert "\nintro a b" in out, "column-0 line was re-indented"


def test_unwrap_leaves_unfenced_text_alone():
    assert unwrap_fence(FILE).strip() == FILE.strip()


def test_prose_around_a_fence_is_not_salvaged():
    """The instruction was to reply with the file and nothing else. Rescuing
    a fenced block out of surrounding prose would be editing the response;
    the compile failing is the correct outcome."""
    messy = f"Here you go:\n\n```lean4\n{FILE}```\n\nHope that helps!"
    assert unwrap_fence(messy).startswith("Here you go")


def test_extract_proof_still_reindents_documenting_why_verbatim_exists():
    """Guards the claim in this module's docstring. If `extract_proof` ever
    stops repairing indentation this test should be updated, not deleted —
    the baseline still must not depend on it."""
    got = extract_proof("```lean4\ntheorem foo : True := by\nexact trivial\n```")
    assert got == "  exact trivial"


# ------------------------------------------------------------- fidelity

def test_faithful_reproduction_passes():
    assert statement_is_verbatim(HEADER, FILE)


def test_reflowed_whitespace_passes():
    """Line breaks are formatting, not meaning."""
    assert statement_is_verbatim(HEADER, FILE.replace("\n(S : Type*)\n",
                                                      " (S : Type*) "))


@pytest.mark.parametrize("mutation,label", [
    ((": ∀ a b : S, a * (b * a) = b := by", ": True := by"),
     "weakened conclusion"),
    (("(hS : ∀ a b : S, (a * b) * a = b)",
      "(hS : ∀ a b : S, a * b = b * a)"), "swapped hypothesis"),
    (("[Mul S]\n", ""), "dropped binder"),
    (("theorem putnam_2001_a1", "theorem putnam_2001_a1'"), "renamed"),
])
def test_altered_statement_is_refused(mutation, label):
    old, new = mutation
    assert not statement_is_verbatim(HEADER, FILE.replace(old, new)), label


def test_empty_response_is_refused():
    assert not statement_is_verbatim(HEADER, "")


# ------------------------------------------------------------ evaluate

def test_statement_mismatch_is_refused_without_compiling(monkeypatch):
    """The gate must cost ZERO compiles: a mismatched file is rejected on
    text, never handed to Lean."""
    called = []
    monkeypatch.setattr("eval.run_minif2f.compile_lean",
                        lambda *a, **k: called.append(1))
    bad = FILE.replace(": ∀ a b : S, a * (b * a) = b := by", ": True := by")
    artifact, ok, errors = evaluate_response(bad, HEADER, "import Mathlib",
                                             _args())
    assert ok is False
    assert "statement_mismatch" in errors
    assert not called, "a mismatched statement must not reach the compiler"


def test_verbatim_compiles_the_model_text_unchanged(monkeypatch):
    seen = {}

    def fake(source, **kw):
        seen["source"] = source
        return types.SimpleNamespace(ok=True, errors=None)

    monkeypatch.setattr("eval.run_minif2f.compile_lean", fake)
    artifact, ok, _ = evaluate_response(f"```lean4\n{FILE}```", HEADER,
                                        "import Mathlib", _args())
    assert ok is True
    assert seen["source"].strip() == FILE.strip()
    assert "\nintro a b" in seen["source"], "harness altered the model's file"


def test_body_mode_still_assembles_under_our_header(monkeypatch):
    """The legacy path must be untouched — existing rows stay comparable."""
    seen = {}

    def fake(header, proof, **kw):
        seen["header"], seen["proof"] = header, proof
        return types.SimpleNamespace(ok=False, errors="boom")

    monkeypatch.setattr("eval.run_minif2f.verify_proof", fake)
    _, ok, errors = evaluate_response("  exact hS a b", HEADER,
                                      "import Mathlib",
                                      _args(response_mode="body"))
    assert ok is False and errors == "boom"
    assert seen["header"] == HEADER


# -------------------------------------------------------------- prompts

def test_bare_prompt_carries_no_strategic_guidance():
    """`legacy` steers toward one-liner closers and one tactic per line.
    Measuring a bare model through that prompt measures the prompt."""
    for banned in ("one-liner", "omega", "nlinarith", "aesop",
                   "single-tactic"):
        assert banned not in WHOLE_PROOF_SYSTEM_BARE
        assert banned not in WHOLE_FILE_SYSTEM_VERBATIM
    assert "one-liner" in WHOLE_PROOF_SYSTEM, "legacy prompt changed"


def test_bare_prompt_permits_structured_proofs():
    assert "have" in WHOLE_PROOF_SYSTEM_BARE
    assert "structured" in WHOLE_FILE_SYSTEM_VERBATIM


def test_verbatim_prompt_forbids_restating_the_theorem():
    low = WHOLE_FILE_SYSTEM_VERBATIM.lower()
    assert "weaken" in low and "exactly" in low


def test_user_prompt_gives_only_the_statement():
    """No premises, no lemma hints, no import list, no proof shape."""
    p = build_prompt(HEADER, "bare")
    assert HEADER in p
    for leak in ("premise", "hint", "lemma:", "try "):
        assert leak not in p.lower()


def test_legacy_is_still_the_default_for_both_switches():
    """Every existing row must stay comparable; the new paths are opt-in."""
    import argparse
    import eval.run_minif2f as m

    src = Path(m.__file__).read_text(encoding="utf-8")
    assert '"--response-mode", choices=("body", "verbatim"),\n' \
           '                    default="body"' in src
    assert '"--prompt-style", choices=("legacy", "bare"),\n' \
           '                    default="legacy"' in src
    assert isinstance(argparse.ArgumentParser(), argparse.ArgumentParser)
