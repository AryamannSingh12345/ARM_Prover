"""Every `first | …` alternative must be parenthesised.

Root cause of the dominant failure in putnam_easy8_sol_v1/v2. The LLM
ladder generator (`--leaf-fallback-mode llm`, the default) proposed the
entry `interval_cases a <;> norm_num at *`. `_wrap_with_fallback`
parenthesised the model's own tactic but spliced fallbacks in bare, so the
assembled block read

    first | (…) | norm_num | … | interval_cases a <;> norm_num at * | decide

where `<;>` swallows the trailing `| decide` and the block fails to parse.
Lean reports `unknown tactic` — pointing at the ladder, not at the entry
that broke it — which is why it took a dumped .lean file to find (the
temp file is deleted after each compile and the trace holds the model's
response, not the assembled source).

Only the dumped source settles this class of bug, so `--debug-dump-lean`
and this guard belong together.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.dag.assembly import _wrap_with_fallback          # noqa: E402
from search.proof_dag import (                                # noqa: E402
    HaveNode, Sketch, assemble_proof_with_map,
)

#: The exact ladder the runner produced on putnam_1974_a1.
LIVE_LADDER = ("norm_num", "omega", "aesop", "simp_all [hconspiratorial]",
               "interval_cases a <;> norm_num at *", "decide")


def _alts(block: str) -> list[str]:
    assert block.startswith("first | ")
    return [a.strip() for a in block[len("first | "):].split(" | ")]


def test_every_alternative_is_parenthesised():
    out = _wrap_with_fallback("simp", LIVE_LADDER)
    for a in _alts(out):
        assert a.startswith("(") and a.endswith(")"), a


def test_combinator_entry_cannot_swallow_the_next_alternative():
    """The specific break: `<;>` followed by an unparenthesised `| decide`."""
    out = _wrap_with_fallback("simp", LIVE_LADDER)
    assert "<;> norm_num at * | decide" not in out
    assert "(interval_cases a <;> norm_num at *)" in out


def test_empty_tactic_path_is_also_parenthesised():
    """Blueprint mode leaves tactics empty, so the ladder stands alone —
    that path had the same defect."""
    out = _wrap_with_fallback("", LIVE_LADDER)
    for a in _alts(out):
        assert a.startswith("(") and a.endswith(")"), a


def test_semicolon_entry_is_contained():
    out = _wrap_with_fallback("simp", ("constructor; omega", "decide"))
    assert "(constructor; omega)" in out
    assert "; omega | decide" not in out


def test_no_fallbacks_returns_tactic_verbatim():
    assert _wrap_with_fallback("simp [foo]", ()) == "simp [foo]"


def test_assembled_body_contains_no_bare_combinator_alternative():
    """End-to-end through the assembler."""
    sk = Sketch(haves=[HaveNode("h1", "True", "trivial", [])],
                closer="exact h1")
    body, _ = assemble_proof_with_map(sk, leaf_fallbacks=LIVE_LADDER)
    for line in body.splitlines():
        if "first |" not in line:
            continue
        for a in _alts(line.strip()):
            assert a.startswith("("), (line, a)
