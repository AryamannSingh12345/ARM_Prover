"""The confirmation vocabulary must be able to express a real citation.

Regression for `putnam_1986_b3` (run `p1986b3_v1`, 2026-08-09). The stuck
leaf was

    ∀ a b : Polynomial ℤ, ∀ m : ℤ, cong a b m ↔ Polynomial.C m ∣ a - b

and the pinned Mathlib already carries it, as

    Polynomial.C_dvd_iff_dvd_coeff : C r ∣ φ ↔ ∀ i, r ∣ φ.coeff i

Retrieval RANKED that lemma — it appears in the trace's candidate list at
858 s. Confirmation still returned `not_confirmed`, and the run went on to
spend ~1434 s of its 2738 s abducing a `noncomputable def aux_coeffQuotient`
plus two lemmas to re-derive it.

Confirmation was not wrong: given only zero-step citation forms, no
expressible tactic closed that leaf. Two things block a bare citation, and
both are structural to PutnamBench rather than particular to this problem:

  1. the leaf goal is `∀`-quantified, so `rw` cannot reach under the
     binders and `exact` would have to match the whole quantified form;
  2. `cong` is a HYPOTHESIS BINDER, not a definition, so nothing in
     Mathlib can mention it until `hcong` has been used.

`intros; simp only [*, {n}]` clears both generically — never naming the
hypothesis, which recognition cannot know — and folds in the orientation
that would otherwise need a separate `.symm` form.

Measured on this leaf: all four zero-step forms FAIL, the new form PASSES,
and full `simp` in its place FAILS (it normalises past the target).

One compile. Marked `live` accordingly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backend.compile_verify import compile_lean                  # noqa: E402
from search.recognize import Candidate, confirm_tactic            # noqa: E402

pytestmark = pytest.mark.live

LEAN_ROOT = Path(__file__).resolve().parents[1] / "lean"

IMPORTS = (
    "import Mathlib.Algebra.Polynomial.Coeff\n"
    "import Mathlib.Data.Nat.Prime.Basic\n"
    "import Mathlib.Data.Int.ModEq\n"
    "import Mathlib.Data.Int.GCD\n"
    "import Mathlib.Tactic\n"
)

#: `make_leaf_theorem` output for the h_cd leaf: the theorem's own binders
#: (so `hcong` is in scope, exactly as the runner poses it) + the leaf goal.
H_CD_LEAF = """example
    (n p : ℕ) (nppos : n > 0 ∧ p > 0) (pprime : Nat.Prime p)
    (cong : Polynomial ℤ → Polynomial ℤ → ℤ → Prop)
    (hcong : ∀ f g m, cong f g m ↔ ∀ i : ℕ, m ∣ (f - g).coeff i)
    (f g h r s : Polynomial ℤ)
    (hcoprime : cong (r * f + s * g) 1 p) (hprod : cong (f * g) h p) :
    ∀ a b : Polynomial ℤ, ∀ m : ℤ, cong a b m ↔ Polynomial.C m ∣ a - b := by
"""

#: Ranked alongside the real lemma, so the tactic under test is the one the
#: pipeline would actually build — not a one-candidate special case.
def _cand(name: str) -> Candidate:
    return Candidate(name=name, kind="theorem", statement="", module="",
                     score=0.0)


CANDIDATES = [_cand("Polynomial.coeff_C_mul"),
              _cand("Polynomial.C_dvd_iff_dvd_coeff"),
              _cand("Polynomial.coeff_sub")]


def _compile(body: str) -> tuple[bool, str]:
    src = f"{IMPORTS}\nset_option maxHeartbeats 1000000\n\n{H_CD_LEAF}{body}\n"
    res = compile_lean(src, timeout_s=1500, lean_project_root=LEAN_ROOT)
    return bool(res.ok), (res.errors or "")


def test_confirmation_closes_the_h_cd_leaf():
    """The whole point: the tactic recognition builds must be accepted."""
    tac = confirm_tactic(CANDIDATES)
    body = "  " + tac.replace("\n", "\n  ")
    ok, errs = _compile(body)
    assert ok, f"confirm_tactic no longer closes the h_cd leaf:\n{errs[:2000]}"


def test_the_zero_step_forms_alone_would_still_fail():
    """Pins WHY the form was added. If this ever passes, one of the four
    original forms started closing the leaf and the extra one may be
    redundant — worth re-measuring rather than assuming."""
    zero_step = [f"({f.format(n='Polynomial.C_dvd_iff_dvd_coeff')})"
                 for f in ("exact {n}", "exact {n} _ _",
                           "simpa using {n}", "rw [{n}]")]
    body = "  first\n    | " + "\n    | ".join(zero_step)
    ok, _ = _compile(body)
    assert not ok, ("a zero-step citation now closes the h_cd leaf — "
                    "re-measure whether `intros; simp only [*, _]` is "
                    "still needed")
