import Mathlib
/--
CRAMÉR'S CONJECTURE, ORIGINAL STRONG FORM. OPEN — AND POSSIBLY FALSE.

`limsup (p_{n+1} - p_n) / (log p_n)^2 = 1`.

Granville's refinement of Cramér's heuristic predicts the limsup is at least
`2 * exp (-γ) ≈ 1.1229 > 1`, so this equality is believed FALSE. It is
included as a formulation exercise and as a refutation target, NOT as
something to prove.
-/
theorem cramer_strong
: (Filter.limsup (fun n : ℕ =>
      ((Nat.nth Nat.Prime (n + 1) : ℝ) - (Nat.nth Nat.Prime n : ℝ))
        / (Real.log (Nat.nth Nat.Prime n : ℝ)) ^ 2) Filter.atTop = 1) := by
  sorry
