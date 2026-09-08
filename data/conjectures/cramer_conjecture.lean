import Mathlib
/--
CRAMÉR'S CONJECTURE (1936), big-O form. OPEN.

Prime gaps are `O((log p)^2)`: there is a constant `C` with
`p_{n+1} - p_n ≤ C * (log p_n)^2` for every `n`.

A single universal `C` is enough — finitely many small `n` are absorbed by
enlarging `C` — so no "sufficiently large" side condition is needed.

Best unconditional result is far weaker (Baker–Harman–Pintz: gaps
`≪ p^0.525`), and even the Riemann Hypothesis only yields `O(√p log p)`.
-/
theorem cramer_conjecture
: (∃ C : ℝ, ∀ n : ℕ,
    ((Nat.nth Nat.Prime (n + 1) : ℝ) - (Nat.nth Nat.Prime n : ℝ))
      ≤ C * (Real.log (Nat.nth Nat.Prime n : ℝ)) ^ 2) := by sorry
