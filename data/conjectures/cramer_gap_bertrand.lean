import Mathlib
/--
VALIDATOR (known true): consecutive primes satisfy `p_{n+1} ≤ 2 * p_n`.

Immediate from Bertrand's postulate (`Nat.exists_prime_lt_and_le_two_mul`,
present in Mathlib): there is a prime in `(p_n, 2*p_n]`, and `p_{n+1}` is the
least prime above `p_n`.

Its job is to check the `Nat.nth Nat.Prime` plumbing BEFORE any budget is
spent on the open statements below. If this fails, nothing after it is
interpretable.
-/
theorem cramer_gap_bertrand
: (∀ n : ℕ, Nat.nth Nat.Prime (n + 1) ≤ 2 * Nat.nth Nat.Prime n) := by sorry
