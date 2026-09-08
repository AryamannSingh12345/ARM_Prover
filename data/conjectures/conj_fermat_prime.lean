import Mathlib
open Nat
/--
Fermat's conjecture (1650): every Fermat number `2^(2^n) + 1` is prime.
FALSE. Euler (1732) found `F₅ = 4294967297 = 641 * 6700417`.

Seeded as a HARNESS VALIDATOR: the counterexample is known, so a run
that fails to find it is a failure of the search, not of the target.
-/
theorem conj_fermat_prime
: (∀ n : ℕ, Nat.Prime (2 ^ (2 ^ n) + 1)) := by sorry
