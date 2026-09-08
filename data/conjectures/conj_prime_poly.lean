import Mathlib
open Nat
/--
Euler's prime-generating polynomial `n^2 + n + 41` is prime for every n.
FALSE at n = 40: `40^2 + 40 + 41 = 1681 = 41^2`.

Seeded as a HARNESS VALIDATOR: the easiest of the three. A run that
cannot refute this one has a broken search, not a hard problem.
-/
theorem conj_prime_poly
: (∀ n : ℕ, Nat.Prime (n ^ 2 + n + 41)) := by sorry
