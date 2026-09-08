import Mathlib
open Nat
/--
Euler's sum of powers conjecture (1769): no fourth power is a sum of
three positive fourth powers.
FALSE. Frye (1988): 95800^4 + 217519^4 + 414560^4 = 422481^4.

Seeded as a HARNESS VALIDATOR: the counterexample is known, but it is
large — this tests whether the model can produce a specific witness it
must recall rather than derive.
-/
theorem conj_euler_powers
: (∀ a b c d : ℕ, 0 < a → 0 < b → 0 < c → 0 < d →
    a ^ 4 + b ^ 4 + c ^ 4 ≠ d ^ 4) := by sorry
