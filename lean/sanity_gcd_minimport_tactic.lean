import Mathlib.Data.Nat.GCD.Basic
import Mathlib.Tactic

example (n : Nat)
    (h? : 0 < n)
    (h? : Nat.gcd n 40 = 10)
    (h? : Nat.lcm n 40 = 280) :
    n = 70 := by
  have hprod := Nat.gcd_mul_lcm n 40
