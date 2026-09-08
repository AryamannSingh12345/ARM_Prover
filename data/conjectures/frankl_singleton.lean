import Mathlib
open Finset
/--
Frankl's union-closed sets conjecture, SINGLETON CASE (Knill / folklore).
If a union-closed family contains the singleton `{a}`, then `a` lies in at
least half of its members.

KNOWN TRUE — this is a validator. Proof: `S ↦ S ∪ {a}` injects the members
avoiding `a` into those containing it. A run that cannot do this one has a
broken search, not a hard target.
-/
theorem frankl_singleton
: (∀ (A : Finset (Finset ℕ)) (a : ℕ),
    ({a} : Finset ℕ) ∈ A →
    (∀ S ∈ A, ∀ T ∈ A, S ∪ T ∈ A) →
    2 * (A.filter (fun S => a ∈ S)).card ≥ A.card) := by sorry
