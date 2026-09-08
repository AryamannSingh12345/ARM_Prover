import Mathlib
open Finset
/--
Frankl's union-closed sets conjecture, TWO-ELEMENT CASE (Sarvate–Renaud).
If a union-closed family contains a 2-element set `{a, b}`, then `a` or `b`
lies in at least half of its members.

KNOWN TRUE, and materially harder than the singleton case — the natural
injection no longer works directly.
-/
theorem frankl_two_element
: (∀ (A : Finset (Finset ℕ)) (a b : ℕ), a ≠ b →
    ({a, b} : Finset ℕ) ∈ A →
    (∀ S ∈ A, ∀ T ∈ A, S ∪ T ∈ A) →
    (2 * (A.filter (fun S => a ∈ S)).card ≥ A.card
      ∨ 2 * (A.filter (fun S => b ∈ S)).card ≥ A.card)) := by sorry
