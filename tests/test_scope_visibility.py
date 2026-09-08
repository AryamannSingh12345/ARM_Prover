"""The lemma prover must be told what is already in scope.

`prove_lemma_prompt` used to receive exactly three things: the statement,
the previous Lean error, and BM25-retrieved Mathlib names. It was never
told which sibling lemmas, store lemmas or defs were already spliced into
the header it would be compiled against — so the model re-derived facts
sitting a few lines above it in the same file.

Measured on `p1963a2_dag_v4`: `aux_odd_isRelPrime_two` was kernel-proved at t=15007s and
spliced into the header; the run then failed `aux_isRelPrime_two_three`,
which that lemma closes in one application, across seven attempts and two
theory rounds, and guessed the Coprime/IsRelPrime bridge six wrong ways
without ever finding `Nat.coprime_iff_isRelPrime`.

These tests pin the two halves of the fix: `declarations_in_scope` reads
the declarations out of a header, and `prove_lemma_prompt` shows them.
"""
from search.proof_dag import declarations_in_scope, prove_lemma_prompt


# The header as it stood when the run failed: prelude, one store-spliced
# def, and two lemmas already kernel-proved.
HEADER = """import Mathlib
set_option maxHeartbeats 800000

open Nat Finset in
def auxDouble (n : ℕ) : ℕ := 2 * n

lemma aux_odd_isRelPrime_two (k : ℕ) (hk : Odd k) : IsRelPrime 2 k := by
  rw [← Nat.coprime_iff_isRelPrime]
  simpa [Nat.coprime_two_left_iff_odd] using hk

lemma aux_mono_step (g : ℕ → ℕ) (h : StrictMono g) (n : ℕ) :
    g n < g (n + 1) := by
  exact h (Nat.lt_succ_self n)

theorem putnam_1963_a2 (f : ℕ → ℕ) : True := by
  trivial
"""


def test_recovers_every_declaration():
    got = declarations_in_scope(HEADER)
    assert len(got) == 4
    assert got[0].startswith("def auxDouble")
    assert got[1].startswith("lemma aux_odd_isRelPrime_two")
    assert got[2].startswith("lemma aux_mono_step")
    assert got[3].startswith("theorem putnam_1963_a2")


def test_lemma_proofs_are_dropped_defs_keep_their_body():
    got = declarations_in_scope(HEADER)
    # The statement is what makes a lemma citable; the proof is token cost.
    assert "IsRelPrime 2 k" in got[1]
    assert "coprime_iff_isRelPrime" not in got[1]
    assert ":=" not in got[1]
    # A def is only usable if you know what it unfolds to.
    assert "2 * n" in got[0]


def test_binder_default_does_not_end_the_signature():
    """`:=` inside brackets is not the body. Depth-aware, not `find`."""
    src = "lemma f (n : ℕ := 3) (h : n > 0) : True := by trivial\n"
    got = declarations_in_scope(src)
    assert got == ["lemma f (n : ℕ := 3) (h : n > 0) : True"]


def test_equation_style_declaration_survives():
    """No top-level `:=` at all — keep the whole block rather than
    silently emitting nothing."""
    src = "def fib : ℕ → ℕ\n  | 0 => 0\n  | 1 => 1\n"
    got = declarations_in_scope(src)
    assert got and got[0].startswith("def fib")
    assert "| 1 => 1" in got[0]


def test_attributes_and_modifiers_are_not_missed():
    src = ("@[simp]\nprivate lemma a_one : (1 : ℕ) = 1 := rfl\n\n"
           "noncomputable def w : ℕ := 0\n")
    got = declarations_in_scope(src)
    assert len(got) == 2
    assert "a_one" in got[0]
    assert "w" in got[1]


def test_truncation_bounds_both_ways():
    long_lemma = "lemma big : " + "P ∧ " * 400 + "Q := by tauto\n"
    got = declarations_in_scope(long_lemma, max_chars=80)
    assert len(got[0]) <= 84 and got[0].endswith("...")
    many = "".join(f"lemma l{i} : True := trivial\n\n" for i in range(50))
    assert len(declarations_in_scope(many, max_decls=12)) == 12


def test_empty_source_is_empty_not_an_error():
    assert declarations_in_scope("") == []
    assert declarations_in_scope("import Mathlib\nopen Nat\n") == []


def test_prompt_shows_scope_and_forbids_re_derivation():
    p = prove_lemma_prompt(
        "lemma aux_isRelPrime_two_three : IsRelPrime 2 3", None,
        in_scope=declarations_in_scope(HEADER))
    assert "ALREADY PROVED AND IN SCOPE" in p
    assert "aux_odd_isRelPrime_two" in p
    assert "CITE THEM BY NAME" in p
    # The regression itself: the lemma that would have closed it is named.
    assert "IsRelPrime 2 k" in p


def test_prompt_unchanged_when_nothing_is_in_scope():
    for scope in (None, []):
        p = prove_lemma_prompt("lemma L : True", None, in_scope=scope)
        assert "ALREADY PROVED AND IN SCOPE" not in p
        assert p.startswith("Lemma to prove:")


def test_syntax_only_repair_ignores_scope():
    """A parse error means the mathematics was never checked; a scope
    listing is noise that invites the model to rethink the proof."""
    p = prove_lemma_prompt("lemma L : True", "unexpected token",
                           prev_proof="simpa using h", syntax_only=True,
                           in_scope=declarations_in_scope(HEADER))
    assert "ALREADY PROVED AND IN SCOPE" not in p
    assert "MINIMAL edit" in p


def test_scope_section_precedes_the_mathlib_names():
    """What is already in the file beats what might be in Mathlib."""
    p = prove_lemma_prompt("lemma L : True", None, ["Nat.succ_le"],
                           in_scope=["lemma a_one : (1 : ℕ) = 1"])
    assert p.index("ALREADY PROVED") < p.index("Retrieved Mathlib names")
