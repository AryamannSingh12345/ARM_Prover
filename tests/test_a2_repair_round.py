"""Regression tests for the four repairs from p1963a2_dag_v1 (2026-08-18).

That run failed on an API error after 11.8 h, but its trace isolated four
distinct defects, each costing real time:

  A. theory LEAF TACTICS got no hygiene. Four consecutive theory rounds
     died at the satisfaction gate on the SAME tactic-level coercion
     (`h_f1 : f 1 = 1` used where `1 ≤ f 1` was wanted), ~1700 s of probe
     compiles, while the model rewrote the LEMMA each time.
  B. the gate stubs every lemma with `sorry`, so its failure CANNOT be a
     lemma's fault — yet the ledger did not say so, which is why the
     model optimised the wrong object four times.
  C. a COMMITTED theory landing on the final repair round wasted itself:
     sketch 1 proved `aux_strictMonoOn_succ_growth`, fixed `h_lower` —
     the obstacle that had blocked both prior rounds — and then had no
     iteration left for the separately-broken closer.
  D. `compile_lean` ran a full Mathlib compile on sources it had ALREADY
     rejected: 22 min to reject a body of `sorry`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SRC = (Path(__file__).resolve().parents[1]
       / "src" / "search" / "proof_dag.py").read_text(encoding="utf-8")


# ---- A: leaf tactics get hygiene -------------------------------------

def test_theory_leaf_tactics_are_hardened():
    i = SRC.index("REPAIR A:")
    tail = SRC[i:i + 1200]
    assert "harden_tactic_block(v)" in tail
    assert "for k, v in tactics.items()" in tail
    # gated by the same flag as the prove step, so it stays ablatable
    assert "if tactic_hygiene:" in tail


def test_hygiene_would_close_the_a2_coercion():
    """`1 ≤ f 1` from `h_f1 : f 1 = 1` is exactly what the sweep closes."""
    from search.proof_dag import harden_tactic_block
    out = harden_tactic_block("exact h_f1")
    assert "all_goals try omega" in out


# ---- B: gate failures are attributed to tactics, not lemmas ----------

def test_gate_failure_says_lemmas_are_assumed():
    i = SRC.index("REPAIR B:")
    tail = SRC[i:i + 2000]
    assert "ASSUMED" in tail and "sorry" in tail
    assert "CANNOT be caused by a lemma" in tail
    # and it must point at the things that CAN be at fault
    assert "leaf TACTIC" in tail


# ---- C: a committed theory earns a round -----------------------------

def test_bonus_round_is_bounded_and_granted_on_commit():
    assert "_MAX_BONUS_ROUNDS = 1" in SRC
    assert "_bonus_granted = 0" in SRC
    # the loop's range must leave room for the bonus ...
    assert "range(repair_rounds + 1 + _MAX_BONUS_ROUNDS)" in SRC
    # ... but the break still fires at the *granted* budget, not the max
    assert "if round_no >= repair_rounds + _bonus_granted:" in SRC


def _drive(commit: bool, theory_on: bool):
    """Run the REAL loop with attributable errors so the leaf-repair path
    (where the bonus is granted) is exercised, not the bare-timeout one.

    The lemma-store harness exits via bare-timeout, which is a DIFFERENT
    branch and does not grant a bonus — checking the bonus there would
    have proved nothing.
    """
    import itertools

    from search import proof_dag
    from search.dag.lemma_store import LemmaStore

    header = "import Mathlib\n\ntheorem tgt (n : ℕ) : n + 0 = n"
    sketch = ('{"haves": [{"id": "h1", "type": "n + 0 = n", '
              '"tactic": "simp", "depends": []}], "closer": "exact h1"}')
    theory = ('{"defs": [], "lemmas": [{"name": "aux_good", "statement": '
              '"lemma aux_good : (1:ℕ) + 0 = 1"}], '
              '"leaf_tactics": {"h1": "exact aux_good"}, "rationale": "r"}')
    counter = itertools.count()

    def llm(system, user):
        if system == proof_dag.SKETCH_SYSTEM:
            return sketch
        if system == proof_dag.THEORY_SYSTEM:
            return theory
        if system == proof_dag.PROVE_LEMMA_SYSTEM:
            return '{"proof": "rfl"}'
        # distinct each round so the loop never exits on "nothing changed"
        return '{"repairs": [{"id": "h1", "tactic": "simp [x%d]"}]}' % next(counter)

    def verify(hdr, body):
        if hdr.rstrip().endswith("lemma aux_good : (1:ℕ) + 0 = 1"):
            return {"ok": bool(commit),
                    "errors": None if commit else "x.lean:1:1: error: nope",
                    "body_line_offset": 1}
        return {"ok": False,
                "errors": "x.lean:4:2: error: unsolved goals\n  ⊢ n + 0 = n",
                "body_line_offset": 3}

    events: list = []
    proof_dag.attempt_dag_proof(
        header, sketch_llm_call=llm, verify_fn=verify,
        probe_fn=lambda h, b: {"ok": True, "errors": None,
                               "body_line_offset": 3},
        sketch_attempts=1, repair_rounds=2,
        abduce_lemmas=theory_on, abduce_mode="theory",
        abduce_theory_trigger="always", abduce_theory_rounds=1,
        lemma_store=LemmaStore(),
        trace=lambda kind, **kw: events.append((kind, kw)))
    return {
        "rounds": [e[1].get("round") for e in events if e[0] == "assembled"],
        "bonus": len([e for e in events if e[0] == "repair_bonus_round"]),
        "break": [e[1].get("reason") for e in events if e[0] == "repair_break"],
    }


def test_bonus_does_not_fire_without_a_commit():
    """repair_rounds=2 must still mean exactly rounds 0,1,2."""
    for theory_on in (False, True):
        r = _drive(commit=False, theory_on=theory_on)
        assert r["rounds"] == [0, 1, 2], (theory_on, r)
        assert r["bonus"] == 0
        assert r["break"] == ["repair rounds exhausted"]


def test_bonus_adds_exactly_one_round_on_commit():
    r = _drive(commit=True, theory_on=True)
    assert r["rounds"] == [0, 1, 2, 3], r
    assert r["bonus"] == 1
    assert r["break"] == ["repair rounds exhausted"]


def test_bonus_granted_at_both_commit_sites():
    """Leaf-theory commit and closer-theory commit both qualify."""
    assert SRC.count("_bonus_granted += 1") == 2
    assert "theory committed — one extra repair" in SRC
    assert "closer theory committed" in SRC


def test_bonus_cannot_be_granted_twice_per_sketch():
    for guard in ("if fixed_ids and _bonus_granted < _MAX_BONUS_ROUNDS:",
                  "if _bonus_granted < _MAX_BONUS_ROUNDS:"):
        assert guard in SRC


# ---- D: no compile for a source already rejected ---------------------

def test_sorry_source_short_circuits_without_lake(tmp_path, monkeypatch):
    from backend import compile_verify

    called = {"n": 0}

    def boom(*a, **kw):
        called["n"] += 1
        raise AssertionError("lake must not run for a rejected source")

    monkeypatch.setattr(compile_verify.subprocess, "run", boom)
    res = compile_verify.compile_lean("theorem t : True := by sorry\n")
    assert res.ok is False
    assert res.used_sorry is True
    assert called["n"] == 0
    assert res.elapsed_s == 0.0
    assert "rejected before compile" in res.errors
    # FAIL-CLOSED: several callers judge by scanning for `error:` rather
    # than reading `ok`. Without the marker a missed opt-out reads the
    # skipped compile as "clean" and ACCEPTS what it should reject —
    # exactly how p1963a2_dag_v2 passed every gate in 0.0s. With it, a
    # missed opt-out over-rejects instead: visible and safe.
    import re as _re

    from backend.compile_verify import _ERROR_LINE_RE
    assert _ERROR_LINE_RE.search(res.errors), (
        "short-circuit message must carry an `error:` marker so callers "
        "that scan error text fail CLOSED, not open")


def test_native_decide_short_circuits(tmp_path, monkeypatch):
    from backend import compile_verify

    monkeypatch.setattr(
        compile_verify.subprocess, "run",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no lake")))
    res = compile_verify.compile_lean("theorem t : True := by native_decide\n")
    assert res.ok is False and res.used_native_decide is True


def test_sorryax_also_short_circuits(monkeypatch):
    """The evasion that started all this still costs zero compiles."""
    from backend import compile_verify

    monkeypatch.setattr(
        compile_verify.subprocess, "run",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no lake")))
    res = compile_verify.compile_lean("theorem t : True := by exact sorryAx _ true\n")
    assert res.ok is False and res.used_sorry is True


def test_sorry_tolerant_callers_still_compile(monkeypatch):
    """reject_sorry=False must actually run lake.

    The short-circuit is only sound for callers that treat `ok` as the
    verdict. Three callers do NOT: the theory satisfaction probe, the
    import statement gate, and the premise-seeded import gate all compile
    a deliberately sorry-stubbed source and read the ERROR TEXT.

    MEASURED live on p1963a2_dag_v2 with the short-circuit unconditional:
    every probe returned in 0.0s, the caller's error-regex never matched
    this module's own rejection message, and so every satisfaction gate
    passed VACUOUSLY — while the import gate accepted a nonexistent
    `Mathlib.Topology.Defs` and nothing compiled for the rest of the run.
    """
    from types import SimpleNamespace

    from backend import compile_verify

    ran = {"n": 0}

    def fake(*a, **kw):
        ran["n"] += 1
        return SimpleNamespace(returncode=1, stdout="",
                               stderr="x.lean:1:0: error: boom\n")

    monkeypatch.setattr(compile_verify, "_write_temp",
                        lambda s: Path(__file__).parent / "_tmp_probe.lean")
    (Path(__file__).parent / "_tmp_probe.lean").write_text("x", encoding="utf-8")
    monkeypatch.setattr(compile_verify.subprocess, "run", fake)
    try:
        res = compile_verify.compile_lean(
            "theorem t : True := by sorry\n", reject_sorry=False,
            check_axioms=False)
        assert ran["n"] == 1, "lake must run for a sorry-tolerant caller"
        assert "error: boom" in res.errors
    finally:
        (Path(__file__).parent / "_tmp_probe.lean").unlink(missing_ok=True)


def test_all_repo_sorry_tolerant_callers_opt_out():
    """Every sorry-stubbed compile in the repo must pass reject_sorry=False.

    Six such callers exist and each was silently broken by the
    short-circuit until fixed:
      run_dag  _probe_fn                (satisfaction gate)
      run_dag  _stmt_gate               (import resolution)
      run_dag  premise-seeded gate      (import seeding)
      run_dag  _refresh_imports         (error-driven import refresh)
      scripts/check_bench_statements.py (statement sanity sweep)

    `_refresh_imports` was found last (2026-08-27) and had been broken
    the longest: it compiles `header := by sorry` to gate a merged
    import set, the strict path rejected that BEFORE compiling, and the
    `error:` marker in the rejection made the gate read "these imports
    are broken" — so it returned None on every call and no import was
    ever adopted. Measured on mf18_amc12a_2003_p23_scaffold_budget: six
    repair rounds of `Unknown constant Nat.divisors` with the
    declaration graph holding the answer the whole time. Hence 4 for
    run_dag, not 3.
    """
    root = Path(__file__).resolve().parents[1]
    # (file, how many compile_lean calls there must be with the opt-out)
    expected = {
        root / "src" / "eval" / "run_dag.py": 4,
        root / "scripts" / "check_bench_statements.py": 1,
    }
    for f, n in expected.items():
        text = f.read_text(encoding="utf-8")
        got = [l for l in text.splitlines()
               if "reject_sorry=False" in l and not l.strip().startswith("#")]
        assert len(got) == n, f"{f.name}: expected {n} opt-outs, got {len(got)}"


def test_every_sorry_tolerant_callsite_opts_out():
    """All four run_dag sorry-stubbed call sites pass reject_sorry=False."""
    src = (Path(__file__).resolve().parents[1]
           / "src" / "eval" / "run_dag.py").read_text(encoding="utf-8")
    # Count CALL SITES, not mentions: the rationale is also written in
    # comments, and counting those made this assertion meaningless.
    call_sites = [l for l in src.splitlines()
                  if "reject_sorry=False" in l and not l.strip().startswith("#")]
    assert len(call_sites) == 4, call_sites

    # Each sorry-stubbed source must have the opt-out within its own call.
    # The window is generous because the rationale comment sits between
    # the anchor and the argument.
    for anchor in ('f"{imports}\\n\\n{header} := by sorry\\n"',
                   'f"{merged}\\n\\n{header} := by sorry\\n"',
                   # the error-driven refresh gate: its stub is
                   # built earlier and passed in as `stubbed`
                   "{stubbed}"):
        i = src.index(anchor)
        assert "reject_sorry=False" in src[i:i + 400], anchor


def test_clean_source_still_compiles(tmp_path, monkeypatch):
    """The short-circuit must not swallow ordinary sources."""
    from types import SimpleNamespace

    from backend import compile_verify

    monkeypatch.setattr(compile_verify, "_write_temp",
                        lambda s: tmp_path / "c.lean")
    (tmp_path / "c.lean").write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        compile_verify.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""))
    res = compile_verify.compile_lean("theorem t : True := trivial\n",
                                      check_axioms=False)
    assert res.ok is True
