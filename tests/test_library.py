"""Disposable, kernel-gated Lean library building.

Motivation: `order5_lrs_a0_unit` was unreachable because Mathlib has 21
`LinearRecurrence` declarations and no power-sum representation. No search
on the target can succeed when the prerequisite does not exist — it has to
be built first. ARM cannot do this: it invents lemmas only in service of a
stuck leaf, so "what does this domain need?" is never asked.

The properties that make a build safe to run are what this file pins down:

* a declaration enters the library ONLY on a real compile with no
  sorry/admit/native_decide;
* a failed declaration does not abort the build (a library is inherently
  partial — 8 of 12 useful declarations beats nothing);
* nothing is written outside the caller's output directory, so
  `rm -rf <out>` is complete cleanup and the persistent cross-run bank
  cannot be polluted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search.library import (                                   # noqa: E402
    build_library, decl_head, parse_plan, LibraryResult, LibraryDecl,
)


def _plan(*names) -> str:
    return json.dumps({"decls": [
        {"name": n, "kind": "lemma",
         "statement": f"lemma {n} : True", "rationale": "r"}
        for n in names]})


def _decl(name: str, body: str = ":= trivial") -> str:
    return json.dumps({"declaration": f"lemma {name} : True {body}",
                       "problem": ""})


class _H:
    """Scripts the planner and the per-declaration builder."""

    def __init__(self, plan: str, replies: list[str], ok):
        self.plan = plan
        self.replies = list(replies)
        self.ok = ok                     # (n_verify) -> bool
        self.n_verify = 0
        self.headers: list[str] = []

    def llm(self, system, user):
        if "planning a small, self-contained" in system:
            return self.plan
        return self.replies.pop(0) if self.replies else _decl("fallback")

    def verify(self, header, body):
        self.n_verify += 1
        self.headers.append(header)
        good = self.ok(self.n_verify)
        return {"ok": good, "errors": None if good else "error: nope"}


# ---- planning ---------------------------------------------------------------

def test_parse_plan_keeps_order_and_dedupes():
    plan, err = parse_plan(_plan("a", "b", "a"), 10)
    assert err is None
    assert [d["name"] for d in plan] == ["a", "b"]


def test_parse_plan_respects_max_decls():
    plan, _ = parse_plan(_plan("a", "b", "c"), 2)
    assert len(plan) == 2


def test_parse_plan_rejects_unknown_kind():
    raw = json.dumps({"decls": [
        {"name": "x", "kind": "lemma", "statement": "notakeyword x : True"}]})
    plan, err = parse_plan(raw, 5)
    assert plan is None and err


def test_decl_head_reads_kind_and_name():
    assert decl_head("noncomputable def foo (n : ℕ) : ℕ := n") == ("def", "foo")
    assert decl_head("structure Bar where") == ("structure", "Bar")
    assert decl_head("nonsense") == (None, None)


# ---- the kernel gate --------------------------------------------------------

def test_only_kernel_accepted_decls_enter_the_library():
    h = _H(_plan("a", "b"), [_decl("a"), _decl("b")],
           ok=lambda n: n == 1)          # only the first compiles
    res = build_library("spec", llm_call=h.llm, verify_fn=h.verify,
                        attempts_per_decl=1)
    assert [d.name for d in res.decls] == ["a"]
    assert [d.name for d in res.failed] == ["b"]


def test_forbidden_tactic_never_reaches_lean():
    h = _H(_plan("a"), [_decl("a", ":= by sorry"), _decl("a")],
           ok=lambda n: True)
    res = build_library("spec", llm_call=h.llm, verify_fn=h.verify,
                        attempts_per_decl=2)
    assert h.n_verify == 1               # the sorry attempt was not compiled
    assert [d.name for d in res.decls] == ["a"]


def test_emitted_declarations_contain_no_escape_hatch():
    """Checked on the DECLARATIONS: the file's doc header legitimately
    mentions `sorry` when stating that none are present."""
    h = _H(_plan("a"), [_decl("a")], ok=lambda n: True)
    res = build_library("spec", llm_call=h.llm, verify_fn=h.verify)
    for d in res.decls:
        for bad in ("sorry", "admit", "native_decide"):
            assert bad not in d.source


def test_failed_decl_does_not_abort_the_build():
    """A library is inherently partial."""
    h = _H(_plan("a", "b", "c"), [_decl("a"), _decl("b"), _decl("c")],
           ok=lambda n: n != 1)          # first fails, rest succeed
    res = build_library("spec", llm_call=h.llm, verify_fn=h.verify,
                        attempts_per_decl=1)
    assert [d.name for d in res.decls] == ["b", "c"]
    assert len(res.failed) == 1


def test_verified_decls_are_in_scope_for_later_ones():
    h = _H(_plan("a", "b"), [_decl("a"), _decl("b")], ok=lambda n: True)
    build_library("spec", llm_call=h.llm, verify_fn=h.verify)
    assert "lemma a : True" in h.headers[1]      # `a` visible when building b


def test_failed_decl_is_not_in_scope_for_later_ones():
    h = _H(_plan("a", "b"), [_decl("a"), _decl("b")],
           ok=lambda n: n != 1)
    build_library("spec", llm_call=h.llm, verify_fn=h.verify,
                  attempts_per_decl=1)
    assert "lemma a : True" not in h.headers[-1]


def test_retries_feed_the_lean_error_back():
    h = _H(_plan("a"), [_decl("a"), _decl("a")], ok=lambda n: n == 2)
    res = build_library("spec", llm_call=h.llm, verify_fn=h.verify,
                        attempts_per_decl=3)
    assert [d.name for d in res.decls] == ["a"]
    assert res.decls[0].attempts == 2


def test_llm_failure_is_survivable():
    def boom(system, user):
        raise RuntimeError("down")
    res = build_library("spec", llm_call=boom,
                        verify_fn=lambda h, b: {"ok": True})
    assert res.decls == [] and res.planned == 0


def test_verifier_failure_is_survivable():
    def boom(header, body):
        raise RuntimeError("lake died")
    h = _H(_plan("a"), [_decl("a")], ok=lambda n: True)
    res = build_library("spec", llm_call=h.llm, verify_fn=boom)
    assert res.decls == [] and len(res.failed) == 1


# ---- the sandbox ------------------------------------------------------------

def test_emit_writes_only_inside_out_dir(tmp_path):
    h = _H(_plan("a"), [_decl("a")], ok=lambda n: True)
    res = build_library("spec", llm_call=h.llm, verify_fn=h.verify,
                        name="MyLib")
    files = res.emit(tmp_path / "lib")
    written = {p.resolve() for p in files}
    assert all(str(p).startswith(str((tmp_path / "lib").resolve()))
               for p in written)
    assert (tmp_path / "lib" / "MyLib.lean").exists()
    assert (tmp_path / "lib" / "manifest.json").exists()


def test_nothing_is_written_before_emit(tmp_path):
    """An abandoned build leaves no trace at all."""
    h = _H(_plan("a"), [_decl("a")], ok=lambda n: True)
    build_library("spec", llm_call=h.llm, verify_fn=h.verify)
    assert list(tmp_path.iterdir()) == []


def test_removing_out_dir_is_complete_cleanup(tmp_path):
    import shutil
    h = _H(_plan("a"), [_decl("a")], ok=lambda n: True)
    res = build_library("spec", llm_call=h.llm, verify_fn=h.verify)
    d = tmp_path / "lib"
    res.emit(d)
    shutil.rmtree(d)
    assert not d.exists() and list(tmp_path.iterdir()) == []


def test_manifest_records_failures_and_is_json(tmp_path):
    h = _H(_plan("a", "b"), [_decl("a"), _decl("b")], ok=lambda n: n == 1)
    res = build_library("spec", llm_call=h.llm, verify_fn=h.verify,
                        attempts_per_decl=1)
    res.emit(tmp_path)
    man = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert man["verified"] == 1 and man["failed"] == 1
    assert man["failed_decls"][0]["name"] == "b"
    assert man["failed_decls"][0]["error"]


def test_runner_refuses_to_write_into_project_state():
    """`--out src/` (or lean/, data/…) must be refused outright."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_bl", Path(__file__).resolve().parents[1]
        / "scripts" / "build_library.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = Path(__file__).resolve().parents[1]
    for bad in ("lean", "src", "data", "tests"):
        with pytest.raises(SystemExit):
            mod._check_out_dir(root / bad / "x")
    mod._check_out_dir(root / "results" / "libraries" / "ok")   # allowed


def test_library_file_has_a_header_and_the_spec(tmp_path):
    h = _H(_plan("a"), [_decl("a")], ok=lambda n: True)
    res = build_library("power sums for LRS", llm_call=h.llm,
                        verify_fn=h.verify, name="LRSPowerSum")
    text = res.to_lean()
    assert "# LRSPowerSum" in text
    assert "power sums for LRS" in text
    assert "import Mathlib" in text
