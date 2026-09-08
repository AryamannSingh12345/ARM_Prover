"""The error-driven import refresh must compile its OWN sorry stub.

`_refresh_imports` decides whether to adopt new imports by compiling
`<merged imports>\n\n<stubbed decls>`, where the stub is sorry-terminated
by construction — either lemma statements it stubs itself, or the bare
`header := by sorry` the error-driven path falls back to.

`compile_lean` rejects sorry-bearing sources BEFORE compiling and returns
an `error:`-MARKED message (deliberately, so strict callers fail closed).
The refresh then reads that marker as "the merged import set is broken"
and keeps the old imports. Net effect: the gate can never pass, and the
mechanism returns None on every call without raising, logging, or
tracing anything.

Measured on `mf18_amc12a_2003_p23_scaffold_budget` (2026-08-27): six
repair rounds of `Unknown constant Nat.divisors`, the declaration graph
holding the right answer the whole time, zero imports adopted.

No Lean: the pre-rejection happens before any compile, and the guard
test reads source.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RUN_DAG = Path(__file__).resolve().parents[1] / "src" / "eval" / "run_dag.py"


def test_strict_compile_rejects_a_sorry_stub_before_compiling():
    """The behaviour the refresh gate was tripping over."""
    from backend.compile_verify import compile_lean
    res = compile_lean("import Mathlib\n\ntheorem t : True := by sorry\n",
                       timeout_s=5)
    assert not res.ok
    # Marked `error:` on purpose — which is exactly why a caller that
    # scans for `error:` mistakes it for a broken import set.
    assert "error:" in (res.errors or "")
    assert "rejected before compile" in (res.errors or "")


def _gate_calls_in_refresh_imports() -> list[ast.Call]:
    """Every compile call inside `_refresh_imports`."""
    tree = ast.parse(RUN_DAG.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_refresh_imports"):
            return [n for n in ast.walk(node)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id in ("_cl", "compile_lean")]
    raise AssertionError("_refresh_imports not found in run_dag.py")


def test_refresh_gate_opts_out_of_sorry_rejection():
    calls = _gate_calls_in_refresh_imports()
    assert calls, "no compile call found in _refresh_imports"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "reject_sorry" in kw, (
            "the import-refresh gate compiles a sorry stub; without "
            "reject_sorry=False it is rejected before compiling and the "
            "refresh silently never adopts an import")
        assert isinstance(kw["reject_sorry"], ast.Constant)
        assert kw["reject_sorry"].value is False


def test_every_sorry_stub_gate_opts_out():
    """Any compile whose source literal mentions `sorry` is a gate, not a
    verification — it must opt out, or it fails closed and silently.

    Written as a sweep rather than a check of the one site that broke:
    the premise-seeded gate got this right and `_refresh_imports` did
    not, so the next one added is the thing to catch.
    """
    tree = ast.parse(RUN_DAG.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("_cl", "compile_lean")):
            continue
        if not node.args:
            continue
        src_arg = ast.dump(node.args[0])
        if "sorry" not in src_arg:
            continue
        kw = {k.arg: k.value for k in node.keywords}
        ok = (isinstance(kw.get("reject_sorry"), ast.Constant)
              and kw["reject_sorry"].value is False)
        if not ok:
            offenders.append(node.lineno)
    assert not offenders, (
        f"run_dag.py compiles a sorry-stubbed source without "
        f"reject_sorry=False at line(s) {offenders}; that source is "
        f"rejected before compiling and the caller reads the `error:` "
        f"marker as a real failure")
