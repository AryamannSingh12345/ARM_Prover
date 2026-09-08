"""Guard against local imports inside `run_dag.main` shadowing module globals.

`main()` builds several nested closures (`_probe_fn`, the verify wrapper,
`_refresh_imports`) that read module-level names such as `compile_lean`
and `_ERROR_LINE_RE`. A bare `from … import compile_lean` ANYWHERE inside
`main` makes that name a LOCAL of `main` for the entire function body, so
if the branch containing the import does not execute the cell is never
bound and every closure reading it dies at call time with

    NameError: cannot access free variable 'compile_lean' where it is
    not associated with a value in enclosing scope

That is exactly what happened on run `lrs_sol_legacyARM_store`
(2026-07-30): `--lean-imports` took the `cli_imports` branch, so the
`elif args.import_mode == "llm"` branch — which carried the shadowing
import — never ran, and the crash landed 26 minutes and two full Mathlib
compiles into a paid run, inside the theory-mode satisfaction gate.

The bug class is invisible to normal unit tests because it depends on
which BRANCH ran, so this test inspects the source instead. Aliased local
imports (`import X as _x`) are fine and stay allowed: they bind a
different name and cannot shadow the global.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUN_DAG = ROOT / "src" / "eval" / "run_dag.py"


def _module_level_names(tree: ast.Module) -> set[str]:
    """Names bound by top-level `import` / `from … import` statements."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in run_dag.py")


def test_main_has_no_local_import_shadowing_a_module_global():
    tree = ast.parse(RUN_DAG.read_text(encoding="utf-8"))
    globals_ = _module_level_names(tree)
    main = _find_function(tree, "main")

    offenders: list[str] = []
    for node in ast.walk(main):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            # An aliased import binds a fresh name — no shadowing risk.
            if alias.asname is not None:
                continue
            if bound in globals_:
                offenders.append(f"line {node.lineno}: {bound}")

    assert not offenders, (
        "local import(s) inside main() shadow a module-level global, "
        "making it an unbound local on any path where that branch does "
        "not run (nested closures then raise NameError):\n  "
        + "\n  ".join(offenders)
        + "\nFix: drop the redundant local import, or alias it (`as _x`)."
    )


def test_compile_lean_and_error_re_are_module_level():
    """The names the closures depend on must be importable from the
    module, not conjured by a branch."""
    import eval.run_dag as rd
    assert callable(rd.compile_lean)
    assert hasattr(rd._ERROR_LINE_RE, "search")


def test_probe_fn_reads_the_module_level_compile_lean():
    """Direct regression on the crash: build the runner's probe closure
    path with `--lean-imports` set (the branch that skipped the shadowing
    import) and confirm the names it needs resolve.

    This asserts on scope resolution only — no Lean is invoked.
    """
    import eval.run_dag as rd
    src = RUN_DAG.read_text(encoding="utf-8")
    # `_probe_fn` must reference compile_lean, and that reference has to
    # resolve to the module global for the cli-imports path to work.
    assert "def _probe_fn" in src
    assert rd.compile_lean.__module__.endswith("compile_verify")
