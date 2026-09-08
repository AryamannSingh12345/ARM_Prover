"""`--debug-dump-lean` keeps the exact source Lean rejected.

Motivation: in `putnam_easy8_sol_v1` the error `unknown tactic` fired eight
times across three lemmas and could not be diagnosed — the temp `.lean` is
deleted after each compile, and the trace stores the model's RESPONSE, not
the assembled file. The dump is the only artifact that ties a Lean error to
the text that produced it.

The backend already supported `debug_dump_path`; only the runner wiring was
missing, and nothing tested it. These tests exercise the backend contract
directly (no lake, no network) plus the flag's presence and default.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backend import compile_verify                          # noqa: E402


@pytest.fixture()
def fake_lake(monkeypatch):
    """Stub the lake subprocess so these tests need no Lean toolchain."""
    calls = {}

    class _Proc:
        returncode = 1
        stdout = "Try.lean:19:5: error: unknown tactic\n"
        stderr = ""

    def _run(argv, **kw):
        calls["argv"] = argv
        calls["cwd"] = kw.get("cwd")
        return _Proc()

    monkeypatch.setattr(compile_verify.subprocess, "run", _run)
    return calls


def test_dump_writes_the_exact_source(tmp_path, fake_lake):
    dump = tmp_path / "001_verify.lean"
    src = "import Mathlib\n\ntheorem t : True := by\n  bogus_tactic\n"
    compile_verify.compile_lean(src, timeout_s=5, debug_dump_path=dump)
    assert dump.exists()
    assert dump.read_text(encoding="utf-8") == src


def test_dump_writes_a_sidecar_with_replay_argv(tmp_path, fake_lake):
    dump = tmp_path / "001_verify.lean"
    compile_verify.compile_lean("import Mathlib\n", timeout_s=5,
                                debug_dump_path=dump)
    side = dump.with_suffix(".json")
    assert side.exists()
    d = json.loads(side.read_text(encoding="utf-8"))
    assert d["ok"] is False
    assert d["status"] in ("errors", "timeout", "ok")
    assert d["replay"]["argv"] and d["replay"]["lean_file"]
    assert "source_prefix" in d


def test_no_dump_path_writes_nothing(tmp_path, fake_lake):
    compile_verify.compile_lean("import Mathlib\n", timeout_s=5)
    assert list(tmp_path.iterdir()) == []


def test_dump_failure_never_breaks_a_run(tmp_path, fake_lake):
    """Debug plumbing must not be able to kill a run."""
    bad = tmp_path / "no_such_dir" / "deep" / "x.lean"
    res = compile_verify.compile_lean("import Mathlib\n", timeout_s=5,
                                      debug_dump_path=bad)
    assert res.ok is False          # returned normally despite unwritable path


# ---- runner flag ----------------------------------------------------------

def test_flag_exists_and_defaults_to_off():
    import eval.run_dag as rd
    ap = None
    import argparse
    import inspect
    src = inspect.getsource(rd.main)
    assert '"--debug-dump-lean"' in src
    # default off: the dump helper must short-circuit on a falsy value
    assert "if not args.debug_dump_lean:" in src


def test_runner_passes_dump_path_to_both_compile_paths():
    """Verify AND probe must both dump — the probe is where theory-mode
    sufficiency failures show up."""
    import inspect
    import eval.run_dag as rd
    src = inspect.getsource(rd.main)
    assert 'debug_dump_path=_dump_path("verify")' in src
    assert 'debug_dump_path=_dump_path("probe")' in src


def test_dump_sequence_is_per_problem():
    import inspect
    import eval.run_dag as rd
    src = inspect.getsource(rd.main)
    # counter reset inside the per-problem loop, not module scope
    assert "_dump_seq = [0]" in src
    assert "_dump_seq[0] += 1" in src
