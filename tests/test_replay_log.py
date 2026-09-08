"""Ordinal replay for pipeline cells, and the trap it exists to avoid.

`run_minif2f`'s Checkpoint keys on content, which is right there. The
DAG loop breaks that assumption: `--sketch-attempts 3` deliberately
re-issues the SAME prompt hoping for a DIFFERENT sample. Content-keyed,
attempt 2 would replay attempt 1 and the run would silently collapse to
one attempt while still reporting three.

So calls are recorded at their ORDINAL within a kind, plus a hash of
their inputs, and the log truncates the moment the trajectory diverges.

No Lean, no network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.replay import ReplayLog  # noqa: E402


def test_disabled_log_is_inert(tmp_path):
    p = tmp_path / "r.json"
    log = ReplayLog(p, enabled=False)
    calls = []
    for _ in range(3):
        log.step("llm", ("sys", "user"), lambda: (calls.append(1), "x")[1])
    assert len(calls) == 3
    assert not p.exists()
    assert log.hits == 0


def test_repeated_prompt_is_a_new_draw_on_a_fresh_run(tmp_path):
    """THE trap: three sketch attempts, one prompt, three samples."""
    p = tmp_path / "r.json"
    log = ReplayLog(p, enabled=True)
    drawn = iter(["sketch A", "sketch B", "sketch C"])
    got = [log.step("llm", ("sys", "same prompt"), lambda: next(drawn))
           for _ in range(3)]
    assert got == ["sketch A", "sketch B", "sketch C"], (
        "identical prompts collapsed into one cached sample")
    assert log.hits == 0


def test_resume_replays_the_sequence_in_order(tmp_path):
    p = tmp_path / "r.json"
    first = ReplayLog(p, enabled=True)
    drawn = iter(["sketch A", "sketch B", "sketch C"])
    for _ in range(3):
        first.step("llm", ("sys", "same prompt"), lambda: next(drawn))

    # The cell died; the chain restarted it.
    second = ReplayLog(p, enabled=True)
    def boom():
        raise AssertionError("recomputed a recorded call")
    got = [second.step("llm", ("sys", "same prompt"), boom) for _ in range(3)]
    assert got == ["sketch A", "sketch B", "sketch C"]
    assert second.hits == 3


def test_resume_continues_live_past_the_end_of_the_log(tmp_path):
    p = tmp_path / "r.json"
    first = ReplayLog(p, enabled=True)
    first.step("llm", ("sys", "p"), lambda: "recorded")

    second = ReplayLog(p, enabled=True)
    assert second.step("llm", ("sys", "p"), lambda: "live") == "recorded"
    assert second.step("llm", ("sys", "p2"), lambda: "live") == "live"
    assert second.hits == 1


def test_divergence_truncates_and_goes_live(tmp_path):
    """A different prompt at position N means a different trajectory.

    Everything recorded after it belongs to the old path and must not
    be replayed into the new one.
    """
    p = tmp_path / "r.json"
    first = ReplayLog(p, enabled=True)
    first.step("llm", ("sys", "prompt one"), lambda: "old 0")
    first.step("llm", ("sys", "prompt two"), lambda: "old 1")

    second = ReplayLog(p, enabled=True)
    got = second.step("llm", ("sys", "DIFFERENT"), lambda: "new 0")
    assert got == "new 0"
    assert second.diverged_at == "llm#0"
    # Position 1's recording was from the abandoned path.
    assert second.step("llm", ("sys", "prompt two"), lambda: "new 1") == "new 1"


def test_kinds_have_independent_ordinals(tmp_path):
    p = tmp_path / "r.json"
    log = ReplayLog(p, enabled=True)
    log.step("llm", ("a",), lambda: "llm0")
    log.step("verify", ("b",), lambda: {"ok": False})
    log.step("probe", ("c",), lambda: {"ok": True})
    again = ReplayLog(p, enabled=True)
    assert again.step("llm", ("a",), lambda: "X") == "llm0"
    assert again.step("verify", ("b",), lambda: {"ok": True}) == {"ok": False}
    assert again.step("probe", ("c",), lambda: {"ok": False}) == {"ok": True}


def test_verify_is_keyed_by_imports_too(tmp_path):
    """Same proof, different import set — a different question."""
    p = tmp_path / "r.json"
    first = ReplayLog(p, enabled=True)
    first.step("verify", ("hdr", "body", "import Mathlib.Data.Finset.Basic"),
               lambda: {"ok": False})
    second = ReplayLog(p, enabled=True)
    got = second.step(
        "verify", ("hdr", "body", "import Mathlib.NumberTheory.Divisors"),
        lambda: {"ok": True})
    assert got == {"ok": True}, "a stale verdict survived an import change"


def test_truncated_log_does_not_kill_the_run(tmp_path):
    p = tmp_path / "r.json"
    p.write_text('{"llm": {"0": {"key": "ab', encoding="utf-8")
    log = ReplayLog(p, enabled=True)
    assert log.step("llm", ("sys", "p"), lambda: "recomputed") == "recomputed"


def test_write_is_atomic_and_summary_reports(tmp_path):
    p = tmp_path / "r.json"
    log = ReplayLog(p, enabled=True)
    log.step("verify", ("h", "b", "i"), lambda: {"ok": True})
    assert p.exists() and not p.with_suffix(".tmp").exists()
    json.loads(p.read_text(encoding="utf-8"))
    s = log.summary()
    assert s["recorded"]["verify"] == 1 and s["diverged_at"] is None


def test_vetoed_result_is_never_recorded(tmp_path):
    """A compile that never ran must not become a permanent verdict.

    Without the veto, an infrastructure failure recorded at ordinal N is
    replayed on every later resume — the run would inherit a fabricated
    failure forever, and the segmented-restart strategy that exists to
    ESCAPE spawn failures would instead cement them.
    """
    p = tmp_path / "r.json"
    infra = {"ok": False, "errors": "error: INFRASTRUCTURE — lake exited"}
    real = lambda v: "INFRASTRUCTURE" not in (v.get("errors") or "")

    first = ReplayLog(p, enabled=True)
    first.step("verify", ("h", "b", "i"), lambda: infra, record=real)

    second = ReplayLog(p, enabled=True)
    got = second.step("verify", ("h", "b", "i"),
                      lambda: {"ok": True, "errors": ""}, record=real)
    assert got == {"ok": True, "errors": ""}, "replayed a fabricated failure"
    assert second.hits == 0


def test_veto_truncates_the_poisoned_tail(tmp_path):
    """Work done after a fabricated verdict was derived from it."""
    p = tmp_path / "r.json"
    real = lambda v: "INFRASTRUCTURE" not in (v.get("errors") or "")
    log = ReplayLog(p, enabled=True)
    log.step("verify", ("a",), lambda: {"ok": False, "errors": "real error"},
             record=real)
    log.step("verify", ("b",), lambda: {"ok": True, "errors": ""}, record=real)
    # Now a run where call #1 comes back as infrastructure.
    again = ReplayLog(p, enabled=True)
    again.step("verify", ("a",), lambda: {"ok": False, "errors": "x"},
               record=real)                      # replayed
    again.step("verify", ("b",),
               lambda: {"ok": False, "errors": "INFRASTRUCTURE"}, record=real)
    third = ReplayLog(p, enabled=True)
    third.step("verify", ("a",), lambda: {"ok": False, "errors": "x"},
               record=real)
    got = third.step("verify", ("b",), lambda: {"ok": True, "errors": ""},
                     record=real)
    assert got["ok"], "poisoned tail survived and was replayed"


def test_real_verdicts_still_replay_normally(tmp_path):
    p = tmp_path / "r.json"
    real = lambda v: "INFRASTRUCTURE" not in (v.get("errors") or "")
    first = ReplayLog(p, enabled=True)
    first.step("verify", ("h", "b", "i"),
               lambda: {"ok": False, "errors": "unsolved goals"}, record=real)
    second = ReplayLog(p, enabled=True)

    def boom():
        raise AssertionError("recomputed a real verdict")

    got = second.step("verify", ("h", "b", "i"), boom, record=real)
    assert got["errors"] == "unsolved goals" and second.hits == 1
