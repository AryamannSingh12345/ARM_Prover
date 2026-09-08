"""A stopped cell resumes where it stopped, and resumes IDENTICALLY.

The failure this prevents: a cell dies at minute 40 of 45 and the
restart re-pays for every sample and every Lean compile it had already
finished. On this host a compile under `import Mathlib` was measured at
900 s, so replaying a cell is not a small waste.

The property that makes it safe is that the checkpoint is a CACHE keyed
by a hash of the exact inputs — a hit returns what the call returned
before, so a resumed cell takes the path an uninterrupted one would. It
never skips work on the strength of "we got here before".

No network, no Lean: the memoised callables are counters.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.run_minif2f import Checkpoint  # noqa: E402


def _samples(*texts):
    return [SimpleNamespace(text=t, score=1.0, source="x", thinking=None)
            for t in texts]


def test_disabled_checkpoint_writes_nothing(tmp_path):
    p = tmp_path / "c.json"
    ck = Checkpoint(p, enabled=False)
    calls = []
    ck.llm("prompt", 3, "initial", lambda: (calls.append(1), _samples("a"))[1])
    ck.verify("src", "import Mathlib", lambda: (calls.append(1), ("p", True, ""))[1])
    assert len(calls) == 2
    assert not p.exists()
    assert ck.hits == 0


def test_llm_replays_instead_of_resampling(tmp_path):
    p = tmp_path / "c.json"
    calls = []

    def draw():
        calls.append(1)
        return _samples("proof one", "proof two")

    first = Checkpoint(p, enabled=True)
    got = first.llm("prompt", 2, "initial", draw)
    assert [s.text for s in got] == ["proof one", "proof two"]

    # A new process — the cell died and the chain restarted it.
    second = Checkpoint(p, enabled=True)
    again = second.llm("prompt", 2, "initial", draw)
    assert [s.text for s in again] == ["proof one", "proof two"]
    assert len(calls) == 1, "the restart re-billed the model"
    assert second.hits == 1
    assert all(s.source == "checkpoint" for s in again)


def test_verify_replays_instead_of_recompiling(tmp_path):
    p = tmp_path / "c.json"
    calls = []

    def compile_it():
        calls.append(1)
        return ("the proof", False, "error: unsolved goals")

    Checkpoint(p, enabled=True).verify("src", "import Mathlib", compile_it)
    proof, ok, errs = Checkpoint(p, enabled=True).verify(
        "src", "import Mathlib", compile_it)
    assert (proof, ok, errs) == ("the proof", False, "error: unsolved goals")
    assert len(calls) == 1, "the restart re-ran a 900s Lean compile"


def test_verdict_is_keyed_by_imports_too(tmp_path):
    """The same proof under different imports is a different question.

    Keyed by source alone, an import refresh would inherit the verdict
    from before the refresh — the exact stale-cache bug that makes a
    resumed run diverge from an uninterrupted one.
    """
    p = tmp_path / "c.json"
    calls = []

    def compile_it():
        calls.append(1)
        return ("proof", len(calls) > 1, "")

    ck = Checkpoint(p, enabled=True)
    ck.verify("same source", "import Mathlib.Data.Finset.Basic", compile_it)
    ck.verify("same source", "import Mathlib.NumberTheory.Divisors", compile_it)
    assert len(calls) == 2, "different import sets shared one verdict"


def test_distinct_prompts_do_not_collide(tmp_path):
    p = tmp_path / "c.json"
    ck = Checkpoint(p, enabled=True)
    a = ck.llm("prompt A", 1, "initial", lambda: _samples("from A"))
    b = ck.llm("prompt B", 1, "initial", lambda: _samples("from B"))
    assert a[0].text == "from A" and b[0].text == "from B"
    # Same prompt, different round tag — a correction is not the sample.
    c = ck.llm("prompt A", 1, "correct1", lambda: _samples("from A round 1"))
    assert c[0].text == "from A round 1"


def test_truncated_checkpoint_does_not_kill_the_run(tmp_path):
    """Killed mid-write: worthless, but it must not raise."""
    p = tmp_path / "c.json"
    p.write_text('{"llm": {"k": ["hal', encoding="utf-8")
    ck = Checkpoint(p, enabled=True)
    out = ck.llm("prompt", 1, "initial", lambda: _samples("recomputed"))
    assert out[0].text == "recomputed"


def test_write_is_atomic(tmp_path):
    """The live file is never a half-written one — .tmp then replace."""
    p = tmp_path / "c.json"
    ck = Checkpoint(p, enabled=True)
    ck.llm("prompt", 1, "initial", lambda: _samples("x"))
    assert p.exists()
    assert not p.with_suffix(".tmp").exists()
    json.loads(p.read_text(encoding="utf-8"))  # parses


def test_identical_samples_get_independent_correction_draws(tmp_path):
    """Two samples with the SAME text must still draw their own fixes.

    Observed live: `theorem placeholder : True := trivial` came back as
    two separate samples on the amc12a_2003_p23 baseline. Their
    correction prompts are then byte-identical, so a memo key built from
    the prompt alone makes sample 1 replay sample 0's correction and the
    second draw never happens — k=3 quietly becomes fewer.

    The sample index in the meta tag is what keeps them apart.
    """
    p = tmp_path / "c.json"
    ck = Checkpoint(p, enabled=True)
    prompt = "fix this:\ntheorem placeholder : True := trivial"
    draws = iter(["fix A", "fix B"])
    a = ck.llm(prompt, 1, "s0r1", lambda: _samples(next(draws)))
    b = ck.llm(prompt, 1, "s1r1", lambda: _samples(next(draws)))
    assert a[0].text == "fix A"
    assert b[0].text == "fix B", "sample 1 replayed sample 0's correction"
    assert ck.hits == 0
