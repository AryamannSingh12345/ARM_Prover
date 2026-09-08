"""Per-problem Lean imports — configurability + plumbing.

Pins the new --lean-imports / --repl-startup-timeout / IMPORT_PROFILES
contracts that let step-search use minimal, theorem-specific imports
instead of full Mathlib (which takes 21+ min through compile and exceeds
the REPL's 600 s budget on this Windows host — measured 2026-06-23).

Mocks only at the boundaries: the search entry point is faked so we never
touch lake, and `LeanReplStepSession._spawn` is patched out for the
session-level tests.
"""
from __future__ import annotations

import pytest

#: Spawns a real `lake env lean` compile — minutes per test on this
#: host. Measured, not guessed: this file did not finish in 25s.
#: Excluded from the fast suite via `pytest -m "not live"`.
pytestmark = pytest.mark.live


import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------- REPL session: configurable imports + startup timeout ----------


def _patch_spawn(monkeypatch):
    """Make _spawn a no-op so no real subprocess launches in tests."""
    from backend.repl_step import LeanReplStepSession

    monkeypatch.setattr(
        LeanReplStepSession, "_spawn",
        lambda self: setattr(self, "_proc",
                              SimpleNamespace(poll=lambda: None))
                     or setattr(self, "_dead", False),
    )


def test_repl_session_sends_configured_imports_not_hardcoded(monkeypatch):
    """Constructing the session with a custom `imports=` value must
    cause startup() to send that exact string to the REPL — full Mathlib
    must NOT be hard-coded into the startup path."""
    from backend.repl_step import LeanReplStepSession

    _patch_spawn(monkeypatch)
    sent: list[dict] = []

    def fake_send(self, cmd, *, timeout_s=None):
        sent.append(dict(cmd))
        return {"env": 0, "messages": []}

    monkeypatch.setattr(LeanReplStepSession, "_send", fake_send)
    session = LeanReplStepSession(imports="import Mathlib.Data.Nat.GCD.Basic")
    session.startup()
    assert sent == [{"cmd": "import Mathlib.Data.Nat.GCD.Basic"}]
    assert session.imports == "import Mathlib.Data.Nat.GCD.Basic"


def test_repl_session_multiline_imports_are_sent_verbatim(monkeypatch):
    """The CLI parser hands newline-joined import lines; the session must
    forward the multi-line string as a single `cmd` payload (the REPL
    accepts multi-line input)."""
    from backend.repl_step import LeanReplStepSession

    _patch_spawn(monkeypatch)
    sent: list[dict] = []

    def fake_send(self, cmd, *, timeout_s=None):
        sent.append(dict(cmd))
        return {"env": 0, "messages": []}

    monkeypatch.setattr(LeanReplStepSession, "_send", fake_send)
    multi = "import Mathlib.Data.Nat.GCD.Basic\nimport Mathlib.Tactic"
    session = LeanReplStepSession(imports=multi)
    session.startup()
    assert sent == [{"cmd": multi}]


def test_repl_startup_timeout_is_threaded_from_constructor(monkeypatch):
    """The constructor's startup_timeout_s must be the timeout passed
    into _send when a caller invokes startup() without an explicit
    timeout_s — proving the CLI --repl-startup-timeout is honoured all
    the way through to the subprocess call."""
    from backend.repl_step import LeanReplStepSession

    _patch_spawn(monkeypatch)
    captured: dict = {}

    def fake_send(self, cmd, *, timeout_s=None):
        captured["timeout_s"] = timeout_s
        return {"env": 0, "messages": []}

    monkeypatch.setattr(LeanReplStepSession, "_send", fake_send)
    session = LeanReplStepSession(
        imports="import Mathlib.Tactic",
        startup_timeout_s=123.0,
    )
    session.startup()  # no explicit timeout_s — falls back to constructor's
    assert captured["timeout_s"] == 123.0
    assert session.startup_timeout_s == 123.0


def test_repl_startup_explicit_timeout_argument_still_wins(monkeypatch):
    """An explicit startup(timeout_s=...) must override the
    constructor's default. The debug script (and tests) rely on this."""
    from backend.repl_step import LeanReplStepSession

    _patch_spawn(monkeypatch)
    captured: dict = {}

    def fake_send(self, cmd, *, timeout_s=None):
        captured["timeout_s"] = timeout_s
        return {"env": 0, "messages": []}

    monkeypatch.setattr(LeanReplStepSession, "_send", fake_send)
    session = LeanReplStepSession(startup_timeout_s=300.0)
    session.startup(timeout_s=42.0)
    assert captured["timeout_s"] == 42.0

