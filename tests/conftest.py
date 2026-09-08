import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def _redirect_usage_log(tmp_path, monkeypatch):
    """No test may append to the real `results/usage_log.jsonl`.

    Policy doubles in the suite call `_log_usage`, which wrote to the repo's
    production ledger unconditionally: 722 of 3075 rows on 2026-08-23 were
    fixture rows (`gpt-5.5`, `claude-haiku-test`, `gpt-4o-mini`, all with null
    token fields). They cost nothing — null reads as 0 — but they inflated
    `scripts/spend.py`'s call count by 23% and are picked up by any
    timestamp-window attribution that does not filter on model.

    Autouse, so a new test cannot forget it. `PROVER_RUN_ID` is cleared for
    the same reason: a stray tag from the ambient shell must not appear in
    rows a test asserts on.
    """
    monkeypatch.setenv("PROVER_USAGE_LOG", str(tmp_path / "usage_log.jsonl"))
    monkeypatch.delenv("PROVER_RUN_ID", raising=False)
    yield
