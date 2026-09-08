"""The usage ledger must be redirectable, and must say which run wrote it.

Two defects found on 2026-08-23 while asking how per-cell spend is counted:

1. `_log_usage` wrote to the repo's real `results/usage_log.jsonl`
   unconditionally, so the unit suite appended fixture rows to the
   production ledger — 722 of 3075 rows, from `gpt-5.5`,
   `claude-haiku-test` and `gpt-4o-mini` doubles with null token fields.
2. Rows carried no run tag, so attributing spend to a run meant matching
   timestamps against the run's start and end. The `putnam_1965_a2`
   campaign cell rests entirely on that forensics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policy import vllm_policy                                # noqa: E402
from policy.vllm_policy import (AnthropicPolicy, OpenAIPolicy,  # noqa: E402
                                current_run_id, usage_log_path)


class _Usage:
    input_tokens = 11
    output_tokens = 22
    cache_read_input_tokens = 3
    prompt_tokens = 11
    completion_tokens = 22
    completion_tokens_details = None
    prompt_tokens_details = None


class _Resp:
    model = "test-model"
    usage = _Usage()
    content: list = []
    stop_reason = "end_turn"
    choices: list = []


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


# ---- the path is redirectable ---------------------------------------------

def test_conftest_redirects_the_ledger_away_from_the_repo():
    """The autouse fixture is the guard; assert it actually bites."""
    real = Path(vllm_policy._DEFAULT_USAGE_LOG)
    assert usage_log_path() != real
    assert usage_log_path().parent != real.parent


def test_default_path_when_unset(monkeypatch):
    monkeypatch.delenv("PROVER_USAGE_LOG", raising=False)
    assert usage_log_path() == Path(vllm_policy._DEFAULT_USAGE_LOG)


# ---- rows carry the run tag ------------------------------------------------

def test_anthropic_row_is_tagged_and_lands_in_the_redirected_file(monkeypatch):
    monkeypatch.setenv("PROVER_RUN_ID", "p2020a2_dag_v3")
    pol = AnthropicPolicy.__new__(AnthropicPolicy)
    pol.model = "test-model"
    pol._rejected_params = set()
    pol._thinking_of = lambda r: ""
    pol._log_usage(_Resp())
    rows = _rows(usage_log_path())
    assert len(rows) == 1
    assert rows[0]["run_id"] == "p2020a2_dag_v3"
    assert rows[0]["input_tokens"] == 11 and rows[0]["output_tokens"] == 22


def test_openai_row_is_tagged(monkeypatch):
    monkeypatch.setenv("PROVER_RUN_ID", "p2020a2_base_k3sc2")
    pol = OpenAIPolicy.__new__(OpenAIPolicy)
    pol.model = "test-model"
    pol._log_usage(_Resp())
    rows = _rows(usage_log_path())
    assert rows[0]["run_id"] == "p2020a2_base_k3sc2"
    assert rows[0]["provider"] == "openai"


def test_run_id_is_null_outside_a_run(monkeypatch):
    """A row written with no run in progress must say so, not guess."""
    monkeypatch.delenv("PROVER_RUN_ID", raising=False)
    assert current_run_id() is None
    pol = OpenAIPolicy.__new__(OpenAIPolicy)
    pol.model = "test-model"
    pol._log_usage(_Resp())
    assert _rows(usage_log_path())[0]["run_id"] is None


def test_empty_run_id_is_treated_as_absent(monkeypatch):
    monkeypatch.setenv("PROVER_RUN_ID", "")
    assert current_run_id() is None


# ---- the runners set it ----------------------------------------------------

def test_runners_stamp_the_run_id():
    """Both arms must tag their rows, or attribution covers only one arm."""
    for name in ("run_dag.py", "run_minif2f.py"):
        src = (Path(__file__).resolve().parents[1] / "src" / "eval" / name
               ).read_text(encoding="utf-8")
        assert 'os.environ["PROVER_RUN_ID"] = args.run_id' in src, name
