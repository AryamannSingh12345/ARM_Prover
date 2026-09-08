"""Premise scorer.

Given a goal_text + a list of candidate premise (name, signature) pairs, return
a numpy array of relevance scores in [0, 1] (higher = more likely to appear in
the proof of the goal). Implemented as batched prompt-based scoring against the
Claude Haiku adapter, cached to SQLite by (goal_hash, premise_name).

Logs one line per LLM call to logs/premise_scoring.jsonl for later analysis.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "premise_scores.sqlite"
LOG_PATH = ROOT / "logs" / "premise_scoring.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class Premise:
    name: str
    signature: str = ""


# ---- prompt -----------------------------------------------------------------

_SYSTEM = (
    "You rate Lean 4 / Mathlib lemma relevance to a goal. You will receive a goal "
    "and a numbered list of candidate lemmas. For each lemma, output an integer "
    "0..10 estimating how likely that lemma appears in the eventual proof of the "
    "goal. Higher = more relevant. Reply with ONLY a JSON array of integers, "
    "length equal to the number of lemmas. NO prose, NO markdown."
)


def _build_prompt(goal_text: str, batch: Sequence[Premise]) -> str:
    lines = [f"Goal:\n{goal_text.strip()}\n", "Candidate lemmas:"]
    for i, p in enumerate(batch, 1):
        sig = p.signature.strip().replace("\n", " ")[:300]
        lines.append(f"{i}. {p.name} : {sig}" if sig else f"{i}. {p.name}")
    lines.append(f"\nReply with a JSON array of {len(batch)} integers in [0,10].")
    return "\n".join(lines)


_ARRAY_RE = re.compile(r"\[\s*[-\d,.\s]+\]")


def _parse_scores(text: str, n_expected: int) -> list[float] | None:
    m = _ARRAY_RE.search(text)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list) or len(arr) != n_expected:
        return None
    out: list[float] = []
    for v in arr:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        # Clamp + normalize 0..10 -> 0..1.
        x = max(0.0, min(10.0, x)) / 10.0
        out.append(x)
    return out


# ---- cache ------------------------------------------------------------------

def _hash_goal(goal_text: str) -> str:
    norm = " ".join(goal_text.split())  # collapse whitespace
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


def _open_cache() -> sqlite3.Connection:
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS scores "
        "(goal_hash TEXT, premise TEXT, score REAL, ts REAL, "
        " PRIMARY KEY (goal_hash, premise))"
    )
    return conn


# ---- scorer -----------------------------------------------------------------

class PremiseScorer:
    """Caches and batches LLM calls. Default callable is the real Haiku
    adapter; tests inject a fake."""

    def __init__(self, llm_call: Callable[[str], str] | None = None,
                 batch_size: int = 16):
        self._llm_call = llm_call or _default_haiku_call
        self.batch_size = batch_size
        self._conn = _open_cache()

    def score(self, goal_text: str, premises: Sequence[Premise]) -> np.ndarray:
        if not premises:
            return np.zeros(0, dtype=np.float32)
        gh = _hash_goal(goal_text)
        scores = np.full(len(premises), np.nan, dtype=np.float32)
        # 1) cache hits
        cache_hits = 0
        cur = self._conn.execute(
            f"SELECT premise, score FROM scores WHERE goal_hash=? AND premise IN "
            f"({','.join('?'*len(premises))})",
            (gh, *[p.name for p in premises]),
        )
        cached = {row[0]: row[1] for row in cur.fetchall()}
        for i, p in enumerate(premises):
            if p.name in cached:
                scores[i] = cached[p.name]
                cache_hits += 1
        # 2) batch the rest
        todo_idx = [i for i in range(len(premises)) if np.isnan(scores[i])]
        for start in range(0, len(todo_idx), self.batch_size):
            chunk = todo_idx[start:start + self.batch_size]
            batch = [premises[i] for i in chunk]
            prompt = _build_prompt(goal_text, batch)
            t0 = time.time()
            text = self._llm_call(prompt)
            latency_ms = int((time.time() - t0) * 1000)
            parsed = _parse_scores(text, len(batch))
            if parsed is None:
                # On parse failure, score the batch as 0 and log a miss.
                parsed = [0.0] * len(batch)
                _log({"goal_hash": gh, "premises": [p.name for p in batch],
                      "scores": None, "latency_ms": latency_ms,
                      "cache_hits": 0, "parse_error": True, "raw": text[:200]})
            else:
                _log({"goal_hash": gh, "premises": [p.name for p in batch],
                      "scores": parsed, "latency_ms": latency_ms,
                      "cache_hits": 0, "parse_error": False})
            now = time.time()
            with self._conn:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO scores VALUES (?,?,?,?)",
                    [(gh, premises[i].name, s, now) for i, s in zip(chunk, parsed)],
                )
            for i, s in zip(chunk, parsed):
                scores[i] = s
        if cache_hits:
            _log({"goal_hash": gh, "cache_hits": cache_hits,
                  "premises_queried": len(premises) - cache_hits,
                  "kind": "summary"})
        return scores


# ---- defaults ---------------------------------------------------------------

def _default_haiku_call(prompt: str) -> str:
    """Single Anthropic call. Imported lazily so unit tests don't need keys."""
    from .vllm_policy import AnthropicPolicy
    pol = AnthropicPolicy(model=os.environ.get("PROVER_SCORER_MODEL",
                                                "claude-haiku-4-5-20251001"))
    resp = pol.client.messages.create(
        model=pol.model,
        max_tokens=512,
        temperature=0.0,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


def _log(row: dict) -> None:
    row["ts"] = time.time()
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
