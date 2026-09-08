"""Replay log for a pipeline cell — resume a 16-hour run where it stopped.

The three PutnamBench pipeline solves took 8.8, 15.3 and 16.1 hours. A
stop at hour 12 currently costs all twelve hours: the sketch, every
repair round, every kernel-proved lemma, and every Lean compile are
recomputed from scratch. On a 7.7 GB host where `import Mathlib`
compiles were measured at 900 s, that is the difference between a
campaign and a wish.

Why an ORDINAL log rather than a content-keyed cache
----------------------------------------------------
`run_minif2f`'s `Checkpoint` keys purely on content, which is right
there: the same proof under the same imports is the same question.

The DAG loop breaks that assumption. `--sketch-attempts 3` deliberately
re-issues the SAME sketch prompt hoping for a DIFFERENT sample, and the
repair loop re-asks after an empty response for the same reason. Key
those on content and attempt 2 replays attempt 1's sketch — the run
would silently collapse to one attempt and still report three.

So each call is recorded at its ORDINAL within its kind, together with
a hash of its inputs. On resume the Nth call of a kind replays the Nth
recorded result, but ONLY if its inputs still hash the same. The moment
the trajectory diverges — a different prompt at position N, because an
earlier result differed — the log is truncated from that point and
everything after it is computed live.

That gives the property that matters: a resumed cell follows the path
an uninterrupted one would, or it stops replaying. It never blends a
recorded past with a different present.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable


class ReplayLog:
    """Ordinal-indexed record of one cell's expensive calls."""

    def __init__(self, path: Path, enabled: bool):
        self.path = path
        self.enabled = enabled
        self.hits = 0
        self.diverged_at: str | None = None
        # {kind: {ordinal_str: {"key": sha, "value": <json>}}}
        self.data: dict[str, dict[str, Any]] = {}
        self._counters: dict[str, int] = {}
        if enabled and path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data = {k: v for k, v in loaded.items()
                                 if isinstance(v, dict)}
            except Exception:
                # Truncated by the kill that made us resume. Worth
                # nothing; must not take the run down with it.
                self.data = {}

    @staticmethod
    def _key(parts: tuple[str, ...]) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(str(p).encode("utf-8", "replace"))
            h.update(b"\x1f")
        return h.hexdigest()

    def _save(self) -> None:
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data), encoding="utf-8")
            tmp.replace(self.path)      # atomic; never a half file
        except Exception:
            pass                        # a log that cannot write is still a run

    def _truncate_from(self, kind: str, ordinal: int) -> None:
        """Drop this entry and every later one for `kind`.

        They belong to a trajectory this run is no longer on.
        """
        bucket = self.data.get(kind) or {}
        for k in [k for k in bucket if k.isdigit() and int(k) >= ordinal]:
            bucket.pop(k, None)

    def step(self, kind: str, key_parts: tuple[str, ...],
             fn: Callable[[], Any],
             record: Callable[[Any], bool] | None = None) -> Any:
        """Return the recorded result for this call, or compute it.

        `record(value) -> bool` vetoes storing a result. Its reason for
        existing: a compile that never ran (lake exiting 0xC0000142 with
        no output) is not a verdict, and recording one would make every
        later resume replay a fabricated failure — permanently. Since
        everything computed after a fabricated verdict was derived from
        it, a vetoed call also TRUNCATES the log from that point: the
        trajectory beyond it belongs to a poisoned run.
        """
        n = self._counters.get(kind, 0)
        self._counters[kind] = n + 1
        if self.enabled:
            entry = (self.data.get(kind) or {}).get(str(n))
            if isinstance(entry, dict):
                if entry.get("key") == self._key(key_parts):
                    self.hits += 1
                    return entry.get("value")
                # Same position, different inputs: the run has diverged.
                if self.diverged_at is None:
                    self.diverged_at = f"{kind}#{n}"
                self._truncate_from(kind, n)
        value = fn()
        if self.enabled:
            if record is not None and not record(value):
                # Not a verdict — never replay it, and drop anything
                # recorded after it on the poisoned trajectory.
                self._truncate_from(kind, n)
                self._save()
                return value
            self.data.setdefault(kind, {})[str(n)] = {
                "key": self._key(key_parts), "value": value}
            self._save()
        return value

    def summary(self) -> dict:
        return {"hits": self.hits, "diverged_at": self.diverged_at,
                "recorded": {k: len(v) for k, v in self.data.items()}}
