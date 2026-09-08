"""Run tracing — "what is the prover doing, and how is it thinking".

One `Tracer` per problem writes an append-only JSONL trace file where
every event is a timestamped record: LLM calls (FULL system prompt,
user prompt, response text, and the model's extended-thinking text),
verify calls with the Lean errors, error→leaf attribution, every
repair/abduction/decomposition decision, and the final outcome.

The trace file is the ground truth (nothing truncated); the console
echo is a live human-readable narrative of the same events, truncated
to previews. `scripts/view_trace.py` replays a trace file as the same
narrative after the fact (`--full` dumps untruncated payloads).

Design rules:
  - Tracing must NEVER break a run: every write is wrapped, failures
    are swallowed after one stderr warning.
  - The tracer is plumbed as a plain callable `trace(kind, **payload)`
    so `search/proof_dag.py` needs no import of this module (an absent
    tracer is a no-op lambda).
  - Console output must survive a cp1252 Windows console: echo lines
    fall back to ascii-replace on UnicodeEncodeError.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Echo levels: how much of each event the console shows. The trace
# FILE always gets everything regardless of level.
_LEVELS = ("quiet", "info", "verbose")
# Preview lengths per level for long payloads (thinking, responses,
# Lean errors) on the console.
_PREVIEW = {"info": 240, "verbose": 1200}


def _clip(s: object, n: int) -> str:
    s = str(s or "").strip().replace("\r", "")
    if len(s) <= n:
        return s
    return s[:n].rstrip() + f"… (+{len(s) - n} chars)"


def _one_line(s: object, n: int) -> str:
    return " ".join(_clip(s, n).split())


def _kchars(s: object) -> str:
    ln = len(str(s or ""))
    return f"{ln / 1000:.1f}K" if ln >= 1000 else str(ln)


def safe_print(line: str) -> None:
    """Print that survives narrow Windows console encodings (cp1252
    chokes on Lean's unicode: ≤, ℝ, ₀ …)."""
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(enc, errors="replace").decode(enc), flush=True)


# ---------- human rendering ---------------------------------------------------

def render_event(ev: dict, level: str = "info") -> list[str]:
    """Render one trace event as console lines. Returns [] for events
    the given level hides. Shared by the live echo and view_trace.py
    so the live narrative and the replay look identical."""
    if level == "quiet":
        return []
    n = _PREVIEW.get(level, 240)
    kind = ev.get("kind", "?")
    t = ev.get("t")
    stamp = f"{t:7.1f}s" if isinstance(t, (int, float)) else " " * 8
    head = f"{stamp} {kind:<18}"
    out: list[str] = []

    def add(msg: str) -> None:
        out.append(f"{head} {msg}")

    def sub(msg: str) -> None:
        out.append(" " * 9 + f"| {msg}")

    if kind == "problem_start":
        add(f"{ev.get('id')}")
        sub(f"header: {_one_line(ev.get('header'), n)}")
    elif kind == "imports_resolved":
        add(f"{ev.get('source')}: {_one_line(ev.get('imports'), n)}")
    elif kind == "premises":
        names = ev.get("names") or []
        add(f"{len(names)} retrieved ({ev.get('strategy')})")
        if names and level == "verbose":
            sub(", ".join(names[:12]) + ("…" if len(names) > 12 else ""))
    elif kind == "fallback_ladder":
        add(f"source={ev.get('source')}: "
            f"{_one_line(' | '.join(ev.get('tactics') or []), n)}")
    elif kind == "llm_call":
        add(f"role={ev.get('role')} model={ev.get('model')} "
            f"{ev.get('duration_s')}s "
            f"thinking={_kchars(ev.get('thinking'))} "
            f"text={_kchars(ev.get('response'))}")
        if ev.get("thinking"):
            sub(f"thinking> {_one_line(ev.get('thinking'), n)}")
        if level == "verbose" and ev.get("response"):
            sub(f"response> {_one_line(ev.get('response'), n)}")
        if not (ev.get("response") or "").strip():
            sub("response> (EMPTY)")
    elif kind == "sketch_attempt":
        add(f"#{ev.get('attempt')}/{ev.get('of')} ({ev.get('mode')})")
    elif kind == "sketch_parsed":
        add(f"{ev.get('n_haves')} haves: {', '.join(ev.get('ids') or [])}")
        for h in (ev.get("haves") or []) if level == "verbose" else []:
            sub(f"{h.get('id')} : {_one_line(h.get('type'), 100)}  "
                f":= {_one_line(h.get('tactic'), 80)}")
        if ev.get("setup"):
            sub(f"setup: {_one_line('; '.join(ev.get('setup')), n)}")
        sub(f"closer: {_one_line(ev.get('closer'), n)}")
    elif kind == "sketch_parse_error":
        add(f"REJECTED: {ev.get('error')}")
    elif kind == "topo_error":
        add(f"REJECTED: {ev.get('error')}")
    elif kind == "assembled":
        add(f"round {ev.get('round')}: {ev.get('body_lines')} body lines, "
            f"segments: {', '.join(s[0] for s in ev.get('segments') or [])}")
    elif kind == "verify_call":
        ok = ev.get("ok")
        add(f"{'OK' if ok else 'FAIL'} backend={ev.get('backend')} "
            f"{ev.get('duration_s')}s")
        if not ok and ev.get("errors"):
            sub(f"lean> {_one_line(ev.get('errors'), n)}")
    elif kind == "error_attribution":
        segs = ev.get("by_segment") or {}
        add(", ".join(f"{k}({len(v)})" for k, v in segs.items()) or "(none)")
        for k, v in segs.items():
            sub(f"{k}: {_one_line(v[0] if v else '', n)}")
    elif kind == "repair_round":
        add(f"round {ev.get('round')}: broken={ev.get('broken')} "
            f"closer_broken={ev.get('closer_broken')}")
    elif kind == "leaf_closer":
        add(f"{ev.get('id')}: {'FIXED' if ev.get('fixed') else 'no fix'}"
            + (f" -> {_one_line(ev.get('tactic'), n)}"
               if ev.get("fixed") else ""))
    elif kind == "abduce":
        add(f"{ev.get('id')}: {ev.get('status')}"
            + (f" ({ev.get('detail')})" if ev.get("detail") else ""))
        for lem in (ev.get("lemmas") or []):
            sub(f"lemma> {_one_line(lem, n)}")
    elif kind == "abduce_theory":
        rnd = ev.get("round")
        add(f"{ev.get('stage')}"
            + (f" [round {rnd}]" if rnd is not None else "")
            + (f" — {_one_line(ev.get('detail'), n)}"
               if ev.get("detail") else ""))
        for lem in (ev.get("lemmas") or []):
            sub(f"lemma> {_one_line(lem, n)}")
        for d in (ev.get("defs") or []):
            sub(f"def>   {_one_line(d, n)}")
    elif kind == "decompose":
        add(f"{ev.get('id')}: "
            f"{'sub-DAG spliced' if ev.get('ok') else 'no decomposition'}")
    elif kind == "repair_applied":
        add(f"applied to {ev.get('ids')}")
    elif kind == "repair_break":
        add(f"STOP: {ev.get('reason')}")
    elif kind == "outcome":
        add(f"{'SOLVED' if ev.get('verified') else 'FAILED'} "
            f"in {ev.get('wall_s')}s"
            + ("" if ev.get("verified")
               else f" (stage={ev.get('stage')}, "
                    f"detail={_one_line(ev.get('detail'), 160)})"))
    else:
        # Unknown kinds still show up — forward compatibility.
        body = {k: v for k, v in ev.items()
                if k not in ("kind", "t", "ts", "seq", "id")}
        add(_one_line(json.dumps(body, ensure_ascii=False, default=str), n))
    return out


# ---------- tracer ------------------------------------------------------------

class Tracer:
    """Per-problem trace writer + live console narrator.

    Usage:
        tr = Tracer(path, problem_id="amc12_2001_p5", echo="info")
        tr.event("llm_call", role="sketch", response=..., thinking=...)
    Pass `tr.event` anywhere a `trace(kind, **payload)` callable is
    accepted.
    """

    def __init__(self, jsonl_path: str | Path, *, problem_id: str = "",
                 echo: str = "info"):
        if echo not in _LEVELS:
            raise ValueError(f"echo must be one of {_LEVELS}")
        self.path = Path(jsonl_path)
        self.problem_id = problem_id
        self.echo = echo
        self._t0 = time.monotonic()
        self._seq = 0
        self._warned = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def event(self, kind: str, **payload) -> None:
        self._seq += 1
        ev = {
            "seq": self._seq,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "t": round(time.monotonic() - self._t0, 1),
            "id": self.problem_id,
            "kind": kind,
            **payload,
        }
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev, ensure_ascii=False, default=str)
                         + "\n")
        except Exception as e:
            if not self._warned:
                self._warned = True
                safe_print(f"[trace] WARNING: cannot write {self.path}: "
                           f"{type(e).__name__}: {e} — tracing to console "
                           f"only")
        for line in render_event(ev, self.echo):
            safe_print(f"[{self.problem_id}] {line}"
                       if self.problem_id else line)
