"""Replay a prover trace file as a human-readable narrative.

Usage:
    python scripts/view_trace.py results/traces/<run-id>/<pid>.trace.jsonl
    python scripts/view_trace.py results/traces/<run-id>          # all problems
    python scripts/view_trace.py <file> --full                    # untruncated
    python scripts/view_trace.py <file> --kind llm_call           # filter
    python scripts/view_trace.py <file> --thinking                # dump every
                                                                  # thinking blob in full

Default output is the same one-line-per-decision narrative the live
run echoes (level 'verbose'). --full dumps each event's long payloads
(prompts, responses, thinking, assembled proofs, Lean errors) in full,
separated by rulers — grep-friendly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diag.tracer import render_event, safe_print  # noqa: E402

# Long-text payload fields worth dumping in --full mode, in print order.
_LONG_FIELDS = ("system", "user", "thinking", "response", "body",
                "errors", "detail", "header")


def _iter_events(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            safe_print(f"[view_trace] skipping unparseable line in {path}")


def _dump_full(ev: dict) -> None:
    for line in render_event(ev, "verbose"):
        safe_print(line)
    for field in _LONG_FIELDS:
        val = ev.get(field)
        if isinstance(val, str) and len(val) > 200:
            safe_print(f"  ┌─ {field} " + "─" * 50)
            for ln in val.splitlines():
                safe_print(f"  │ {ln}")
            safe_print("  └" + "─" * 58)


def show_file(path: Path, args) -> None:
    safe_print(f"\n===== {path} =====")
    for ev in _iter_events(path):
        if args.kind and ev.get("kind") not in args.kind:
            continue
        if args.thinking:
            th = ev.get("thinking")
            if th:
                safe_print(f"\n--- seq {ev.get('seq')} t={ev.get('t')}s "
                           f"role={ev.get('role')} thinking "
                           f"({len(th)} chars) ---")
                safe_print(th)
            continue
        if args.full:
            _dump_full(ev)
        else:
            for line in render_event(ev, "verbose"):
                safe_print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="trace .jsonl file, or a run's trace dir")
    ap.add_argument("--full", action="store_true",
                    help="dump long payloads (prompts/responses/thinking/"
                         "proofs/errors) untruncated")
    ap.add_argument("--kind", action="append", default=None,
                    help="only show events of this kind (repeatable), "
                         "e.g. --kind llm_call --kind verify_call")
    ap.add_argument("--thinking", action="store_true",
                    help="print ONLY the model's thinking blobs, in full")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        safe_print(f"not found: {p}")
        return 1
    files = sorted(p.glob("*.trace.jsonl")) if p.is_dir() else [p]
    if not files:
        safe_print(f"no *.trace.jsonl files in {p}")
        return 1
    for f in files:
        show_file(f, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
