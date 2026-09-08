"""Sum results/usage_log.jsonl into dollars (Anthropic pricing).

Usage: python scripts/spend.py [--since 2026-07-06T00:00:00]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# $/1M tokens: (input, output, cache_read)
#
# NOTE on the OpenAI rows: `output_tokens` there is the SDK's
# `completion_tokens`, which ALREADY INCLUDES hidden reasoning tokens
# (`completion_tokens_details.reasoning_tokens` is a broken-out subset,
# logged separately for diagnosis only). So reasoning must NOT be added
# again here — doing so double-counts the dominant cost term.
#
# The gpt-5.6 prices below are UNVERIFIED against OpenAI's current
# pricing page; confirm before quoting a figure anywhere it matters.
PRICES = {
    "claude-opus-4-8": (5.0, 25.0, 0.5),
    "claude-opus-4-7": (5.0, 25.0, 0.5),
    "claude-haiku-4-5": (1.0, 5.0, 0.1),
    "gpt-5.6-sol": (5.0, 30.0, 0.5),
    "gpt-5.6-terra": (1.25, 10.0, 0.125),
    "gpt-5.6-luna": (0.5, 4.0, 0.05),
}


def price_for(model: str) -> tuple[float, float, float]:
    for prefix, p in PRICES.items():
        if model.startswith(prefix):
            return p
    return (5.0, 25.0, 0.5)  # assume Opus-tier if unknown


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    ap.add_argument("--until", default=None,
                    help="ISO timestamp; rows at or after it are skipped")
    ap.add_argument("--run-id", default=None,
                    help="only rows tagged with this run (see --by-run)")
    ap.add_argument("--by-run", action="store_true",
                    help="one line per run_id instead of a single total")
    args = ap.parse_args()

    path = ROOT / "results" / "usage_log.jsonl"
    if not path.exists():
        print("no usage log yet")
        return 0
    # calls | in | out | cache_read | cost, keyed by run (None = untagged)
    tally: dict[str | None, list] = {}
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if args.since and (r.get("ts") or "") < args.since:
            continue
        if args.until and (r.get("ts") or "") >= args.until:
            continue
        # A row with NO token counts is not a billed call. Policy doubles in
        # the unit suite wrote 722 such rows into this ledger before the log
        # path became redirectable (2026-08-23); counting them inflated the
        # call count by 23% while contributing nothing to cost.
        if r.get("input_tokens") is None and r.get("output_tokens") is None:
            skipped += 1
            continue
        run = r.get("run_id")
        if args.run_id and run != args.run_id:
            continue
        pi, po, pc = price_for(r.get("model") or "")
        i = r.get("input_tokens") or 0
        o = r.get("output_tokens") or 0
        c = r.get("cache_read_input_tokens") or 0
        acc = tally.setdefault(run if args.by_run else "", [0, 0, 0, 0, 0.0])
        acc[0] += 1
        acc[1] += i; acc[2] += o; acc[3] += c
        acc[4] += (i * pi + o * po + c * pc) / 1e6

    def _line(label, a):
        head = f"{label:28} " if label else ""
        return (f"{head}{a[0]} calls | in {a[1]:,} out {a[2]:,} "
                f"cache_read {a[3]:,} | est. cost ${a[4]:.2f}")

    if args.by_run:
        for run in sorted(tally, key=lambda k: (k is None, k or "")):
            print(_line(run or "(untagged)", tally[run]))
        tot = [sum(t[j] for t in tally.values()) for j in range(5)]
        print(_line("TOTAL", tot))
    else:
        print(_line("", tally.get("", [0, 0, 0, 0, 0.0])))
    if skipped:
        print(f"({skipped} rows with no token counts skipped — "
              f"pre-2026-08-23 test fixtures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
