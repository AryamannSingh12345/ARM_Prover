"""One-command launcher: pick a provider, pick a model, run.

The abstraction over `src/eval/run_dag.py` for when you don't want to
remember any flags. It fetches the CURRENT model list live from the
provider's API (nothing hardcoded — new models appear here the day the
provider ships them), lets you pick interactively, applies the
campaign-tuned defaults for that provider, prints the full underlying
command (so the abstraction teaches rather than hides), and launches.

Usage:
  python scripts/run.py claude                 # menu → pick → launch
  python scripts/run.py gpt                    # same, OpenAI models
  python scripts/run.py gpt --list             # just show the menu
  python scripts/run.py gpt --model gpt-5.6-sol      # skip the menu
  python scripts/run.py claude --subset data/dev_subset.txt --bench-dir ""
  python scripts/run.py gpt --repair-rounds 3        # unknown flags
                                                     # forward to run_dag

Defaults target the b5 bare-statement campaign (data/_b5_bare.txt vs
PutnamBench); override --subset/--bench-dir for anything else. Keys come
from the OS keyring (service `prover`/`proofswarm`) or *_API_KEY env
vars — same lookup the policy adapter uses.

Spend safety: results/spend_cap_usd.txt caps ANTHROPIC spend only; the
OpenAI path has no cap — watch the balance yourself.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy.vllm_policy import _get_key  # noqa: E402

_PROVIDER_ALIASES = {
    "gpt": "openai", "openai": "openai", "oai": "openai",
    "claude": "anthropic", "anthropic": "anthropic",
}

# Non-text / special-purpose OpenAI models excluded from the menu.
_OPENAI_EXCLUDE = re.compile(
    r"audio|realtime|tts|transcribe|whisper|image|dall-e|embed|"
    r"moderation|search-api|instruct", re.I)
_OPENAI_INCLUDE_PREFIXES = ("gpt-4", "gpt-5", "o1", "o3", "o4")

# Campaign-tuned run_dag defaults for the "b5 bare runs". Provider-specific: task budgets + continuations are Anthropic
# betas; OpenAI reasoning models bill hidden reasoning inside the
# visible token budget, hence the higher ceiling.
_COMMON = [
    "--abduce-lemmas", "--abduce-mode", "theory",
    "--verify-backend", "compile", "--verify-timeout", "1200",
    "--abduce-theory-rounds", "4",
]
_PER_PROVIDER = {
    "anthropic": ["--max-tokens", "24000", "--task-budget", "20000",
                  "--continuation-rounds", "2"],
    "openai": ["--max-tokens", "32000"],
}


def _fetch_models(provider: str, key: str) -> list[str]:
    """Live model ids from the provider, newest first where the API
    exposes a timestamp. No local model knowledge."""
    if provider == "openai":
        from openai import OpenAI
        models = list(OpenAI(api_key=key).models.list())
        keep = [
            m for m in models
            if m.id.startswith(_OPENAI_INCLUDE_PREFIXES)
            and not _OPENAI_EXCLUDE.search(m.id)
        ]
        keep.sort(key=lambda m: getattr(m, "created", 0), reverse=True)
        return [m.id for m in keep]
    import anthropic
    page = anthropic.Anthropic(api_key=key).models.list(limit=100)
    return [m.id for m in page.data]


def _pick(models: list[str]) -> str:
    for i, mid in enumerate(models, 1):
        print(f"  {i:3d}. {mid}")
    while True:
        raw = input("model # (or q to quit): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            raise SystemExit(0)
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        print(f"pick 1..{len(models)}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Provider → model menu → run_dag, no flags to "
                    "remember. Extra run_dag args go after `--`.")
    ap.add_argument("provider", choices=sorted(_PROVIDER_ALIASES),
                    help="gpt/openai or claude/anthropic")
    ap.add_argument("--list", action="store_true",
                    help="print the model menu and exit")
    ap.add_argument("--model", default=None,
                    help="skip the menu, use this model id")
    ap.add_argument("--subset", default="data/_b5_bare.txt")
    ap.add_argument("--bench-dir", default="data/putnambench/Test",
                    help="problem dir; empty string = run_dag default "
                         "(miniF2F)")
    ap.add_argument("--run-id", default=None,
                    help="default: <subset-stem>_<model>_<HHMM>")
    # Unknown flags forward to run_dag verbatim (REMAINDER is a trap:
    # it swallows the launcher's own flags after the positional).
    args, extra_args = ap.parse_known_args()

    provider = _PROVIDER_ALIASES[args.provider]
    key = _get_key(provider)
    if not key:
        print(f"No {provider} key found (keyring services "
              f"'prover'/'proofswarm', or {provider.upper()}_API_KEY).")
        return 1

    model = args.model
    if model is None or args.list:
        print(f"[{provider}] fetching live model list…")
        try:
            models = _fetch_models(provider, key)
        except Exception as e:
            print(f"model listing failed: {type(e).__name__}: {e}")
            return 1
        if not models:
            print("no models visible to this key")
            return 1
        if args.list:
            for mid in models:
                print(f"  {mid}")
            return 0
        model = _pick(models)

    run_id = args.run_id or "{}_{}_{}".format(
        Path(args.subset).stem.strip("_"),
        re.sub(r"[^A-Za-z0-9]+", "", model)[:24],
        time.strftime("%m%d_%H%M"))

    cmd = [sys.executable, str(ROOT / "src" / "eval" / "run_dag.py"),
           "--subset", args.subset,
           "--provider", provider, "--model", model,
           "--run-id", run_id,
           *_COMMON, *_PER_PROVIDER[provider]]
    if args.bench_dir:
        cmd += ["--bench-dir", args.bench_dir]
    cmd += [a for a in extra_args if a != "--"]

    print("\nlaunching:\n  " + " ".join(cmd[1:]) + "\n")
    if provider == "openai":
        print("NOTE: spend_cap_usd.txt does NOT cap OpenAI spend.\n")
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
