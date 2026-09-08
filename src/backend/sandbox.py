"""Scrubbed-env helper for any subprocess that runs untrusted (LLM-generated) Lean.

LLM-generated Lean can `#eval IO.getEnv "FOO"`, so API-key env vars must never
leak into the lake subprocess. Deny-list by name prefix — allowlists are brittle
across OSes.
"""
from __future__ import annotations

import os
import re

# Anything matching one of these prefixes is stripped before subprocess launch.
_DENY_PREFIXES = (
    "ANTHROPIC_", "OPENAI_", "GOOGLE_", "GEMINI_", "CLAUDE_",
    "HF_", "HUGGINGFACE_", "AWS_", "AZURE_", "GITHUB_TOKEN",
)
_DENY_RE = re.compile(r"(?i)(api[_-]?key|secret|token|password)")


def scrubbed_env() -> dict[str, str]:
    """Return os.environ with credential-shaped vars removed."""
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        if any(k.startswith(p) for p in _DENY_PREFIXES):
            continue
        if _DENY_RE.search(k):
            continue
        out[k] = v
    return out
