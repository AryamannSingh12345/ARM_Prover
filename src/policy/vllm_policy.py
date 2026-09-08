"""Policy adapter — cloud-LLM stand-in for vLLM on this no-GPU Windows host.

Adapters return:
    sample_topk(prompt, k) -> list[Sample(text, score, source)]

Score CONTRACT (read this before changing an adapter):
    `score` is a NEG-LOG-P-LIKE COST. Lower is better. A caller that
    consumes it as `neg_logp` MUST NOT apply -log to it. Two backends
    produce it:

      - source="logprob"        : score = -mean(token_logprob) over the
                                  completion. Real information signal.
      - source="rank-fallback"  : score = (rank + 1) / k, monotonically
                                  increasing with rank (sample 0 cheapest).
                                  Used when the provider exposes no
                                  per-token logprobs.

    The proposal calls this distinction the "score_source confounder"
    (§VIII risk B); the eval runner logs it per row so ablations are
    interpretable.

Keys read from OS keyring (proofswarm pattern). Never from .env.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import keyring


# ---------- transient-failure retry ------------------------------------------
#
# MEASURED: three long DAG runs on putnam_1963_a2 died to a SINGLE failed
# provider call and lost everything —
#     p1963a2_dag_v2       t=12766s  InternalServerError: 500
#     p1963a2_dag_v3gate   t=18662s  APITimeoutError: Request timed out
# In both cases the run had already spent hours of compile time, had
# kernel-proved lemmas in hand, and wrote NO result row: `run_dag` treats a
# policy exception as a terminal `stage=llm` failure. A 5-hour experiment
# should not be decided by one 500.
#
# This is deliberately NOT `max_retries` on the SDK client. That was set to
# 0 on purpose (see OpenAIPolicy.__init__): a silent SDK retry of a
# multi-minute reasoning call doubles the bill and appears nowhere. Retries
# here are BOUNDED, LOUD, and only for transient classes.
#
# COST NOTE: a provider may bill a call that then 500s, so a retry can be
# charged twice. Bounded at _RETRY_ATTEMPTS total tries; every retry prints
# and is therefore auditable against results/usage_log.jsonl.
_RETRY_ATTEMPTS = int(os.environ.get("LLM_RETRY_ATTEMPTS", "4"))
_RETRY_BASE_S = float(os.environ.get("LLM_RETRY_BASE_S", "5"))
_RETRY_MAX_S = float(os.environ.get("LLM_RETRY_MAX_S", "120"))

#: Exception CLASS NAMES treated as transient. Matched by name rather than
#: by import so this works for the OpenAI SDK, the Anthropic SDK and any
#: vLLM/OpenAI-compatible client without coupling to either package.
_TRANSIENT_EXC_NAMES = frozenset({
    "APITimeoutError",        # request exceeded the client timeout
    "APIConnectionError",     # socket/DNS/TLS failure mid-flight
    "InternalServerError",    # 5xx
    "RateLimitError",         # 429 — retry after backoff
    "ServiceUnavailableError",
    "OverloadedError",        # Anthropic 529
})


def _is_transient(exc: BaseException) -> bool:
    """Is this failure worth retrying?

    Deterministic failures must NOT be retried: a 400 for a rejected
    parameter, a 401 for a bad key, or a 404 for a wrong model will fail
    identically every time, and retrying them turns a clear error into a
    slow one. The parameter-drift net (`_call_with_param_net`) is the
    correct handler for 400s.
    """
    if type(exc).__name__ in _TRANSIENT_EXC_NAMES:
        return True
    code = getattr(exc, "status_code", None)
    if isinstance(code, int) and (code in (408, 429) or code >= 500):
        return True
    return False


def _with_transient_retry(fn, *, what: str):
    """Call `fn()`, retrying transient provider failures with backoff.

    Re-raises the original exception once the budget is spent, so a run
    that genuinely cannot reach the provider still fails — loudly and with
    the provider's own message, not with a wrapper's.
    """
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — re-raised below
            if not _is_transient(exc) or attempt == _RETRY_ATTEMPTS - 1:
                raise
            delay = min(_RETRY_MAX_S, _RETRY_BASE_S * (2 ** attempt))
            delay *= 0.5 + random.random()  # jitter: avoid lockstep retries
            print(f"[policy] TRANSIENT {type(exc).__name__} on {what} "
                  f"(attempt {attempt + 1}/{_RETRY_ATTEMPTS}) — retrying in "
                  f"{delay:.0f}s. {str(exc)[:200]}", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable: retry loop exited without returning")


# ---------- usage-log location and run tagging -------------------------------
#
# Two things every usage row needs and did not have.
#
# 1. A PATH THE TESTS CAN REDIRECT. `_log_usage` wrote to the repo's real
#    `results/usage_log.jsonl` unconditionally, so the unit suite appended
#    fixture rows to the production ledger: 722 of 3075 rows on 2026-08-23
#    came from `gpt-5.5` / `claude-haiku-test` / `gpt-4o-mini` doubles, with
#    null token fields. Cost sums were unaffected (null reads as 0) but
#    `scripts/spend.py`'s CALL COUNT was inflated by 23%, and any
#    timestamp-window attribution that does not filter by model picks them up.
# 2. A RUN TAG. Rows carried only `ts`, so attributing spend to a run meant
#    timestamp forensics against the run's start and end — see the
#    `putnam_1965_a2` cell, whose whole cost line rests on that.
#
# Both are read from the environment rather than passed in: a policy is
# constructed deep inside the runner and the loggers are called from a dozen
# places, so threading a parameter through would touch far more code than the
# problem is worth.

_DEFAULT_USAGE_LOG = (Path(__file__).resolve().parents[2]
                      / "results" / "usage_log.jsonl")


def usage_log_path() -> Path:
    """Where per-call usage rows are appended.

    `PROVER_USAGE_LOG` overrides the default. `tests/conftest.py` sets it
    for every test, so a test run can never touch the real ledger.
    """
    override = os.environ.get("PROVER_USAGE_LOG")
    return Path(override) if override else _DEFAULT_USAGE_LOG


def current_run_id() -> str | None:
    """The run this call belongs to, or None outside a run.

    Set from `PROVER_RUN_ID` by `run_dag` / `run_minif2f` at startup. It
    identifies a RUN, not a problem: a run covering several problems (say
    `hard6_base_k3sc2`) still needs the timestamp order to separate them.
    """
    return os.environ.get("PROVER_RUN_ID") or None


@dataclass(slots=True)
class Sample:
    text: str
    score: float
    source: str  # "logprob" | "rank-fallback" | "rank_fallback"
    # Optional provider-side diagnostics, populated when the provider
    # exposes them. These are critical for debugging GPT-5+/o-series
    # runs where `text` can come back empty because the model consumed
    # the entire max_completion_tokens budget on hidden reasoning tokens.
    finish_reason: str | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    # The model's extended-thinking text (Anthropic thinking blocks),
    # when the provider exposes it. Diagnostics only — never parsed as
    # the answer. None for providers with hidden reasoning (OpenAI
    # o-series) or none at all.
    thinking: str | None = None


def _get_key(service: str) -> str | None:
    return (
        keyring.get_password("prover", service)
        or keyring.get_password("proofswarm", service)
        or os.environ.get(f"{service.upper()}_API_KEY")
    )


# Anthropic model families where `temperature` is REMOVED (400 on send),
# not merely deprecated. Fable/Mythos additionally reject an explicit
# thinking config other than adaptive/omitted.
_ANTHROPIC_NO_TEMPERATURE_PREFIXES = (
    "claude-opus-4-7", "claude-opus-4-8", "claude-fable", "claude-mythos",
    # claude-*-5 added 2026-07-30. INFERRED from the 4.7/4.8 behaviour,
    # not verified (no Anthropic credit to probe with). Getting this wrong
    # is self-correcting: the runtime learned-rejection net drops any
    # param the API 400s on, loudly.
    "claude-opus-5", "claude-sonnet-5",
)
# Models supporting adaptive thinking. On Fable/Mythos thinking is always
# on (omit the param); on Opus 4.6+/Sonnet 4.6 we opt in explicitly —
# proof sketching is exactly the multi-step-reasoning shape it helps.
#
# claude-*-5 added 2026-07-30. This half of the fix matters MORE than the
# temperature list: omitting the thinking opt-in raises no error, so the
# learned-rejection net can never discover it. Before this, a
# claude-opus-5 run would have silently sketched WITHOUT extended
# thinking — a quiet capability loss that would have made any
# flagship-vs-flagship comparison (e.g. against gpt-5.6-sol, whose
# reasoning is always on) invalid.
_ANTHROPIC_ADAPTIVE_THINKING_PREFIXES = (
    "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-opus-5", "claude-sonnet-5",
)
_ANTHROPIC_ALWAYS_THINKING_PREFIXES = ("claude-fable", "claude-mythos")
# Models supporting the Task Budgets beta (server-injected countdown the
# model sees while generating — header task-budgets-2026-03-13).
_ANTHROPIC_TASK_BUDGET_PREFIXES = (
    "claude-opus-4-7", "claude-opus-4-8", "claude-fable", "claude-mythos",
    "claude-sonnet-5",
)
# Models accepting `output_config.effort` (GA — no beta header). Effort
# controls how deep adaptive thinking goes and, with it, total token
# spend. The API default is "high"; omitting the field is the same as
# sending "high", which is why every run before 2026-08-27 was a
# high-effort run whether or not anyone chose that.
_ANTHROPIC_EFFORT_PREFIXES = (
    "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
    "claude-sonnet-4-6", "claude-sonnet-5",
    "claude-opus-5", "claude-fable", "claude-mythos",
)
# `xhigh` sits between `high` and `max` and arrived with Opus 4.7 — the
# 4.6 generation takes low/medium/high/max only and 400s on xhigh.
_ANTHROPIC_XHIGH_EFFORT_PREFIXES = (
    "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-5",
    "claude-opus-5", "claude-fable", "claude-mythos",
)
_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


class AnthropicPolicy:
    def __init__(self, model: str = "claude-opus-4-8"):
        import anthropic
        api_key = _get_key("anthropic")
        if not api_key:
            raise RuntimeError("No anthropic key in keyring or env")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        # Params the API rejected for this model at runtime — learned, not
        # hardcoded. Prefix tables below remain as fast-path hints only.
        self._rejected_params: set[str] = set()
        # Opt-in output-budget features, set by the runner (both off by
        # default so existing runs stay comparable):
        # task_budget > 0 — server-side Task Budget (beta): the model sees
        #   a live token countdown and wraps up gracefully instead of being
        #   guillotined mid-JSON at max_tokens. API minimum is 20000.
        # continuation_rounds > 0 — when a call stops on max_tokens WITH
        #   visible text, replay the partial as a NON-final assistant turn
        #   (last-turn prefill is a 400 on Opus 4.6+) plus a continue
        #   instruction, and splice the texts. Up to N extra billed calls.
        self.task_budget: int = 0
        self.continuation_rounds: int = 0
        # effort — `output_config.effort`, one of _EFFORT_LEVELS. None
        # leaves the field off, which the API treats as "high"; set it
        # explicitly to make the depth of thinking a recorded choice
        # rather than a default nobody wrote down. Raising it BUYS more
        # thinking out of the same max_tokens, so on a model that
        # already truncates it makes empty responses MORE likely, not
        # less — see the empty-retry path in _one_call, which drops to
        # "low" precisely to free output budget.
        self.effort: str | None = None
        # Thinking text of the most recent _one_call (all billed calls
        # behind it — retries/continuations — joined). Read by callers
        # that trace "how the model was thinking"; never fed back into
        # prompts.
        self.last_thinking: str | None = None

    def _build_kwargs(self, *, system: str | None, prompt: str,
                      temperature: float, max_tokens: int) -> dict:
        m = (self.model or "").lower()
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system or "",
            "messages": [{"role": "user", "content": prompt}],
        }
        if not any(m.startswith(p) for p in _ANTHROPIC_NO_TEMPERATURE_PREFIXES):
            kwargs["temperature"] = temperature
        if any(m.startswith(p) for p in _ANTHROPIC_ADAPTIVE_THINKING_PREFIXES):
            kwargs["thinking"] = {"type": "adaptive"}
        # Fable/Mythos: thinking always on — omit the param entirely.
        if (int(getattr(self, "task_budget", 0) or 0) > 0
                and any(m.startswith(p)
                        for p in _ANTHROPIC_TASK_BUDGET_PREFIXES)):
            kwargs["output_config"] = {
                "task_budget": {"type": "tokens",
                                "total": max(20000, int(self.task_budget))}}
            kwargs["betas"] = ["task-budgets-2026-03-13"]
        # Effort shares `output_config` with the task budget, so merge
        # rather than assign — assigning would silently drop whichever
        # of the two was set first.
        eff = getattr(self, "effort", None)
        if eff:
            oc = dict(kwargs.get("output_config") or {})
            oc["effort"] = eff
            kwargs["output_config"] = oc
        # Learned rejections beat the prefix hints: once the API 400s a
        # param for this model, never send it again this process.
        for p in getattr(self, "_rejected_params", ()):
            kwargs.pop(p, None)
        if "output_config" in self._rejected_params:
            kwargs.pop("betas", None)  # betas travels with output_config
        return kwargs

    @staticmethod
    def _text_of(resp) -> str:
        if getattr(resp, "stop_reason", None) == "refusal":
            return ""
        return "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        ).strip()

    @staticmethod
    def _thinking_of(resp) -> str:
        # `redacted_thinking` blocks carry opaque `data`, not text — skip.
        # Attribute-robust: the probe on 2026-07-26 showed a block with
        # type == "thinking" whose `.thinking` was EMPTY in the streamed
        # final message (SDK/API drift) — an entire campaign logged
        # thinking=0 while the model was thinking. Try the known field
        # names; a `summary` may be a list of sub-blocks.
        out: list[str] = []
        for b in getattr(resp, "content", []) or []:
            if getattr(b, "type", "") != "thinking":
                continue
            t = (getattr(b, "thinking", None)
                 or getattr(b, "text", None)
                 or getattr(b, "summary", None) or "")
            if isinstance(t, (list, tuple)):
                t = "\n".join(
                    getattr(x, "text", None) or getattr(x, "thinking", "")
                    or "" for x in t)
            if isinstance(t, str) and t.strip():
                out.append(t.strip())
        return "\n".join(out).strip()

    def _log_usage(self, resp) -> None:
        """Append per-call token usage to results/usage_log.jsonl.

        Every billed call is logged — including the extra call behind an
        empty-text retry — so summing this file gives the run's true API
        spend. Never allowed to break a run.
        """
        try:
            import json as _json
            import time as _time
            from pathlib import Path
            u = getattr(resp, "usage", None)
            content = getattr(resp, "content", []) or []
            row = {
                "ts": _time.strftime("%Y-%m-%dT%H:%M:%S"),
                "model": getattr(resp, "model", self.model),
                "input_tokens": getattr(u, "input_tokens", None),
                "output_tokens": getattr(u, "output_tokens", None),
                "cache_read_input_tokens": getattr(
                    u, "cache_read_input_tokens", None),
                # thinking-visibility diagnostics (b5_bare_v1 ran an
                # entire campaign at thinking=0 with no way to tell
                # whether the API, the adapter, or the model was the
                # cause):
                "stop_reason": getattr(resp, "stop_reason", None),
                "thinking_blocks": sum(
                    1 for b in content
                    if getattr(b, "type", "") == "thinking"),
                "thinking_chars": len(self._thinking_of(resp)),
                "rejected_params": sorted(self._rejected_params) or None,
                "run_id": current_run_id(),
            }
            path = usage_log_path()
            with path.open("a", encoding="utf-8") as fh:
                fh.write(_json.dumps(row) + "\n")
        except Exception:
            pass

    def _check_spend_cap(self) -> None:
        """Refuse to place a call once cumulative logged spend exceeds
        the cap. Cap sources, highest precedence first:
        results/spend_cap_usd.txt (live-editable while a run is going),
        then the PROVER_SPEND_CAP_USD env var. Neither set = no cap.
        Raising here surfaces as failure_stage="llm" with a recognizable
        message, so capped rows are excluded from solve-rate denominators
        like any API failure — and nothing is billed."""
        cap = os.environ.get("PROVER_SPEND_CAP_USD")
        try:
            from pathlib import Path as _P
            override = (_P(__file__).resolve().parents[2]
                        / "results" / "spend_cap_usd.txt")
            if override.exists():
                # utf-8-sig: tolerate a BOM (PowerShell's `-Encoding
                # utf8` writes one; float("﻿80") is a ValueError
                # that killed a run before its first billed call).
                txt = override.read_text(encoding="utf-8-sig").strip()
                if txt:
                    cap = txt
        except Exception:
            pass
        if not cap:
            return
        try:
            import json as _json
            # The SAME ledger this process writes to: a cap that sums a
            # different file cannot see the run it is meant to stop.
            path = usage_log_path()
            cost = 0.0
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    r = _json.loads(line)
                except Exception:
                    continue
                cost += ((r.get("input_tokens") or 0) * 5.0
                         + (r.get("output_tokens") or 0) * 25.0
                         + (r.get("cache_read_input_tokens") or 0) * 0.5) / 1e6
        except FileNotFoundError:
            return
        if cost >= float(cap):
            raise RuntimeError(
                f"SPEND_CAP reached: ${cost:.2f} >= ${float(cap):.2f} "
                "(PROVER_SPEND_CAP_USD)")

    def _stream_final(self, kwargs: dict):
        # Stream + get_final_message: at 16K+ max_tokens (and with adaptive
        # thinking spend) non-streaming requests risk SDK HTTP timeouts.
        # Requests carrying `betas` (Task Budgets) go through the beta
        # namespace; the GA namespace rejects the param.
        api = (self.client.beta.messages if "betas" in kwargs
               else self.client.messages)
        with api.stream(**kwargs) as stream:
            return stream.get_final_message()

    def _call_with_param_net(self, kwargs: dict):
        """One billed call with the parameter-drift safety net.

        For models this adapter's prefix lists don't know about yet: on a
        400 naming a known-optional param, strip it, remember the
        rejection for the rest of the process, and retry once.
        """
        import anthropic as _ant
        try:
            resp = _with_transient_retry(
                lambda: self._stream_final(kwargs),
                what=f"anthropic {self.model}")
        except _ant.BadRequestError as e:
            msg = str(getattr(e, "message", e))
            retry = dict(kwargs)
            if "temperature" in msg:
                retry.pop("temperature", None)
                self._rejected_params.add("temperature")
                dropped = "temperature"
            elif "thinking" in msg:
                retry.pop("thinking", None)
                self._rejected_params.add("thinking")
                dropped = "thinking"
            elif ("output_config" in msg or "task_budget" in msg
                  or "betas" in msg):
                retry.pop("output_config", None)
                retry.pop("betas", None)
                self._rejected_params.add("output_config")
                dropped = "output_config"
            else:
                raise
            # LOUD, once per param: a silent drop here is how an entire
            # run loses extended thinking with no trace (every call in
            # b5_bare_v1 logged thinking=0 and nothing said why).
            print(f"[policy] WARNING: API rejected param '{dropped}' for "
                  f"{self.model} — disabled for the rest of this process. "
                  f"API message: {msg[:200]}", flush=True)
            resp = _with_transient_retry(
                lambda: self._stream_final(retry),
                what=f"anthropic {self.model} (post-param-drop)")
        self._log_usage(resp)
        return resp

    _CONTINUE_INSTRUCTION = (
        "Your previous response hit the output-token limit and was cut off "
        "mid-stream. Continue EXACTLY from where it stopped — the first "
        "character of your reply must directly continue the interrupted "
        "text. Do not repeat anything already written, do not add a "
        "preamble, commentary, or code fences."
    )

    def _one_call(self, kwargs: dict) -> str:
        # Thinking text from every billed call behind this logical call
        # (base + empty-retry + continuations), for diagnostics.
        thinks: list[str] = []

        def _note_thinking(r) -> None:
            th = self._thinking_of(r)
            if th:
                thinks.append(th)

        self._check_spend_cap()
        resp = self._call_with_param_net(kwargs)
        _note_thinking(resp)
        text = self._text_of(resp)
        if (not text
                and getattr(resp, "stop_reason", None) != "refusal"):
            # Empty visible text: adaptive thinking consumed the whole
            # output budget (stop_reason max_tokens — observed on
            # amc12_2000_p20: three empty responses at 16K), OR the
            # model ended its turn with thinking blocks only (observed
            # on repair calls in smoke10_v3: empty repair responses
            # silently killed the repair loop). One retry with double
            # the budget covers both; refusals are final. max() guard:
            # never SHRINK a base budget already above the 64K clamp.
            bigger = dict(kwargs)
            cur = int(kwargs.get("max_tokens", 16000))
            bigger["max_tokens"] = max(cur, min(cur * 2, 64000))
            # ADAPTIVE THINKING EXPANDS TO FILL THE BUDGET. Doubling
            # max_tokens alone does not fix an empty response — it hands
            # `{"type": "adaptive"}` more room to think in, and the model
            # can still finish the budget without ever opening a text
            # block. MEASURED on mf18 (2026-08-26): `aime_1995_p7` and
            # `imo_2019_p1` each returned three EMPTY sketches, every
            # billed call stop_reason=max_tokens at 16000 then 32000,
            # text=0 and thinking_chars=0 (the platform withholds
            # reasoning text, so an all-thinking response logs as
            # nothing at all). Two problems voided and $7.32 was spent
            # on one cell for zero rows.
            #
            # So bound thinking explicitly on the retry: a fixed
            # budget_tokens strictly below max_tokens forces the model to
            # stop thinking while output budget remains, which is what
            # makes a text block REACHABLE. Half is deliberate — enough
            # thinking to be worth having, enough output to answer with.
            # If the API rejects the bounded form, `_call_with_param_net`
            # drops `thinking` entirely, which also frees the budget for
            # text; either way the retry can produce an answer.
            # HOW to bound it is model-specific. Opus 4.8 REJECTS the
            # `{"type": "enabled", "budget_tokens": N}` form outright:
            #   '"thinking.type.enabled" is not supported for this model.
            #    Use "thinking.type.adaptive" and "output_config.effort"'
            # (measured 2026-08-26). Sending it anyway "works" only by
            # accident — the 400 makes `_call_with_param_net` drop
            # `thinking` for the WHOLE PROCESS, so every later call in
            # the run silently loses thinking. Ask for low effort
            # instead, which keeps adaptive thinking on and shortens it.
            think = bigger.get("thinking")
            if (isinstance(think, dict) and think.get("type") == "adaptive"
                    and "output_config" not in self._rejected_params):
                oc = dict(bigger.get("output_config") or {})
                oc["effort"] = "low"
                bigger["output_config"] = oc
                print(f"[policy] empty response at max_tokens={cur} — "
                      f"retrying at {bigger['max_tokens']} with "
                      f"output_config.effort=low so thinking stops while "
                      f"output budget remains.", flush=True)
            resp = self._call_with_param_net(bigger)
            _note_thinking(resp)
            text = self._text_of(resp)
        # Continuation protocol (opt-in): the call produced text but was
        # truncated at max_tokens. Replay the partial as a NON-final
        # assistant turn (assistant turns elsewhere in the conversation
        # are allowed; only last-turn prefill 400s on Opus 4.6+) followed
        # by a continue instruction, and splice the raw texts before the
        # caller parses. Caveat: _text_of strips whitespace, so a cut
        # exactly at whitespace inside a string literal can lose it —
        # acceptable for sketch JSON, whose seams are structural tokens.
        rounds = int(getattr(self, "continuation_rounds", 0) or 0)
        while (rounds > 0 and text
               and getattr(resp, "stop_reason", None) == "max_tokens"):
            rounds -= 1
            self._check_spend_cap()
            cont = dict(kwargs)
            cont["messages"] = list(kwargs["messages"]) + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": self._CONTINUE_INSTRUCTION},
            ]
            resp = self._call_with_param_net(cont)
            _note_thinking(resp)
            more = self._text_of(resp)
            if not more:
                break
            text = text + more
        self.last_thinking = "\n\n".join(thinks) or None
        return text

    def sample_topk(self, prompt: str, k: int = 8, *, system: str | None = None,
                    temperature: float = 0.8, max_tokens: int = 256) -> list[Sample]:
        # Anthropic's API has no native n=k; we issue k calls and assign a
        # monotonically increasing rank-based cost. score = (i+1)/k satisfies
        # the contract (lower = better, sample 0 cheapest).
        kwargs = self._build_kwargs(system=system, prompt=prompt,
                                    temperature=temperature,
                                    max_tokens=max_tokens)
        out: list[Sample] = []
        for i in range(k):
            text = self._one_call(kwargs)
            out.append(Sample(text=text, score=float(i + 1) / k,
                              source="rank-fallback",
                              thinking=self.last_thinking))
        return out


# Model-prefix list of OpenAI reasoning-class models. These reject the
# legacy `max_tokens` parameter AND reject custom `temperature` AND do
# not expose token-level `logprobs` on the Chat Completions endpoint.
# Conservative: every model whose name starts with one of these prefixes
# is treated as reasoning-class.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _openai_supports_custom_temperature(model: str) -> bool:
    """Whether `chat.completions.create(temperature=X)` is honoured.

    GPT-5 / o-series reasoning models reject any non-default temperature
    with a 400 BadRequest. Other OpenAI models accept the parameter.
    """
    m = (model or "").lower()
    return not any(m.startswith(p) for p in _REASONING_MODEL_PREFIXES)


def _openai_supports_logprobs(model: str) -> bool:
    """Whether `chat.completions.create(logprobs=True)` is honoured AND
    the response choices expose `.logprobs.content` with per-token
    log-probabilities.

    GPT-5 / o-series reasoning models reject the parameter outright.
    Older / non-reasoning OpenAI models populate per-token logprobs.
    """
    m = (model or "").lower()
    return not any(m.startswith(p) for p in _REASONING_MODEL_PREFIXES)


def _openai_token_budget_param(max_tokens: int) -> dict[str, int]:
    """Return the token-budget kwarg dict for OpenAI Chat Completions.

    Newer OpenAI models (o-series, gpt-4o family, gpt-5.x) REJECT the
    legacy `max_tokens` parameter and require `max_completion_tokens`.
    Hard error reported by the SDK:
        "Unsupported parameter: 'max_tokens' is not supported with this
         model. Use 'max_completion_tokens' instead."

    We send `max_completion_tokens` unconditionally — it has been the
    canonical spelling since the o1 family launched in 2024 and is
    accepted by every model relevant to this repo. Wrap as a tiny dict
    so the intent (and any future spelling change) lives in one place.

    NOTE: legacy text-completions / vLLM completions endpoints still use
    `max_tokens`; do NOT apply this helper there (see VLLMRemotePolicy).
    """
    return {"max_completion_tokens": max_tokens}


#: Per-request timeout for OpenAI calls, seconds. The SDK default (600s)
#: is not enough for flagship reasoning models on research-grade goals:
#: run `lrs_sol_legacyARM_store_v2` died on `APITimeoutError` after a
#: theory round that had just emitted a 4.1K-char proposal with 10 defs.
#: Override with OPENAI_REQUEST_TIMEOUT_S.
_OPENAI_TIMEOUT_S = float(os.environ.get("OPENAI_REQUEST_TIMEOUT_S", "1800"))


class OpenAIPolicy:
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        api_key = _get_key("openai")
        if not api_key:
            raise RuntimeError("No openai key in keyring or env")
        # `max_retries=0`: a silent SDK retry of a 2-minute reasoning call
        # doubles the bill without appearing anywhere in the trace. The
        # repair/theory loops already have their own retry semantics.
        self.client = OpenAI(api_key=api_key,
                             timeout=_OPENAI_TIMEOUT_S, max_retries=0)
        self.model = model

    def _log_usage(self, resp) -> None:
        """Append per-call token usage to results/usage_log.jsonl.

        Mirrors `AnthropicPolicy._log_usage` so `scripts/spend.py` sees
        OpenAI rows too. Until this existed the OpenAI path logged
        NOTHING, so a Sol run's cost was unknowable after the fact — the
        visible response text is not a proxy, because reasoning tokens are
        billed as output and never returned (a 177s call in
        p25_sol_legacyARM_store emitted 89 visible characters).
        Never allowed to break a run.
        """
        try:
            import json as _json
            import time as _time
            from pathlib import Path
            u = getattr(resp, "usage", None)
            details = getattr(u, "completion_tokens_details", None) if u else None
            cached = getattr(u, "prompt_tokens_details", None) if u else None
            row = {
                "ts": _time.strftime("%Y-%m-%dT%H:%M:%S"),
                "provider": "openai",
                "model": getattr(resp, "model", self.model),
                # Named to match the Anthropic rows so spend.py can sum
                # one file across providers.
                "input_tokens": getattr(u, "prompt_tokens", None),
                "output_tokens": getattr(u, "completion_tokens", None),
                "cache_read_input_tokens": (
                    getattr(cached, "cached_tokens", None) if cached else None),
                # The number that makes an OpenAI bill explicable: billed
                # as output, invisible in the response.
                "reasoning_tokens": (
                    getattr(details, "reasoning_tokens", None)
                    if details else None),
                "stop_reason": (
                    getattr(resp.choices[0], "finish_reason", None)
                    if getattr(resp, "choices", None) else None),
                "run_id": current_run_id(),
            }
            path = usage_log_path()
            with path.open("a", encoding="utf-8") as fh:
                fh.write(_json.dumps(row) + "\n")
        except Exception:
            pass

    def sample_topk(self, prompt: str, k: int = 8, *, system: str | None = None,
                    temperature: float = 0.8, max_tokens: int = 256) -> list[Sample]:
        # Build kwargs incrementally so we can OMIT parameters that the
        # target model rejects. Reasoning models (gpt-5/o-series) reject
        # both `temperature` and `logprobs`; passing them anyway yields a
        # 400 BadRequest. See _openai_supports_*.
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system or ""},
                {"role": "user", "content": prompt},
            ],
            "n": k,
            # newer models reject `max_tokens` — see _openai_token_budget_param
            **_openai_token_budget_param(max_tokens),
        }
        if _openai_supports_custom_temperature(self.model):
            kwargs["temperature"] = temperature
        if _openai_supports_logprobs(self.model):
            kwargs["logprobs"] = True

        resp = _with_transient_retry(
            lambda: self.client.chat.completions.create(**kwargs),
            what=f"openai {self.model} (k={k})")
        self._log_usage(resp)

        # Pull whole-response usage diagnostics once. On reasoning models
        # `usage.completion_tokens_details.reasoning_tokens` is the
        # single most useful number: a non-zero value with empty visible
        # output is the signature of "max_completion_tokens exhausted by
        # hidden reasoning". Defensive getattr — older SDKs / non-reasoning
        # responses don't have this nested attribute.
        usage = getattr(resp, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None

        out: list[Sample] = []
        for i, choice in enumerate(resp.choices):
            text = (choice.message.content or "").strip()
            lp_obj = getattr(choice, "logprobs", None)
            lp_content = getattr(lp_obj, "content", None) if lp_obj else None
            if lp_content:
                # Real signal: average per-token -log p over the completion.
                lps = [t.logprob for t in lp_content]
                score = -sum(lps) / max(len(lps), 1)
                source = "logprob"
            else:
                # Deterministic rank fallback: cost = 0.1 * sample_index.
                # Lower is better; preserves policy order (index 0 cheapest);
                # k-independent so cross-row comparisons are clean.
                # Tagged "rank_fallback" (underscore) — distinct from the
                # historical Anthropic "rank-fallback" tag, so an ablation
                # can tell GPT-5 reasoning rows from Anthropic rank rows.
                score = 0.1 * i
                source = "rank_fallback"
            finish_reason = getattr(choice, "finish_reason", None)
            out.append(Sample(
                text=text,
                score=float(score),
                source=source,
                finish_reason=finish_reason,
                # `usage` is whole-response; we attach per-Sample so an empty
                # tactic can be diagnosed without correlating across logs.
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
            ))
        return out


class VLLMRemotePolicy:
    """Talks to a remote vLLM OpenAI-compatible endpoint (e.g., Modal).

    Two modes:
      - completions  (default): /v1/completions, raw prompt. Works for
        text-completion-style models like BFS-Prover-V2-7B that take a
        Lean source prefix and continue it.
      - chat (`chat_mode=True`): /v1/chat/completions, the model's chat
        template is auto-applied by vLLM. Required for Goedel-Prover-V2
        and any other Qwen/Llama-chat-tuned prover model; using the
        completions endpoint on these skips the chat template and
        produces low-quality / refusal-style outputs.
    """

    def __init__(self, model: str, base_url: str, api_key: str | None = None,
                 chat_mode: bool = False):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url.rstrip("/") + "/v1",
                             api_key=api_key or "EMPTY")
        self.model = model
        self.chat_mode = chat_mode

    def _sample_topk_completions(
        self, prompt: str, k: int, *, system: str | None,
        temperature: float, max_tokens: int,
    ) -> list[Sample]:
        resp = self.client.completions.create(
            model=self.model,
            prompt=(f"{system}\n\n{prompt}" if system else prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            n=k,
            logprobs=1,
        )
        out: list[Sample] = []
        for i, ch in enumerate(resp.choices):
            text = (ch.text or "").strip()
            if ch.logprobs and ch.logprobs.token_logprobs:
                lps = [lp for lp in ch.logprobs.token_logprobs
                       if lp is not None]
                score = -sum(lps) / max(len(lps), 1)
                src = "logprob"
            else:
                score = float(i + 1) / k
                src = "rank-fallback"
            out.append(Sample(text=text, score=float(score), source=src))
        return out

    def _sample_topk_chat(
        self, prompt: str, k: int, *, system: str | None,
        temperature: float, max_tokens: int,
    ) -> list[Sample]:
        # Build a chat-style message list. The system content is empty
        # when no system prompt is supplied — the chat template adds a
        # default if the model has one.
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = _with_transient_retry(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=k,
                # logprobs work for chat too on vLLM, but the response shape
                # differs (logprobs.content list of token-prob dicts). For
                # simplicity we fall back to rank-based scoring in chat mode
                # — whole-proof pass@k doesn't rely on per-sample logprobs
                # since each candidate is independently verified.
            ),
            what=f"vllm-chat {self.model} (k={k})")
        out: list[Sample] = []
        for i, ch in enumerate(resp.choices):
            text = (ch.message.content or "").strip()
            out.append(Sample(
                text=text,
                score=float(i + 1) / k,
                source="rank-fallback",
            ))
        return out

    def sample_topk(self, prompt: str, k: int = 8, *, system: str | None = None,
                    temperature: float = 0.8, max_tokens: int = 256) -> list[Sample]:
        if self.chat_mode:
            return self._sample_topk_chat(
                prompt, k, system=system,
                temperature=temperature, max_tokens=max_tokens,
            )
        return self._sample_topk_completions(
            prompt, k, system=system,
            temperature=temperature, max_tokens=max_tokens,
        )


def check_effort_support(model: str, effort: str | None) -> str | None:
    """Validate an --effort value against the model, loudly.

    Returns the effort to send, or None to send nothing. A silent drop
    is the trap this project has already been bitten by once: a param
    the API rejects gets removed for the WHOLE process by the learned
    rejection net, so every later call in the run quietly loses it. Say
    so at launch instead, before any money is spent.
    """
    if not effort:
        return None
    m = (model or "").lower()
    if effort not in _EFFORT_LEVELS:
        raise SystemExit(
            f"--effort {effort!r} is not one of {list(_EFFORT_LEVELS)}")
    if not any(m.startswith(p) for p in _ANTHROPIC_EFFORT_PREFIXES):
        print(f"[warn] --effort {effort} ignored: model {model!r} does "
              f"not accept output_config.effort", flush=True)
        return None
    if (effort == "xhigh"
            and not any(m.startswith(p)
                        for p in _ANTHROPIC_XHIGH_EFFORT_PREFIXES)):
        print(f"[warn] --effort xhigh is not available on {model!r} "
              f"(arrived with Opus 4.7); sending 'high' instead.",
              flush=True)
        return "high"
    return effort


def make_policy(provider: str, model: str | None = None,
                base_url: str | None = None, api_key: str | None = None,
                chat_mode: bool = False):
    if provider == "anthropic":
        return AnthropicPolicy(model=model or "claude-opus-4-8")
    if provider == "openai":
        return OpenAIPolicy(model=model or "gpt-4o-mini")
    if provider == "vllm":
        url = base_url or os.environ.get("MODAL_VLLM_URL")
        if not url:
            raise RuntimeError("vllm provider needs base_url or MODAL_VLLM_URL env var")
        key = api_key or _get_key("vllm") or os.environ.get("VLLM_API_KEY")
        return VLLMRemotePolicy(
            model=model or "ByteDance-Seed/BFS-Prover-V2-7B",
            base_url=url, api_key=key,
            chat_mode=chat_mode,
        )
    raise ValueError(f"unsupported provider: {provider}")
