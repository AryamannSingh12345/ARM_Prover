"""Phase-1 whole-proof runner: pass@1 on a small miniF2F subset.

For each problem:
  1) Read theorem statement from data/miniF2F/MiniF2F/Test/<name>.lean
  2) Strip `:= by sorry` and ask the cloud-LLM policy for a proof
  3) Substitute the proof, write to a temp file
  4) Verify via the compile-based verifier (Mathlib import)
  5) Log JSONL row to results/<run_id>.jsonl
Resumable: skip ids already in the JSONL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.compile_verify import verify_proof, compile_lean  # noqa: E402
from policy.prompts import (WHOLE_PROOF_SYSTEM, WHOLE_PROOF_SYSTEM_BARE,  # noqa: E402
                            WHOLE_FILE_SYSTEM_VERBATIM)
from policy.vllm_policy import (  # noqa: E402
    make_policy, check_effort_support)

ROOT = Path(__file__).resolve().parents[2]
MINIF2F_TEST = ROOT / "data" / "miniF2F" / "MiniF2F" / "Test"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

_THEOREM_RE = re.compile(
    r"(theorem\s+\w+.*?):=\s*by\s+sorry\s*$",
    re.S,
)


def load_problem(name: str, bench_dir: Path = MINIF2F_TEST) -> tuple[str, str]:
    """Return (statement_header, full_source). Generalized loader: any
    sorry-terminated declaration kind; the file prelude (abbrevs, opens,
    variables, notation) is carried inside the header."""
    from eval.loader import load_problem as _load
    return _load(name, bench_dir)


_OPEN_LINE_RE = re.compile(r"^(?:open|set_option)\s+[^\n]+$", re.M)


def prepend_open_scopes(src: str, header: str) -> str:
    """Carry the problem file's `open …` scopes and `set_option …`
    lines onto the extracted theorem header as `… in` prefixes.

    miniF2F files declare `open BigOperators Real Nat Topology Rat`
    and `set_option maxHeartbeats 0` at file level; dropping them
    breaks the statement's own elaboration:
      - `(10000!)` without `open Nat` fails to parse, surfacing as a
        header-level `expected token` no proof can ever fix
        (amc12_2001_p5, smoke10_v3);
      - heavyweight statements die on the default 200000-heartbeat
        budget (`(deterministic) timeout at whnf` on
        algebra_abpbcpcageq3, smoke10_v4). The wall-clock verify
        timeout still bounds runtime when heartbeats are unlimited.
    Callers must run `infer_imports_from_header` on the BARE header
    first: the open line's namespace words (`Real`, `Nat`, …) would
    otherwise perturb the inference rules for every problem."""
    prefixes = [m.group(0).rstrip() for m in _OPEN_LINE_RE.finditer(src)]
    if not prefixes:
        return header
    return "\n".join(f"{p} in" for p in prefixes) + "\n" + header


def build_prompt(theorem_header: str, style: str = "legacy") -> str:
    """Whole-proof prompt.

    `legacy` is the miniF2F/Goedel-8B contract and is the default so that
    every existing row stays comparable. `bare` removes all strategic
    guidance — see WHOLE_PROOF_SYSTEM_BARE for why a baseline measurement
    must not inherit "prefer one-liner closers" or "one tactic per line".
    The output contract is identical in both, because `extract_proof`
    parses against it.
    """
    if style == "bare":
        return (
            "Theorem to prove (Lean 4, full Mathlib imported):\n\n"
            f"{theorem_header} := by\n"
            "  <FILL THIS IN>\n\n"
            "Reply with ONLY the proof body to substitute for `<FILL THIS IN>`, "
            "indented with two spaces. No English outside Lean comments, "
            "no fences, no theorem keyword."
        )
    return (
        "Theorem to prove (Lean 4 + Mathlib already imported):\n\n"
        f"{theorem_header} := by\n"
        "  <FILL THIS IN>\n\n"
        "Reply with ONLY the proof body to substitute for `<FILL THIS IN>`. "
        "No English, no fences, no theorem keyword. "
        "Each line is one Lean 4 tactic, indented with two spaces. "
        "Try a single-tactic closer first (omega, decide, norm_num, linarith, "
        "nlinarith, ring, simp_all, rfl, tauto, aesop)."
    )


def build_file_prompt(theorem_header: str) -> str:
    """Verbatim mode: ask for a complete file, give nothing but the statement."""
    return (
        "Prove the following theorem in Lean 4.\n\n"
        f"{theorem_header} := by\n  sorry\n\n"
        "Reply with a complete Lean 4 file: your imports, any `open` lines, "
        "the theorem reproduced exactly as above, and a real proof replacing "
        "`sorry`. It is compiled exactly as you write it."
    )


def build_file_correction_prompt(prev_source: str, lean_error: str) -> str:
    """Verbatim-mode correction. The model may revise ANY part of the file,
    imports included — that is what gives it the same live import-fixing
    ability the pipeline has through `refresh_imports_call`.
    """
    return (
        "Your Lean 4 file FAILED to compile. Here is the file you wrote:\n\n"
        f"```lean4\n{prev_source}\n```\n\n"
        f"Lean reported:\n{lean_error[:1500]}\n\n"
        "Reply with a corrected COMPLETE Lean 4 file. You may change any "
        "part of it, including the imports, but the theorem statement must "
        "stay exactly as originally given. It is compiled exactly as you "
        "write it."
    )


class Checkpoint:
    """Content-keyed memo for one (run_id, problem) cell, on disk.

    A cell that dies at minute 40 of 45 used to restart from attempt 1,
    re-paying for every sample and every Lean compile it had already
    finished. This makes both replayable.

    It is a CACHE, not a control-flow change: every entry is keyed by a
    hash of the exact inputs, so a hit returns precisely what the call
    returned before, and a resumed cell takes the path an uninterrupted
    one would. Nothing is skipped on the strength of "we got here
    before" -- that would let a resumed run diverge silently.

    The Lean verdict is keyed by (source, imports) TOGETHER: the same
    proof text under a different import set is a different question, and
    caching it by proof alone would resurrect a stale verdict from
    before an import change.

    Disabled it is inert -- no file read, no file written.
    """

    def __init__(self, path, enabled: bool):
        self.path = path
        self.enabled = enabled
        self.hits = 0
        self.data = {"llm": {}, "verify": {}}
        if enabled and path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                self.data["llm"] = loaded.get("llm") or {}
                self.data["verify"] = loaded.get("verify") or {}
            except Exception:
                # A truncated checkpoint (killed mid-write) is worth
                # nothing, but must never take the run down with it.
                self.data = {"llm": {}, "verify": {}}

    def _save(self) -> None:
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data), encoding="utf-8")
            tmp.replace(self.path)   # atomic: never a half-written file
        except Exception:
            pass                     # a cache that cannot write is still a run

    @staticmethod
    def _key(*parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8", "replace"))
            h.update(b"\x1f")        # unit separator: no key collisions
        return h.hexdigest()

    def llm(self, prompt: str, k: int, meta: str, fn):
        """Cached sample texts. `fn()` returns objects carrying `.text`."""
        key = self._key("llm", prompt, str(k), meta)
        if self.enabled and key in self.data["llm"]:
            self.hits += 1
            return [SimpleNamespace(text=t, score=1.0, source="checkpoint",
                                    thinking=None)
                    for t in self.data["llm"][key]]
        out = fn()
        if self.enabled:
            self.data["llm"][key] = [getattr(x, "text", "") for x in out]
            self._save()
        return out

    def verify(self, source: str, imports: str, fn):
        """Cached (proof, ok, errors) for one Lean compile."""
        key = self._key("verify", source, imports)
        if self.enabled and key in self.data["verify"]:
            self.hits += 1
            row = self.data["verify"][key]
            return row[0], row[1], row[2]
        proof, ok, errors = fn()
        if self.enabled:
            self.data["verify"][key] = [proof, ok, errors]
            self._save()
        return proof, ok, errors


def evaluate_response(text: str, header: str, verify_imports: str,
                      args) -> tuple[str, bool, str]:
    """(artifact compiled, ok, errors) for one model response.

    `body` is the legacy path: extract a proof body and assemble it under
    the known-good header. `verbatim` compiles the model's own file with no
    edit beyond markdown unwrapping, and refuses it outright if it does not
    reproduce the statement.
    """
    if args.response_mode == "verbatim":
        source = unwrap_fence(text)
        if not statement_is_verbatim(header, source):
            return source, False, (
                "statement_mismatch: the file does not reproduce the "
                "benchmark theorem statement verbatim; not compiled")
        res = compile_lean(source, timeout_s=args.verify_timeout)
        return source, res.ok, (res.errors or "")
    proof = extract_proof(text)
    res = verify_proof(header, proof, imports=verify_imports,
                       timeout_s=args.verify_timeout)
    return proof, res.ok, (res.errors or "")


def build_correction_prompt(header: str, prev_proof: str, lean_error: str,
                            style: str = "legacy") -> str:
    """Correction prompt. `bare` drops the one-tactic-per-line requirement,
    which would otherwise forbid the `have … := by` structure a competition
    proof needs — in the correction path as well as the first.
    """
    tail = (
        "Produce a corrected proof. Reply with ONLY the proof body — no "
        "theorem keyword, no English, no markdown fences."
        if style == "bare" else
        "Produce a corrected proof. Reply with ONLY the proof body — no "
        "theorem keyword, no English, no markdown fences. Each line is "
        "one Lean 4 tactic indented with two spaces."
    )
    return (
        f"Your previous Lean 4 proof for the theorem FAILED to compile. "
        f"The header is:\n\n{header} := by\n  <FILL THIS IN>\n\n"
        f"Your previous proof attempt was:\n\n"
        f"```lean4\n{prev_proof}\n```\n\n"
        f"Lean error:\n{lean_error[:1500]}\n\n{tail}"
    )


def extract_proof(text: str) -> str:
    """Extract the proof body from a model response.

    Tolerates several formats:
      - Bare tactic lines (the old prompt's contract): "  omega"
      - One ```lean4 ... ``` fenced block (with or without preamble like
        "### Proof"): pulls the inner content, then strips a leading
        `theorem … := by` so only the tactic body survives.
      - Plain `theorem … := by\n  …` (no fence): strips the preamble.

    Goedel-Prover-V2 emits markdown-wrapped Lean code with a "### Proof"
    header and a `theorem … := by` block inside the fence; the original
    extractor only matched fences at position 0 and otherwise let the
    "###" through to Lean, causing every attempt to fail with
    `unexpected token '#'`. The matcher below extracts the first
    ```lean[4]?``` block anywhere in the response, then peels the
    theorem-declaration preamble inside it.
    """
    if not text:
        return ""
    t = text.strip()

    # 1. Prefer a fenced ```lean[4]?``` block anywhere in the response.
    #    re.DOTALL so the body can span lines.
    m = re.search(r"```(?:lean4?|lean)?\s*\n(.*?)```", t, flags=re.DOTALL)
    if m:
        t = m.group(1).strip()
    else:
        # 2. No fence — strip a bare leading fence if present (legacy
        #    behaviour).
        t = re.sub(r"^```(?:lean4?|lean)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    # 3. Inside the fence (or bare text) the model often re-emits the
    #    whole declaration: `theorem foo (...) : ... := by\n  <body>`.
    #    Strip everything up to and including the first `:= by` (or
    #    `:=` followed by a newline that the model used in lieu of `by`).
    m2 = re.search(r":=\s*by\s*\n", t)
    if m2:
        t = t[m2.end():]
    else:
        # Fallback: strip a leading `:= by` (no newline) or `by`.
        t = re.sub(r"^:=\s*by\s*", "", t)
        t = re.sub(r"^by\s*", "", t)
    t = t.strip("\n")

    # 4. Ensure each non-empty line is indented by at least two spaces —
    #    Lean tactic-mode block requires a non-zero indent under `by`.
    if t and not all((not line.strip()) or line.startswith(" ")
                      for line in t.splitlines()):
        t = "\n".join(
            ("  " + l) if (l.strip() and not l.startswith(" ")) else l
            for l in t.splitlines()
        )
    return t


_FENCE_RE = re.compile(r"^```(?:lean4?|lean)?\s*\n(.*?)\n?```\s*$",
                       re.DOTALL)
_COMMENT_RE = re.compile(r"/-.*?-/", re.DOTALL)


def unwrap_fence(text: str) -> str:
    """Undo markdown transport, and NOTHING else.

    The ONLY transformation applied in `--response-mode verbatim`. A model
    that wraps its whole answer in one ```lean fence is using markdown as an
    envelope, not writing Lean — stripping it decodes the transport rather
    than editing the proof. Anything else the model emits (indentation,
    imports, structure, stray prose) reaches the compiler untouched and
    fails if it is wrong.

    Deliberately anchored: a fence must enclose the WHOLE response. A
    response with prose around a fenced block is left alone, so it fails to
    compile — which is the correct outcome, since the instruction was to
    reply with the file and nothing else.
    """
    t = (text or "").strip()
    m = _FENCE_RE.match(t)
    return m.group(1) if m else t


def statement_is_verbatim(header: str, source: str) -> bool:
    """True iff `source` reproduces the benchmark declarations verbatim.

    The kernel does not protect against a model that proves a DIFFERENT,
    weaker theorem — it faithfully verifies whatever it is handed. In
    verbatim mode the model writes the whole file, including the statement,
    so this is the gate that keeps a solve honest. It implements check 1 of
    the audit protocol.

    Comparison is on whitespace-normalised text with comment blocks removed,
    so reflowing or re-indenting the statement is allowed but changing a
    binder, a hypothesis or the conclusion is not. `abbrev …_solution`
    declarations are part of the statement and are checked with it.
    """
    def norm(s: str) -> str:
        return " ".join(_COMMENT_RE.sub(" ", s).split())

    want, got = norm(header), norm(source)
    if want and want in got:
        return True
    # Fall back to the declaration alone: `header` also carries prelude
    # lines (`open …`, `set_option …`) that a model may legitimately order
    # differently or split with `in`.
    m = re.search(r"\b(theorem|lemma)\b", want)
    return bool(m) and want[m.start():] in got


def already_done(jsonl: Path) -> set[str]:
    if not jsonl.exists():
        return set()
    done: set[str] = set()
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        try:
            done.add(json.loads(line)["id"])
        except Exception:
            pass
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/dev_subset.txt")
    ap.add_argument("--bench-dir", default=None,
                    help="Problem directory (one `<id>.lean` per problem, "
                         "`theorem ... := by sorry`). Relative to repo "
                         "root. Default: miniF2F test dir. E.g. "
                         "data/proofnet/Test or data/putnambench/Test.")
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--k", type=int, default=1, help="candidates per problem")
    ap.add_argument("--run-id", default=time.strftime("phase1_%Y%m%d_%H%M%S"))
    ap.add_argument("--effort",
                    choices=("low", "medium", "high", "xhigh", "max"),
                    default=None,
                    help="Anthropic only: output_config.effort — how deep "
                         "adaptive thinking goes, and with it total token "
                         "spend. Unset (default) sends no field, which the "
                         "API treats as 'high', so an unset run and an "
                         "--effort high run are the SAME run. Raising it "
                         "spends more of max_tokens on thinking: on "
                         "2026-08-26 three mf18 samples returned zero text "
                         "with stop_reason=max_tokens at 16000 AND at "
                         "32000, so max/xhigh make that failure MORE "
                         "likely, not less. Pair a raise with a larger "
                         "--max-tokens (streaming allows up to 128000) or "
                         "--task-budget. Recorded in the result row so an "
                         "effort run is never confused with a default one.")
    ap.add_argument("--resume", action="store_true",
                    help="Replay a cell from where it stopped instead of "
                         "from attempt 1. Samples and Lean verdicts are "
                         "memoised to results/checkpoints/<run-id>__<pid>."
                         "json, keyed by a hash of their exact inputs, so "
                         "a resumed cell takes the path an uninterrupted "
                         "one would and a hit returns what the call "
                         "returned before. Off by default: a campaign "
                         "already in flight must not change behaviour "
                         "mid-run, even to something equivalent.")
    ap.add_argument("--base-url", default=None,
                    help="Override the vLLM endpoint base URL. Falls back "
                         "to $MODAL_VLLM_URL.")
    ap.add_argument("--vllm-chat-mode", action="store_true",
                    help="Use /v1/chat/completions (chat-template applied) "
                         "instead of /v1/completions. REQUIRED for chat-"
                         "tuned prover models like Goedel-Prover-V2-8B; "
                         "no-op for text-completion models like "
                         "BFS-Prover-V2-7B.")
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="Sampling temperature. 0.5 keeps diversity for "
                         "pass@k without going into nonsense territory.")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="Per-sample output token budget. Goedel can "
                         "produce multi-hundred-line proofs; 2048 covers "
                         "most miniF2F-style proofs while keeping cost "
                         "predictable. Raise for self-correction mode.")
    ap.add_argument("--verify-timeout", type=int, default=600,
                    help="Per-attempt verify_proof timeout (seconds). "
                         "On this Windows host `import Mathlib.Tactic` "
                         "alone takes ~180s and full `import Mathlib` "
                         "exceeds 600s; default 600 covers Mathlib."
                         "Tactic comfortably.")
    ap.add_argument("--lean-imports", default=None,
                    help="Override per-problem inferred imports with "
                         "this string (newline-joined import lines). "
                         "Useful when the auto-inferred set falls back "
                         "to full `import Mathlib` (~600s load on "
                         "Windows) for problems where `import "
                         "Mathlib.Tactic` would suffice. Use semicolons "
                         "in CLI to separate multiple imports.")
    ap.add_argument("--self-correct-rounds", type=int, default=0,
                    help="If > 0, after each failed sample feed the "
                         "Lean error back to the LLM and ask for a "
                         "corrected proof. Up to N correction rounds "
                         "per ORIGINAL sample. Total verify calls per "
                         "problem ≤ k * (1 + N). Goedel-V2 published "
                         "numbers include self-correction; +5–7 points "
                         "vs no-correction on MiniF2F.")
    ap.add_argument("--prompt-style", choices=("legacy", "bare"),
                    default="legacy",
                    help="legacy (default): the miniF2F/Goedel contract, "
                         "one tactic per line and prefer one-liner closers "
                         "— keeps existing rows comparable. bare: output "
                         "contract only, no strategic guidance. Use `bare` "
                         "for any bare-model BASELINE measurement; `legacy` "
                         "forbids the `have … := by` structure a competition "
                         "proof needs and would measure the prompt.")
    ap.add_argument("--response-mode", choices=("body", "verbatim"),
                    default="body",
                    help="body (default): the model returns a proof body, "
                         "which is extracted and assembled under our header "
                         "— note `extract_proof` also RE-INDENTS column-0 "
                         "lines, repairing a syntax error the model made. "
                         "verbatim: the model returns a complete file "
                         "(its own imports included) and it is compiled "
                         "exactly as written, with only markdown unwrapping; "
                         "a file that does not reproduce the statement is "
                         "refused uncompiled. Use `verbatim` for a bare-model "
                         "baseline.")
    args = ap.parse_args()
    if args.response_mode == "verbatim":
        system_prompt = WHOLE_FILE_SYSTEM_VERBATIM
    elif args.prompt_style == "bare":
        system_prompt = WHOLE_PROOF_SYSTEM_BARE
    else:
        system_prompt = WHOLE_PROOF_SYSTEM

    subset = (ROOT / args.subset).read_text(encoding="utf-8").splitlines()
    problem_ids = [s.strip() for s in subset if s.strip() and not s.startswith("#")]
    bench_dir = (ROOT / args.bench_dir) if args.bench_dir else MINIF2F_TEST

    out_path = RESULTS / f"{args.run_id}.jsonl"
    # Stamp every usage row this process writes with the run it belongs to,
    # so spend attribution stops being timestamp forensics. Read by
    # policy.vllm_policy.current_run_id at log time.
    os.environ["PROVER_RUN_ID"] = args.run_id
    done = already_done(out_path)
    policy = make_policy(
        args.provider, args.model,
        base_url=args.base_url,
        chat_mode=args.vllm_chat_mode,
    )
    # Same opt-in shape as run_dag: the baseline arm has to be able to
    # run the identical configuration, or the two arms stop being
    # comparable on anything except the scaffold.
    args.effort = check_effort_support(args.model, args.effort)
    if args.effort:
        if hasattr(policy, "effort"):
            policy.effort = args.effort
        else:
            print(f"[warn] --effort ignored: provider "
                  f"'{args.provider}' does not support it")

    solved = 0
    attempted = 0
    # Use the same inferred-import path as the step harness so we don't
    # pay the ~21-minute `import Mathlib` cost per verification on this
    # Windows host. Falls back to `import Mathlib` only when no rule
    # fires. CLI --lean-imports overrides both paths.
    from search.import_inference import infer_imports_from_header
    cli_imports: str | None = None
    if args.lean_imports:
        cli_imports = "\n".join(
            p.strip() for p in args.lean_imports.split(";") if p.strip()
        )
    for pid in problem_ids:
        if pid in done:
            continue
        attempted += 1
        try:
            header, src = load_problem(pid, bench_dir)
        except Exception as e:
            print(f"[{pid}] load failed: {e}", flush=True)
            continue
        # Per-problem imports. CLI wins; else inferred; else fallback.
        # Inference runs on the bare header; opens are attached after.
        inferred = infer_imports_from_header(header)
        if cli_imports:
            verify_imports = cli_imports
            inferred_rules = ["cli"]
        else:
            verify_imports = inferred.imports
            inferred_rules = list(inferred.matched_rules) or ["fallback_mathlib"]
        # opens/prelude already live inside `header` (eval.loader)
        # One checkpoint per (run, problem). Inert unless --resume.
        ckpt = Checkpoint(
            Path("results") / "checkpoints" / f"{args.run_id}__{pid}.json",
            enabled=args.resume)
        t0 = time.time()
        first_prompt = (build_file_prompt(header)
                        if args.response_mode == "verbatim"
                        else build_prompt(header, args.prompt_style))
        samples = ckpt.llm(
            first_prompt, args.k, "initial",
            lambda: policy.sample_topk(
                first_prompt, k=args.k,
                system=system_prompt,
                temperature=args.temperature, max_tokens=args.max_tokens,
            ))
        outcome = "failed"
        proof_used: str | None = None
        last_attempt: str | None = None
        err: str | None = None
        attempts_log: list[dict] = []
        correction_attempts = 0  # bookkeeping
        # `si` is load-bearing, not decoration: two samples can carry
        # IDENTICAL text (observed live — `theorem placeholder : True :=
        # trivial` came back as two separate samples on the
        # amc12a_2003_p23 baseline). Their correction prompts are then
        # identical too, so without the sample index in the memo key
        # sample 1's correction replays sample 0's and the second draw
        # never happens. The verify above is content-keyed on purpose:
        # the same source under the same imports IS the same compile.
        for si, s in enumerate(samples):
            proof, ok, errors = ckpt.verify(
                s.text, verify_imports,
                lambda s=s: evaluate_response(
                    s.text, header, verify_imports, args))
            last_attempt = proof
            attempts_log.append({
                "round": 0,
                "proof_preview": proof[:200],
                "ok": ok,
                "errors": errors[:300] if not ok else None,
            })
            if ok:
                outcome = "solved"
                proof_used = proof
                break
            err = errors[:400]
            # Self-correction loop: ask the LLM to fix this proof given
            # the Lean error. Up to args.self_correct_rounds attempts
            # per ORIGINAL sample. Each correction is one more verify
            # call. The chat-mode policy sends [system, user(original),
            # user(correction)] so Goedel sees the failure context.
            if args.self_correct_rounds > 0:
                history = [proof]
                last_err = errors
                for round_idx in range(1, args.self_correct_rounds + 1):
                    correction_attempts += 1
                    correct_prompt = (
                        build_file_correction_prompt(history[-1], last_err)
                        if args.response_mode == "verbatim" else
                        build_correction_prompt(header, history[-1], last_err,
                                                args.prompt_style)
                    )
                    try:
                        corr_samples = ckpt.llm(
                            correct_prompt, 1, f"s{si}r{round_idx}",
                            lambda: policy.sample_topk(
                                correct_prompt, k=1, system=system_prompt,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                            ))
                    except Exception as e:
                        attempts_log.append({
                            "round": round_idx,
                            "ok": False,
                            "errors": f"policy_error: {type(e).__name__}: {e}"[:300],
                        })
                        break
                    if not corr_samples:
                        break
                    new_proof, ok2, errors2 = ckpt.verify(
                        corr_samples[0].text, verify_imports,
                        lambda: evaluate_response(
                            corr_samples[0].text, header,
                            verify_imports, args))
                    history.append(new_proof)
                    attempts_log.append({
                        "round": round_idx,
                        "proof_preview": new_proof[:200],
                        "ok": ok2,
                        "errors": errors2[:300] if not ok2 else None,
                    })
                    if ok2:
                        outcome = "solved"
                        proof_used = new_proof
                        last_attempt = new_proof
                        break
                    last_err = errors2
                    err = errors2[:400]
                if outcome == "solved":
                    break
        if outcome == "solved":
            solved += 1
        dt = time.time() - t0
        row = {
            "id": pid, "outcome": outcome, "wall_s": round(dt, 1),
            "bench_dir": args.bench_dir or "minif2f",
            "k": args.k, "provider": args.provider, "model": args.model,
            "self_correct_rounds": args.self_correct_rounds,
            # How much of this cell was replayed from a checkpoint
            # rather than recomputed. 0 on an uninterrupted run.
            "checkpoint_hits": ckpt.hits,
            "proof": proof_used, "last_attempt": last_attempt,
            "last_error": err if outcome == "failed" else None,
            "score_source": samples[0].source if samples else None,
            # In verbatim mode the model writes its own imports, so ours are
            # never used — recording them would misrepresent what compiled.
            "verify_imports": (None if args.response_mode == "verbatim"
                               else verify_imports),
            "import_rules": (["model_supplied"]
                             if args.response_mode == "verbatim"
                             else inferred_rules),
            "attempts": attempts_log,
            "correction_attempts": correction_attempts,
            "prompt_style": args.prompt_style,
            "response_mode": args.response_mode,
            "max_tokens": args.max_tokens,
            # None = the API default ("high"), not "no thinking".
            "effort": args.effort,
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{pid}] {outcome} in {dt:.1f}s", flush=True)

    print(f"\n{solved}/{attempted} solved -- log: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
