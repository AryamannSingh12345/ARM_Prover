"""Phase-2.5 runner: sketch-guided proof-DAG decomposition.

Per problem:
  1) Load theorem header (same path as run_minif2f.py).
  2) Resolve imports (CLI override > inferred > fallback).
  3) Call `attempt_dag_proof` — LLM emits a typed DAG of `have` claims,
     we assemble + verify in a fresh Lean session.
  4) Log a JSONL row with sketch, assembled proof, outcome, and the
     failure_stage / detail so a triage script can categorise misses.

V2+: targeted leaf repair (Lean errors line-mapped to individual
`have` leaves; only broken leaves regenerate), leaf fallback ladder,
optional specialized leaf closer, recursive leaf decomposition, and a
persistent-REPL verify backend. See `search/proof_dag.py`.

CLI mirrors run_minif2f.py: --provider/--model/--base-url/--vllm-chat-mode/
--lean-imports/--verify-timeout/--sketch-attempts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.compile_verify import (  # noqa: E402
    compile_lean, verify_proof, _ERROR_LINE_RE,
)
from diag.tracer import Tracer  # noqa: E402
from eval.run_minif2f import prepend_open_scopes  # noqa: E402
from policy.vllm_policy import (  # noqa: E402
    make_policy, check_effort_support)
from eval.replay import ReplayLog  # noqa: E402
from search.proof_dag import (  # noqa: E402
    attempt_dag_proof, ABDUCE_SYSTEM, DEFAULT_LEAF_FALLBACKS,
    has_header_level_error, PROVE_LEMMA_SYSTEM, REPAIR_SYSTEM,
    SKETCH_SYSTEM, THEORY_SYSTEM,
)
from search.dag import retry_budget as _retry_budget  # noqa: E402
from search.dag.lemma_store import LemmaStore as _LemmaStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MINIF2F_TEST = ROOT / "data" / "miniF2F" / "MiniF2F" / "Test"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

_THEOREM_RE = re.compile(
    r"(theorem\s+\w+.*?):=\s*by\s+sorry\s*$",
    re.S,
)


def load_problem(name: str, bench_dir: Path = MINIF2F_TEST) -> tuple[str, str]:
    # Generalized loader: any sorry-terminated declaration kind, full file
    # prelude (abbrevs/opens/variables/notation) carried in the header.
    from eval.loader import load_problem as _load
    return _load(name, bench_dir)


def infer_split(bench_dir: Path, override: str | None) -> str:
    """Best-effort dataset-split label for lemma-bank provenance. An
    explicit --source-split always wins; otherwise infer from the bench
    dir path. Unknown is the safe default (records default to
    allowed_for_eval=False regardless, so a wrong guess never makes a
    lemma eval-eligible)."""
    if override:
        return override
    low = str(bench_dir).lower()
    for key in ("putnam", "valid", "test", "dev"):
        if key in low:
            return key
    return "unknown"


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
    # Default policy switched to GPT-5.6 Sol 2026-07-29 (user directive).
    # NOTE for any later comparison: every result in results/ produced
    # before this date ran on claude-opus-4-8, so a run that relies on the
    # defaults is NOT policy-matched against those rows. Pass
    # `--provider anthropic --model claude-opus-4-8` to reproduce them.
    # Also note `spend_cap_usd.txt` caps Anthropic spend only — the
    # default provider is now the UNCAPPED one.
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--run-id",
                    default=time.strftime("dag_%Y%m%d_%H%M%S"))
    ap.add_argument("--sketch-attempts", type=int, default=3,
                    help="LLM sketch attempts per problem. On a verify "
                         "failure, the retry prompt feeds the Lean "
                         "error back. Default 3.")
    ap.add_argument("--repair-rounds", type=int, default=2,
                    help="Per sketch: targeted leaf-repair rounds. Lean "
                         "errors are mapped to individual `have` leaves "
                         "by line number; only broken leaves are "
                         "regenerated, verified ones are kept. Default 2.")
    ap.add_argument("--temperature", type=float, default=0.4,
                    help="Sketch sampling temperature. Ignored by models "
                         "that removed the parameter (Opus 4.7+).")
    ap.add_argument("--max-tokens", type=int, default=16000,
                    help="Per-call output budget. 2048 (the old default) "
                         "truncated sketches mid-JSON on hard problems — "
                         "the historical parse-failure mode.")
    ap.add_argument("--task-budget", type=int, default=0,
                    help="Anthropic policies only. Server-side Task Budget "
                         "(beta task-budgets-2026-03-13): the model sees a "
                         "live token countdown while generating and wraps "
                         "up gracefully instead of truncating mid-JSON at "
                         "max_tokens. API minimum 20000 (clamped up). Set "
                         "it BELOW --max-tokens so the soft budget binds "
                         "before the hard cap. 0 = off (default).")
    ap.add_argument("--continuation-rounds", type=int, default=0,
                    help="Anthropic policies only. If a call still stops "
                         "on max_tokens WITH visible text, replay the "
                         "partial as a non-final assistant turn plus a "
                         "'continue exactly from there' instruction, and "
                         "splice the texts — up to N extra billed calls "
                         "per original call. (Last-turn prefill is a 400 "
                         "on Opus 4.6+, hence the replay shape.) "
                         "0 = off (default).")
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
                    help="Replay a pipeline cell from where it stopped "
                         "instead of from the first sketch. LLM calls and "
                         "Lean compiles are recorded to "
                         "results/checkpoints/<run-id>__<pid>.replay.json "
                         "at their ORDINAL within the run, each with a "
                         "hash of its inputs; on resume the Nth call "
                         "replays the Nth result only while the inputs "
                         "still match, and the log is truncated the moment "
                         "the trajectory diverges. Ordinal, not "
                         "content-keyed, BECAUSE --sketch-attempts "
                         "deliberately re-issues the same prompt for a "
                         "different sample — content keying would make "
                         "attempt 2 replay attempt 1 and silently collapse "
                         "three attempts into one. The Putnam pipeline "
                         "solves ran 8.8-16.1 h each; a stop at hour 12 "
                         "otherwise costs twelve hours. Off by default.")
    ap.add_argument("--base-url", default=None,
                    help="Override vLLM endpoint base URL.")
    ap.add_argument("--vllm-chat-mode", action="store_true",
                    help="Use /v1/chat/completions instead of "
                         "/v1/completions. Needed for chat-tuned models "
                         "like Goedel-Prover-V2-8B.")
    ap.add_argument("--lean-imports", default=None,
                    help="Override per-problem inferred imports with "
                         "this string. Use semicolons to separate.")
    ap.add_argument("--leaf-fallback-mode", choices=("llm", "static"),
                    default="llm",
                    help="'llm' (default): the policy proposes up to 6 "
                         "domain-appropriate closer tactics per problem "
                         "(static arithmetic ladder as fallback). "
                         "'static': the legacy hardcoded ladder. Use "
                         "--no-leaf-fallback to disable ladders entirely.")
    ap.add_argument("--import-mode", choices=("llm", "rules"),
                    default="llm",
                    help="How to pick Mathlib imports when --lean-imports "
                         "is not given. 'llm' (default): the policy model "
                         "reads the statement and proposes imports, gated "
                         "by compiling the bare statement, with Lean-error "
                         "feedback and an `import Mathlib` fallback — no "
                         "hardcoded module knowledge. 'rules': the legacy "
                         "regex rule table (miniF2F-tuned).")
    ap.add_argument("--premise-seeded-imports",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Seed the import set from the RETRIEVED PREMISES "
                         "by resolving each name through the declaration "
                         "graph (deterministic; no LLM, no compiles beyond "
                         "one statement gate). Import selection otherwise "
                         "runs before retrieval and from the statement "
                         "alone, so the names BM25 just surfaced never "
                         "inform the imports and the refresh hook has to "
                         "rediscover them one failed compile at a time — "
                         "on p1981a1_dag_v1 the needed module arrived at "
                         "t=13522s of a 14172s run. "
                         "OFF BY DEFAULT because it is only as good as "
                         "retrieval, and `premise_retrieval` is BM25 over "
                         "declaration NAMES ('what is CALLED this'), not "
                         "statements ('what STATES this'). Measured on the "
                         "real graph: putnam_1967_b5 seeds Nat.Choose.Sum "
                         "/ Vandermonde / BigOperators (exactly right), "
                         "while putnam_1981_a1 seeds Coxeter matrices and "
                         "spectral sequences and never surfaces Nat.Prime "
                         "at all. Enable per problem, or better, wait for "
                         "seeding off the STATEMENT index (recognize.py). "
                         "Ignored when --lean-imports pins the set.")
    ap.add_argument("--premise-import-limit", type=int, default=12,
                    help="Cap on premise-seeded imports (each costs "
                         "elaboration time). Modules are taken in "
                         "retrieval-rank order.")
    ap.add_argument("--verify-timeout", type=int, default=600,
                    help="Per-attempt verify_proof timeout (seconds).")
    ap.add_argument("--max-heartbeats", type=int, default=1000000,
                    help="Rewrite the problem header's `set_option "
                         "maxHeartbeats` to this value (5x Lean's "
                         "default 200000). miniF2F headers say 0 = "
                         "unlimited, which lets a hopeless ladder "
                         "tactic (aesop/decide on a wrong leaf) burn "
                         "the whole --verify-timeout and return a bare "
                         "'timeout' instead of a line-mapped Lean "
                         "error — starving the repair loop (observed "
                         "on putnam_1966_b5). A finite cap makes bad "
                         "tactics FAIL FAST WITH AN ERROR the repair "
                         "loop can eat. 0 = keep the file's setting.")
    ap.add_argument("--leaf-fallback", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Wrap each leaf's tactic in Lean's `first | "
                         "(<llm>) | nlinarith | linarith | omega | "
                         "norm_num | positivity | ring_nf | ring | "
                         "field_simp | simp_all | aesop | decide`. "
                         "Lean tries the LLM's suggestion first and "
                         "falls through on failure — catches the "
                         "common 'wrong arithmetic closer' mistake. "
                         "Cannot rescue mathematically false claims. "
                         "Default ON; disable with --no-leaf-fallback.")
    ap.add_argument("--closer-fallback", action="store_true",
                    help="Also wrap the final closer with the "
                         "fallback ladder. Off by default (closer "
                         "tactic typically needs specific hints).")
    ap.add_argument("--leaf-closer-provider", default=None,
                    help="Enable a specialized leaf-closer prover for "
                         "repair rounds (e.g. `vllm` for a Goedel "
                         "endpoint). Broken `have` leaves are posed as "
                         "standalone theorems to this policy before the "
                         "sketch LLM sees them. Off when unset.")
    ap.add_argument("--leaf-closer-model",
                    default="Goedel-LM/Goedel-Prover-V2-8B")
    ap.add_argument("--leaf-closer-base-url", default=None,
                    help="Endpoint for the leaf-closer policy (falls "
                         "back to $MODAL_VLLM_URL for provider vllm).")
    ap.add_argument("--leaf-closer-chat-mode",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Chat endpoint for the leaf closer. Goedel-"
                         "Prover-V2 is chat-tuned — leave ON for it.")
    ap.add_argument("--leaf-closer-k", type=int, default=4,
                    help="Candidates sampled per broken leaf. They are "
                         "combined into `first | (c1) | (c2) | …` so ONE "
                         "assembled-proof compile tests all of them.")
    ap.add_argument("--leaf-closer-max-tokens", type=int, default=2048)
    ap.add_argument("--premises", choices=("none", "bm25", "graph"),
                    default="bm25",
                    help="Retrieve Mathlib declaration names for the "
                         "theorem and splice them into every prompt "
                         "(sketch, retry, repair, sub-decomposition). "
                         "Grounds lemma references in REAL names — the "
                         "dominant hard-problem failure is invented "
                         "ones. Degrades to none if the graph data or "
                         "deps are unavailable.")
    ap.add_argument("--premises-k", type=int, default=24,
                    help="How many retrieved names go into the prompt.")
    ap.add_argument("--decompose-depth", type=int, default=1,
                    help="Max recursive decompositions per stubborn "
                         "leaf (broken twice in a row -> the leaf itself "
                         "is sketched into a sub-DAG of `have`s). "
                         "0 disables recursion.")
    ap.add_argument("--abduce-lemmas", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="RISING-SEA depth-2: a leaf broken twice in a "
                         "row (or dying on a heartbeat timeout — a "
                         "certificate-search blow-up more repair cannot "
                         "fix) triggers one lemma-abduction call: the "
                         "model invents helper lemmas — any domain of "
                         "mathematics, full proofs, no count/length "
                         "caps — which are kernel-gated in ONE compile "
                         "(lemma proofs + new leaf tactic together) "
                         "before being spliced into the header. "
                         "Circular lemmas (goal restated) are rejected. "
                         "Off by default.")
    ap.add_argument("--abduce-mode", choices=("eager", "theory"),
                    default="eager",
                    help="eager (default): invented lemmas must arrive "
                         "WITH proofs, one gated compile. theory: "
                         "deferred middle ground — the model proposes "
                         "lemma STATEMENTS only (novel defs/abstract "
                         "structures welcome, no count/length caps), a "
                         "minimal covering set is kept, a sorry-stubbed "
                         "PROBE compile first checks ALL stuck leaves "
                         "close under the assumed theory, and only then "
                         "are the lemmas proved one by one (kernel-"
                         "gated; proof failures feed back into a "
                         "revised theory). Probe compiles are never "
                         "counted as solves. Needs --abduce-lemmas.")
    ap.add_argument("--abduce-theory-trigger",
                    choices=("stuck", "always"), default="stuck",
                    help="theory mode: WHEN the ARM loop fires. stuck "
                         "(default): only after a leaf has been broken on "
                         "two consecutive repair passes, or has died on a "
                         "heartbeat timeout. That streak needs "
                         "--repair-rounds >= 2 to be reachable at all, so "
                         "shrinking the repair budget silently makes ARM "
                         "unreachable rather than merely rarer. always: "
                         "fires on the first round with any broken leaf. "
                         "Use `always` when ARM ITSELF is the object of "
                         "measurement: in full pipeline runs the loop "
                         "keeps getting "
                         "bypassed, because a good sketch closes the "
                         "problem before any leaf gets stuck. `always` also "
                         "decouples ARM from the repair budget, which is "
                         "what makes a 1x1 ARM ablation possible.")
    ap.add_argument("--abduce-theory-rounds", type=int, default=2,
                    help="theory mode: max theory revisions after "
                         "satisfaction-gate or lemma-proof failures. "
                         "Default 2.")
    ap.add_argument("--abduce-minimize",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="theory mode: greedily drop lemmas not needed "
                         "to satisfy the leaves (one probe compile per "
                         "candidate drop). Default ON.")
    ap.add_argument("--abduce-minimize-verify",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="theory mode (Point 7): after the reference-based "
                         "lemma prune, probe the pruned set; if the leaves "
                         "no longer close (a use mediated by notation/macro/"
                         "def that the textual prune missed), revert to the "
                         "full proved set instead of wasting the round at "
                         "final verify. Costs one extra probe when a drop "
                         "occurred. Off by default (ablatable).")
    ap.add_argument("--store-feedback",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Splice kernel-proved store lemmas into the "
                         "header at the start of each later sketch "
                         "attempt, so the sketcher can BUILD ON them. "
                         "Without it the store only prevents re-proving "
                         "an identical statement and never tells the "
                         "sketcher the lemma exists: on putnam6_easy_v2 "
                         "aux_powerSum_recurrence (the identity the whole "
                         "problem turns on) was proved at t=9372s, "
                         "retained at t=11734s, and never used. Requires "
                         "--abduce-lemma-store. Off by default.")
    ap.add_argument("--debug-dump-lean", default=None, metavar="DIR",
                    help="Keep every .lean file handed to Lean under "
                         "DIR/<run-id>/<problem>/NNN_<tag>.lean, each with a "
                         "JSON sidecar carrying the replay argv, status and "
                         "timings. Off by default. Turn it on when a Lean "
                         "error needs diagnosing: the temp file is deleted "
                         "after each compile and the trace records the "
                         "model's RESPONSE, not the assembled source — so "
                         "without this an error like `unknown tactic` cannot "
                         "be traced to the text that caused it.")
    ap.add_argument("--sketch-mode", choices=("tactics", "blueprint"),
                    default="tactics",
                    help="'tactics' (default, legacy): one call yields "
                         "every `have` WITH its tactic. 'blueprint': the "
                         "model emits ordered claim STATEMENTS only, "
                         "tactics come from the fallback ladder, and the "
                         "plan is checked by a sorry-stubbed sufficiency "
                         "probe BEFORE anything is proved — separating "
                         "'bad tactics' from 'these claims do not imply "
                         "the goal', which the legacy contract cannot "
                         "distinguish. Still one assembled verify per "
                         "round.")
    ap.add_argument("--blueprint-sequential",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="blueprint mode: generate claims ONE AT A TIME, "
                         "each conditioned on those established so far, "
                         "instead of in a single call. Buys adaptivity "
                         "(a later claim can supply what an earlier one "
                         "needed) for k LLM calls instead of 1. Ablation "
                         "lever; off by default.")
    ap.add_argument("--blueprint-rounds", type=int, default=2,
                    help="blueprint mode: regenerations allowed when the "
                         "sufficiency probe rejects the plan, with the "
                         "probe's Lean errors fed back (default 2).")
    ap.add_argument("--recognize-leaves",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Before proposing any theory for a stuck leaf, "
                         "check whether Mathlib ALREADY PROVES it: BM25 "
                         "over declaration STATEMENTS (not names, which "
                         "is all premise retrieval indexes), then the "
                         "kernel decides. Costs ONE compile per stuck "
                         "leaf — every candidate is tried in a single "
                         "`first | ...` — against a theory loop that "
                         "costs many. Inventing a lemma Mathlib already "
                         "carries is the most expensive way to fail. "
                         "Needs --statement-index; off by default.")
    ap.add_argument("--statement-index",
                    default=str(ROOT / "data" / "mathlib_statements.jsonl"),
                    help="statement index for --recognize-leaves. Build "
                         "with scripts/build_statement_index.py (~45s, no "
                         "network/model). REBUILD after any toolchain or "
                         "Mathlib pin change.")
    ap.add_argument("--recognize-k", type=int, default=8,
                    help="candidates retrieved per stuck leaf (default "
                         "8). They all share one compile, so this trades "
                         "prompt-free recall against nothing but a longer "
                         "tactic.")
    ap.add_argument("--reframe-on-abandon",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="When the theory loop has exhausted its "
                         "in-language rounds on a SINGLE stuck leaf, spend "
                         "one more proposal round asking to CHANGE THE "
                         "FRAME: introduce new objects, bridge them to the "
                         "problem, and derive the goal there. Gated twice "
                         "before any proving — a free leverage test drops "
                         "frames whose hardest claim is no easier than the "
                         "target, then one sorry-stubbed compile checks "
                         "the derivation actually closes the goal. The "
                         "result is judged by the ORDINARY theory pipeline "
                         "(circularity guard, quality gate, per-lemma "
                         "kernel proving, all-or-nothing commit), so it "
                         "can supply a proof or stay silent, never assert. "
                         "Off by default.")
    ap.add_argument("--reframe-rounds", type=int, default=2,
                    help="reframe: proposals allowed before giving up, "
                         "with the leverage/probe failure fed back "
                         "(default 2).")
    ap.add_argument("--reframe-build-library",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Stage 3: after a reframing passes its "
                         "sufficiency probe, BUILD a small library of "
                         "elementary facts about its new objects before "
                         "proving its claims. A reframing introduces "
                         "objects nothing is known about, so its bridges "
                         "can be unprovable purely for lack of basic "
                         "lemmas. Each declaration is individually "
                         "kernel-verified; proved lemmas enter the "
                         "proved-lemma bank, so the prove step reuses "
                         "them for ZERO extra compiles. EXPENSIVE: up to "
                         "--reframe-library-decls x "
                         "--reframe-library-attempts real compiles on top "
                         "of the normal budget. Off by default; a failure "
                         "returns nothing and the reframe proceeds "
                         "unchanged.")
    ap.add_argument("--reframe-library-decls", type=int, default=4,
                    help="reframe library: max declarations to plan "
                         "(default 4). Each costs at least one compile.")
    ap.add_argument("--reframe-library-attempts", type=int, default=2,
                    help="reframe library: attempts per declaration, with "
                         "the Lean error fed back (default 2).")
    ap.add_argument("--reframe-leverage-factor", type=float, default=0.9,
                    help="reframe: a frame is rejected unless its hardest "
                         "claim scores below factor x the target's "
                         "difficulty (default 0.9). Costs no compile.")
    ap.add_argument("--refute-first",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Pre-flight: before proving, spend up to "
                         "--refute-attempts LLM calls trying to prove the "
                         "goal FALSE. A refutation is believed ONLY on a "
                         "kernel-verified proof of the negation, is "
                         "reported as outcome='refuted' (never 'solved'), "
                         "and ends that problem. A failed search changes "
                         "nothing. Entirely problem-agnostic — the goal is "
                         "treated as an opaque proposition. Motivation: "
                         "lrs_sol_legacyARM_store_v3 spent 8h22m and 37 "
                         "compiles on a false statement. Off by default.")
    ap.add_argument("--refute-attempts", type=int, default=5,
                    help="Refutation attempts per problem (default 5). "
                         "Each failed attempt feeds its Lean errors back "
                         "into the next. The search stops early if the "
                         "model reports no counterexample. Only used when "
                         "--refute-first is set.")
    ap.add_argument("--refute-max-declines", type=int, default=None,
                    help="How many 'no counterexample' refusals to tolerate "
                         "before ending the refutation search. Default: the "
                         "full --refute-attempts budget, so a reluctant "
                         "model never short-circuits it — each refusal earns "
                         "escalating push-back (search->construct, then "
                         "commit-and-let-the-kernel-judge). Lower it to trade "
                         "coverage for wall time: a refusal costs one LLM "
                         "call, a committed-but-wrong construction costs a "
                         "full compile.")
    ap.add_argument("--abduce-lemma-store",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="theory mode: keep kernel-accepted lemmas in a "
                         "PER-PROBLEM, in-memory store so a theory that "
                         "abandons (one lemma unproved) no longer discards "
                         "the lemmas it already proved — they stay in scope "
                         "for the next theory, revision and sketch attempt. "
                         "Nothing is written to disk and the store never "
                         "crosses problems (cross-problem reuse would "
                         "contaminate an ablation). A reused lemma is still "
                         "re-verified by the normal loop, so it can never "
                         "manufacture a solve. Off by default (ablatable).")
    ap.add_argument("--abduce-retry-budget",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="theory mode: budget lemma-proof retries by WHY the "
                         "attempt failed instead of a flat two tries — "
                         "static errors (unknown identifier/tactic, parse) "
                         "get up to 4 with the error fed back; an empty or "
                         "`sorry` response gets 1 and is escalated in the "
                         "revision ledger rather than re-asked (12 of 19 "
                         "failed attempts in p25_2021_legacyARM_0729 were "
                         "that wasted re-ask). Off by default (ablatable).")
    ap.add_argument("--abduce-quality-gate",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="theory mode: after the circularity guard, reject "
                         "(before any compile) a proposed theory whose "
                         "hardest lemma is not meaningfully easier than the "
                         "hardest stuck obligation (a syntactic difficulty "
                         "proxy; see search/difficulty.py). Catches "
                         "non-circular but theorem-sized lemmas that "
                         "preserve complexity. Off by default (ablatable).")
    ap.add_argument("--abduce-quality-factor", type=float, default=0.9,
                    help="Quality-gate threshold: a lemma is rejected when "
                         "its difficulty >= factor * D(hardest obligation). "
                         "Lower = stricter. Default 0.9.")
    ap.add_argument("--causal-attribution",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Point 7: verify line-mapped blame semantically — "
                         "each blamed leaf is probe-compiled STANDALONE in "
                         "its dependency context; a leaf that closes on its "
                         "own is exonerated (blame reroutes to the closer). "
                         "Kills collateral-blame repairs at the cost of one "
                         "probe per blamed leaf per round. Off by default.")
    ap.add_argument("--circularity-mode",
                    choices=("text", "syntactic", "elaborated"),
                    default="text",
                    help="Theory-abduction circularity guard (Point 7a). "
                         "text (default): legacy whitespace-stripped "
                         "conclusion equality. syntactic: α-invariant "
                         "fingerprint — also catches renamed-binder "
                         "restatements. elaborated: syntactic prefilter, "
                         "then a Lean definitional-equality probe "
                         "(`example : Pₐ ↔ P_b := Iff.rfl`) that also "
                         "catches notation / reducible-wrapper restatements "
                         "an undecidable probe is treated as NOT circular.")
    ap.add_argument("--lemma-library", default="results/invented_lemmas.lean",
                    help="Append-only archive of kernel-verified abduced "
                         "lemmas ('the rising sea'). Write-only in this "
                         "version — not auto-loaded into other problems. A "
                         "provenance JSONL sidecar (<path>.jsonl) records "
                         "source split/problem/run/model/commit per lemma "
                         "for later split- and chronology-safe reuse.")
    ap.add_argument("--source-split", default=None,
                    help="Dataset split these problems belong to "
                         "(dev|valid|test|putnam|custom). Recorded in the "
                         "lemma-bank provenance sidecar so a clean eval can "
                         "exclude lemmas derived from the split under test. "
                         "Default: inferred from --bench-dir.")
    ap.add_argument("--trace", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Write a per-problem diagnostic trace to "
                         "results/traces/<run-id>/<pid>.trace.jsonl: "
                         "every LLM call (FULL prompts, response, and "
                         "the model's extended-thinking text), every "
                         "verify with its Lean errors, error-to-leaf "
                         "attribution, and every repair/abduce/"
                         "decompose decision. Replay with "
                         "scripts/view_trace.py. Default ON; "
                         "--no-trace disables.")
    ap.add_argument("--trace-echo", choices=("quiet", "info", "verbose"),
                    default="info",
                    help="Console narration level for trace events. "
                         "'info' (default): one line per decision plus "
                         "a thinking preview. 'verbose': longer "
                         "previews incl. responses and per-have "
                         "sketches. 'quiet': file only, legacy "
                         "console output.")
    ap.add_argument("--verify-backend", choices=("compile", "repl"),
                    default="compile",
                    help="compile: cold `lake env lean` per verify (~3min "
                         "Mathlib import each). repl: persistent Lean REPL "
                         "with per-import-set env cache — repair rounds "
                         "cost seconds; each import set loads once per "
                         "run. REPL-accepted proofs are still confirmed "
                         "by ONE fresh compile (the only oracle) before "
                         "counting as solved. Prefer repl once "
                         "lean/.repl is built.")
    # Phase 3 scaffold (opt-in deterministic leaf solvers). ALL default
    # off; when off, no candidate-generation code executes and a run is
    # byte-identical to legacy. Wiring into the leaf solver is gated
    # behind review — these flags are currently plumbed and recorded only.
    ap.add_argument("--dag-proof-prior", action="store_true",
                    help="(scaffold, off) use the proof-prior index for "
                         "deterministic leaf candidates before the LLM.")
    ap.add_argument("--dag-proof-prior-path", default=None,
                    help="proof-prior JSONL for --dag-proof-prior.")
    ap.add_argument("--dag-template-tactics", action="store_true",
                    help="(scaffold, off) deterministic premise-template "
                         "leaf candidates.")
    ap.add_argument("--dag-dependency-exploration", action="store_true",
                    help="(scaffold, off) graph dependency-exploration "
                         "leaf candidates.")
    ap.add_argument("--dag-llm-local-fallback", action="store_true",
                    help="(scaffold, off) allow the LLM as a local leaf "
                         "solver after deterministic candidates.")
    ap.add_argument("--dag-repair-engine",
                    choices=("legacy", "shadow", "scheduler"),
                    default="legacy",
                    help="Repair-loop engine (conservative migration). "
                         "legacy (default): the current, behaviourally-"
                         "verified loop, unchanged. shadow: legacy makes "
                         "every real decision while the scheduler records "
                         "what it WOULD have planned and logs divergences "
                         "(no Lean, no tokens, no mutation). scheduler: "
                         "the RepairPolicy object AUTHORISES each repair "
                         "stage on the live path; for the default policy "
                         "this is parity-identical to legacy by "
                         "construction. Execution parity on real problems "
                         "is validated by a run before this becomes default.")
    args = ap.parse_args()

    # Execution engine for the live loop: only "scheduler" changes the
    # repair path (policy-authorised stages). "shadow" is observe-only
    # (realised by the EventRecorder wrapper) and RUNS LEGACY here.
    _exec_engine = ("scheduler" if args.dag_repair_engine == "scheduler"
                    else "legacy")

    # Point-3 leaf-solver configuration (providers built below; the
    # portfolio executes as Pass 1.25 of the repair loop when enabled).
    from search.dag.solvers import LeafSolverConfig
    _leaf_cfg = LeafSolverConfig(
        proof_prior=args.dag_proof_prior,
        template_tactics=args.dag_template_tactics,
        dependency_exploration=args.dag_dependency_exploration,
        llm_local_fallback=args.dag_llm_local_fallback,
        proof_prior_path=args.dag_proof_prior_path)
    # Point 3: build the deterministic leaf-candidate portfolio. Empty
    # (and importing nothing) when every flag is off — the default run is
    # byte-identical. `llm_local_fallback` maps to the existing Pass-3
    # LLM repair, so it adds no provider here (recorded for provenance).
    from search.dag.solvers import build_leaf_solvers, SolverContext
    _leaf_solvers = build_leaf_solvers(_leaf_cfg)
    if _leaf_cfg.any_enabled:
        print(f"[portfolio] leaf solvers enabled: "
              f"{[p.name for p in _leaf_solvers] or ['(llm fallback only)']}")

    # Point 4 hook: provability prior for the decomposition-quality gate.
    # Built from the proof-prior index when a path is given — the max
    # observed prior_probability among moves suggested for the goal's
    # feature shape (a closure-confidence proxy). None when the index is
    # absent/empty for that shape; failures are soft (prior contributes
    # nothing). Only consulted when --abduce-quality-gate is on.
    _provability_prior_fn = None
    if args.dag_proof_prior_path:
        from pathlib import Path as _Path
        _pp = _Path(args.dag_proof_prior_path)
        if _pp.exists():
            from search.proof_prior import ProofPriorIndex
            from search.state_features import extract_state_features
            _prior_idx = ProofPriorIndex.load(_pp)

            def _provability_prior_fn(goal_text: str):
                try:
                    feats = extract_state_features(goal_text,
                                                   proof_prefix=[])
                    moves = _prior_idx.suggest(feats, top_k=10)
                    if not moves:
                        return None
                    return max(0.0, min(1.0, max(
                        m.prior_probability for m in moves)))
                except Exception:
                    return None

    # utf-8-sig: tolerate a BOM (PowerShell's `-Encoding utf8` writes
    # one) — a BOM'd first line otherwise becomes an invisible-prefix
    # problem id that fails to load.
    subset = (ROOT / args.subset).read_text(encoding="utf-8-sig").splitlines()
    problem_ids = [s.strip() for s in subset
                    if s.strip() and not s.startswith("#")]
    bench_dir = (ROOT / args.bench_dir) if args.bench_dir else MINIF2F_TEST

    out_path = RESULTS / f"{args.run_id}.jsonl"
    # Stamp every usage row this process writes with the run it belongs to,
    # so spend attribution stops being timestamp forensics. Read by
    # policy.vllm_policy.current_run_id at log time.
    os.environ["PROVER_RUN_ID"] = args.run_id
    done = already_done(out_path)
    # Static lemma-bank provenance (run-level); imports + source_problem
    # are filled in per problem at the attempt call site.
    from search.lemma_bank import mathlib_commit_from_manifest
    _source_split = infer_split(bench_dir, args.source_split)
    _mathlib_commit = mathlib_commit_from_manifest(ROOT / "lean")
    policy = make_policy(
        args.provider, args.model,
        base_url=args.base_url,
        chat_mode=args.vllm_chat_mode,
    )
    # Opt-in anti-truncation features — only the Anthropic adapter
    # declares these attributes; silently ignored elsewhere.
    args.effort = check_effort_support(args.model, args.effort)
    for _feat in ("task_budget", "continuation_rounds", "effort"):
        _val = getattr(args, _feat)
        if _val:
            if hasattr(policy, _feat):
                setattr(policy, _feat, _val)
            else:
                print(f"[warn] --{_feat.replace('_', '-')} ignored: "
                      f"provider '{args.provider}' does not support it")

    # Per-problem tracer, rebound each loop iteration. A mutable holder
    # (not a closure variable) so the LLM/verify wrappers defined once
    # here see the current problem's tracer. _trace() is a no-op when
    # tracing is off or no problem is active.
    tracer_holder: dict = {"tracer": None}

    def _trace(kind: str, **payload) -> None:
        t = tracer_holder["tracer"]
        if t is not None:
            t.event(kind, **payload)

    # Which pipeline stage an LLM call belongs to, recovered from the
    # system prompt identity — avoids threading a role argument through
    # attempt_dag_proof's callable contract.
    _SYSTEM_ROLES = {
        SKETCH_SYSTEM: "sketch",
        REPAIR_SYSTEM: "repair",
        ABDUCE_SYSTEM: "abduce",
        THEORY_SYSTEM: "abduce_theory",
        PROVE_LEMMA_SYSTEM: "prove_lemma",
    }

    # Wrap the policy as a (system, user) -> raw_text callable. Single
    # sample; the DAG harness handles multiple sketch attempts itself.
    # Per-problem replay log, rebound each loop iteration. A mutable
    # holder (not a closure variable) so the wrappers defined once here
    # see the current problem's log. Disabled unless --resume.
    replay_holder: dict = {"log": ReplayLog(Path("nul"), False)}

    def _is_real_verdict(r: dict) -> bool:
        """False when the compile never ran, so it is never replayed.

        A spawn failure is an environment event, not a statement about
        the proof. Recording one would make every subsequent resume
        replay a fabricated failure for the rest of the cell's life.
        """
        return "INFRASTRUCTURE" not in ((r or {}).get("errors") or "")

    def _sketch_llm_call(system: str, user: str) -> str:
        t0 = time.time()

        def _draw() -> dict:
            smp = policy.sample_topk(
                user, k=1, system=system,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            return {"text": smp[0].text if smp else "",
                    "thinking": (getattr(smp[0], "thinking", None)
                                 if smp else None)}

        # Keyed by ordinal + (system, user): a re-issued sketch prompt
        # is a NEW draw on a fresh run and replays as itself on a
        # resumed one.
        rec = replay_holder["log"].step("llm", (system, user), _draw)
        text = (rec or {}).get("text", "")
        thinking = (rec or {}).get("thinking")
        _trace(
            "llm_call",
            role=_SYSTEM_ROLES.get(system, "aux"),
            model=args.model,
            duration_s=round(time.time() - t0, 1),
            system=system, user=user,
            response=text, thinking=thinking,
        )
        return text

    # Optional specialized leaf closer (Goedel-class whole-proof prover).
    leaf_closer_call = None
    if args.leaf_closer_provider:
        from eval.run_minif2f import build_prompt as _leaf_prompt
        from eval.run_minif2f import extract_proof as _extract_proof
        from policy.prompts import WHOLE_PROOF_SYSTEM
        from search.proof_dag import combine_candidates
        leaf_policy = make_policy(
            args.leaf_closer_provider, args.leaf_closer_model,
            base_url=args.leaf_closer_base_url,
            chat_mode=args.leaf_closer_chat_mode,
        )

        def leaf_closer_call(leaf_stmt: str) -> str:
            t0 = time.time()
            samples = leaf_policy.sample_topk(
                _leaf_prompt(leaf_stmt), k=args.leaf_closer_k,
                system=WHOLE_PROOF_SYSTEM,
                temperature=0.7,
                max_tokens=args.leaf_closer_max_tokens,
            )
            cands = [_extract_proof(s.text) for s in samples]
            combined = combine_candidates([c for c in cands if c.strip()])
            _trace(
                "llm_call",
                role="leaf_closer", model=args.leaf_closer_model,
                duration_s=round(time.time() - t0, 1),
                user=leaf_stmt,
                candidates=[c for c in cands if c.strip()],
                response=combined, thinking=None,
            )
            return combined

    # Resolve imports. CLI > inferred > fallback Mathlib.
    from search.import_inference import infer_imports_from_header
    cli_imports: str | None = None
    if args.lean_imports:
        cli_imports = "\n".join(
            p.strip() for p in args.lean_imports.split(";") if p.strip()
        )

    # Optional persistent-REPL verify backend. Lazy startup: the first
    # verify pays the import load; later verifies (and every repair
    # round) reuse the warm env.
    repl_session = None
    if args.verify_backend == "repl":
        from backend.repl_step import LeanReplStepSession
        repl_session = LeanReplStepSession(
            imports="import Mathlib.Tactic",
            startup_timeout_s=900,
        )

    # Statement index for --recognize-leaves. Loaded ONCE for the whole
    # run (~306k declarations, ~25s) and shared across problems. Held in a
    # one-element list so the per-problem closures can read it without a
    # nonlocal. A missing index disables recognition rather than killing
    # the run — it is an assist, never a dependency.
    _statement_index: list = [None]
    if args.recognize_leaves:
        _idx_path = Path(args.statement_index)
        if not _idx_path.exists():
            print(f"[recognize] no statement index at {_idx_path} — "
                  f"recognition DISABLED. Build it with "
                  f"`python scripts/build_statement_index.py`.")
        else:
            from search.recognize import StatementIndex
            _t_idx = time.time()
            _statement_index[0] = StatementIndex.load(_idx_path)
            print(f"[recognize] statement index: "
                  f"{len(_statement_index[0])} declarations "
                  f"({time.time() - _t_idx:.1f}s)")

    solved = 0
    attempted = 0
    trace_dir = RESULTS / "traces" / args.run_id
    for pid in problem_ids:
        if pid in done:
            continue
        attempted += 1
        # Per-problem counter for --debug-dump-lean, so dumps sort in the
        # order Lean saw them.
        _dump_seq = [0]
        tracer_holder["tracer"] = (
            Tracer(trace_dir / f"{pid}.trace.jsonl",
                   problem_id=pid, echo=args.trace_echo)
            if args.trace else None)
        try:
            header, src = load_problem(pid, bench_dir)
        except Exception as e:
            print(f"[{pid}] load failed: {e}", flush=True)
            _trace("load_error", detail=f"{type(e).__name__}: {e}")
            continue

        # Bound tactic elaboration so hopeless tactics fail fast WITH a
        # line-mapped error instead of eating --verify-timeout (see
        # --max-heartbeats help). Rewrite an existing set_option, else
        # prepend one (set_option is legal before `open` in a header).
        if args.max_heartbeats > 0:
            hb = f"set_option maxHeartbeats {args.max_heartbeats}"
            header, n_sub = re.subn(
                r"set_option\s+maxHeartbeats\s+\d+", hb, header)
            if n_sub == 0:
                header = f"{hb}\n\n{header}"
        _trace("problem_start", header=header)
        replay_holder["log"] = ReplayLog(
            ROOT / "results" / "checkpoints"
            / f"{args.run_id}__{pid}.replay.json",
            enabled=args.resume)

        # Inference and retrieval run on the bare header; the file's
        # `open …` scopes are attached after (prepend_open_scopes).
        inferred = infer_imports_from_header(header)
        if cli_imports:
            verify_imports = cli_imports
            import_source = "cli"
        elif args.import_mode == "llm":
            # LLM-resolved imports with a compile gate: the model reads the
            # statement (with its opens), proposes imports, and gets the
            # Lean error back on failure. No hardcoded module knowledge.
            # NB: do NOT import `compile_lean` / `_ERROR_LINE_RE` here.
            # Both are module-level imports (top of file). A local
            # `from … import` inside main() rebinds them as LOCALS of
            # main for its whole body, so when this branch does not run
            # (e.g. `--lean-imports` takes the `cli_imports` path above)
            # the cells stay unbound and every nested closure that reads
            # them — `_probe_fn`, the verify wrapper — dies with
            # "cannot access free variable ... in enclosing scope".
            # That made `--lean-imports` + theory mode a guaranteed
            # crash (lrs_sol_legacyARM_store, 2026-07-30).
            from search.llm_imports import resolve_imports_llm

            def _stmt_gate(imports: str) -> str | None:
                # reject_sorry=False: the gate source is `:= by sorry` BY
                # DESIGN — it asks "does the STATEMENT elaborate under
                # these imports", and reads the error text, not `ok`.
                # Skipping the compile would make every proposed import
                # set pass (observed live: p1963a2_dag_v2 accepted a
                # nonexistent `Mathlib.Topology.Defs` and could not
                # compile anything for the rest of the run).
                res = compile_lean(
                    f"{imports}\n\n{header} := by sorry\n",
                    timeout_s=args.verify_timeout,
                    reject_sorry=False)
                bad = (_ERROR_LINE_RE.search(res.errors)
                       or res.errors.startswith("timeout"))
                return res.errors[:2000] if bad else None

            verify_imports, import_source = resolve_imports_llm(
                header, _sketch_llm_call, _stmt_gate)
            print(f"[{pid}] imports via {import_source}", flush=True)
        else:
            verify_imports = inferred.imports
            import_source = ("inferred" if inferred.matched
                              else "fallback_mathlib")
        _trace("imports_resolved", imports=verify_imports,
               source=import_source)

        # Retrieve premise names for the prompts. Soft-fails to none —
        # a missing graph archive or optional dep must not kill a run.
        premise_names: list[str] | None = None
        if args.premises != "none":
            try:
                from search.premise_retrieval import retrieve
                strategy = "B" if args.premises == "bm25" else "A"
                premise_names = [
                    p.name for p in retrieve(header, strategy=strategy,
                                              cap=args.premises_k)
                ] or None
            except Exception as e:
                print(f"[{pid}] premise retrieval unavailable: "
                      f"{type(e).__name__}: {e}", flush=True)
                premise_names = None
            # NOTE (2026-07-26): a curated division-lemma splice that
            # lived here was removed as hardcoding — the mechanical
            # `apply_pin_renames` rewrite in proof_dag covers the
            # stale-name failure mode without naming lemmas in the
            # retrieval path.
            _trace("premises", strategy=args.premises,
                   names=premise_names or [])

        # PREMISE-SEEDED IMPORTS.
        #
        # Import selection ran BEFORE retrieval and from the STATEMENT
        # alone, so the names BM25 had just surfaced never reached the
        # import set; the refresh hook then rediscovered them reactively,
        # one failed compile at a time. On p1981a1_dag_v1 the guess was
        # Polynomial/Finset.Interval/Deriv for a 5-adic valuation
        # problem, `Nat.Prime` was never imported, and
        # Nat.Factorization.Defs arrived at t=13522s of 14172s.
        #
        # Deterministic (graph lookup, no LLM, no compiles) and
        # environment-derived, so it adds no problem-shaped knowledge.
        # Skipped when the user pinned imports explicitly.
        if (args.premise_seeded_imports and premise_names
                and not args.lean_imports):
            from search.decl_module_index import modules_for_premises
            seeded = modules_for_premises(
                premise_names, limit=args.premise_import_limit)
            have = set(re.findall(r"^\s*import\s+(\S+)", verify_imports,
                                  re.M))
            added = [m for m in seeded if m not in have]
            if added:
                # Statement-gate the merged set: a seeded module must
                # never make the problem's own statement stop compiling.
                merged = verify_imports + "\n" + "\n".join(
                    f"import {m}" for m in added)
                # reject_sorry=False — same reason as _stmt_gate: the
                # source is deliberately sorry-terminated and the verdict
                # comes from the error text.
                gate = compile_lean(f"{merged}\n\n{header} := by sorry\n",
                                    timeout_s=args.verify_timeout,
                                    reject_sorry=False)
                if not _ERROR_LINE_RE.search(gate.errors or ""):
                    verify_imports = merged
                    import_source = f"{import_source}+premise_seeded"
                    _trace("imports_seeded", added=added,
                           source="premise_graph")
                    print(f"[{pid}] +{len(added)} premise-seeded imports",
                          flush=True)
                else:
                    _trace("imports_seeded", added=[], rejected=added,
                           detail=(gate.errors or "")[:300])

        # (opens/preludes already live inside `header` — see eval.loader)

        def _refresh_imports(new_decls: str) -> str | None:
            """ARM-loop import refresh: when a theory proposal arrives,
            ask the policy model whether the NEW declarations need
            imports the current set lacks; compile-gate the stubbed
            declarations under the merged set before adopting it. The
            verify/probe closures read `verify_imports` live, so a
            successful merge takes effect immediately. Soft-fails to
            None (imports unchanged) on any problem."""
            nonlocal verify_imports
            from backend.compile_verify import compile_lean as _cl
            from backend.compile_verify import _ERROR_LINE_RE as _ELR
            from search.llm_imports import (llm_import_prompt,
                                            parse_llm_imports)
            from search.decl_module_index import (resolve_decl_module,
                                                  unknown_names)
            try:
                # The hook's text may carry `--` comment lines (the
                # error-driven path appends the Lean failure). The LLM
                # prompt sees everything; the compile gate sees only
                # the code, with proof-less lemma/theorem statements
                # sorry-stubbed.
                parts = []
                for d in new_decls.split("\n\n"):
                    code = "\n".join(
                        ln for ln in d.splitlines()
                        if not ln.lstrip().startswith("--")).strip()
                    if not code:
                        continue
                    if (code.split(None, 1)[0] in ("lemma", "theorem")
                            and ":=" not in code):
                        parts.append(f"{code} := by sorry")
                    else:
                        parts.append(code)
                stubbed = "\n\n".join(parts)
                if not stubbed:
                    # Error-driven asks carry only `--` comment lines
                    # (Lean errors, no new declarations) — gate the
                    # merged imports on the problem statement itself.
                    stubbed = f"{header} := by sorry"
                cur = [ln for ln in verify_imports.splitlines()
                       if ln.strip()]
                # Deterministic first: names cited by unknown-name
                # diagnostics are resolved against the local Mathlib
                # declaration graph — the environment answers before
                # the model guesses (v4: the model guessed
                # Data.Fin.Basic for a betweenness lemma). Hallucinated
                # names resolve to None and add nothing.
                extra: list[str] = []
                for nm in unknown_names(new_decls):
                    mod = resolve_decl_module(nm)
                    if mod:
                        line = f"import {mod}"
                        if line not in cur and line not in extra:
                            extra.append(line)
                source = "graph"
                if not extra:
                    sys_p, user_p = llm_import_prompt(
                        f"-- Current imports (already available):\n"
                        f"{verify_imports}\n\n{new_decls}")
                    cand = parse_llm_imports(
                        _sketch_llm_call(sys_p, user_p))
                    if not cand:
                        return None
                    extra = [ln for ln in cand.splitlines()
                             if ln.strip() and ln not in cur]
                    source = "llm"
                if not extra:
                    return None
                merged = "\n".join(cur + extra)
                # reject_sorry=False: `stubbed` is sorry-terminated BY
                # CONSTRUCTION — either lemma statements this function
                # stubbed itself, or the bare `header := by sorry` the
                # error-driven path falls back to. With the default,
                # compile_verify rejects the source BEFORE compiling and
                # returns an `error:`-marked message, which the check
                # below then reads as "the merged set is broken" — so the
                # gate could never pass and this whole mechanism silently
                # returned None on every call, on every path.
                # MEASURED on mf18_amc12a_2003_p23_scaffold_budget
                # (2026-08-27): six repair rounds of `Unknown constant
                # Nat.divisors` with the resolver holding the answer
                # (Mathlib.NumberTheory.Divisors) and not one import ever
                # adopted. compile_verify's own docstring names the
                # import statement gates as callers that MUST opt out.
                gate = _cl(f"{merged}\n\n{stubbed}\n",
                           timeout_s=args.verify_timeout,
                           reject_sorry=False)
                errs = gate.errors or ""
                if _ELR.search(errs) or errs.startswith("timeout"):
                    # Merged set breaks the stubbed decls (bad module
                    # name, clash…) — keep the current imports; the
                    # prove step's own error feedback takes it from
                    # here.
                    return None
                verify_imports = merged
                return f"+{len(extra)} ({source}): " + "; ".join(extra)
            except Exception:
                return None

        def _dump_path(tag: str):
            """Next dump path for this problem, or None when disabled.

            Keeps the EXACT .lean file handed to Lean, plus a JSON sidecar
            with the replay argv. Without this the evidence is gone: the
            temp file is deleted after each compile, and the trace records
            the model's response rather than the assembled source — which
            made `unknown tactic` unfalsifiable in putnam_easy8_sol_v1
            (8 occurrences, cause undetermined)."""
            if not args.debug_dump_lean:
                return None
            _dump_seq[0] += 1
            d = Path(args.debug_dump_lean) / args.run_id / pid
            d.mkdir(parents=True, exist_ok=True)
            return d / f"{_dump_seq[0]:03d}_{tag}.lean"

        def _compile_verify(theorem_header: str, body: str) -> dict:
            res = verify_proof(
                theorem_header, body,
                imports=verify_imports,
                timeout_s=args.verify_timeout,
                debug_dump_path=_dump_path("verify"),
            )
            # body_line_offset: file lines before body line 1. Mirrors the
            # exact source layout in verify_proof:
            #   f"{imports}\n\n{theorem_header} := by\n{body}\n"
            prefix = f"{verify_imports}\n\n{theorem_header} := by"
            return {"ok": res.ok, "errors": res.errors,
                    "body_line_offset": prefix.count("\n") + 1}

        def _recognize_leaf(leaf_statement: str) -> str | None:
            """Has this stuck leaf already been proved by somebody?

            Retrieval searches the leaf's GOAL (the conclusion is what
            identifies a lemma); confirmation uses the WHOLE statement,
            because a goal lifted out of its binders has unbound
            variables. `_verify_fn` is the oracle, so a wrong suggestion
            costs one compile and can never enter a proof.

            The index is loaded ONCE per process and shared across
            problems. That is retrieval infrastructure derived from the
            pin — like `decl_module_index` — not cross-problem transfer:
            nothing a run PROVES is carried anywhere.
            """
            idx = _statement_index[0]
            if idx is None:
                return None
            from search.proof_dag import split_theorem_header
            from search.recognize import recognize_for_statement
            split = split_theorem_header(leaf_statement)
            goal = split[1] if split else leaf_statement
            return recognize_for_statement(
                leaf_statement, goal, idx,
                verify_fn=_verify_fn, k=args.recognize_k, trace=_trace)

        def _library_compile_fn(source: str) -> dict:
            """Compile a COMPLETE Lean source — not a (header, body) pair.

            A library declaration is a finished unit (`def ... := ...`,
            `lemma ... := by ...`), so it must NOT go through
            `verify_proof`, which appends `:= by\\n<body>` and would turn
            every declaration into a syntax error. Imports are prepended
            here for the same reason the other closures own them.

            `used_sorry` is rejected explicitly: a library exists to be
            RELIED ON by later proofs, so a stubbed declaration would
            silently poison everything built on top of it.
            """
            t0l = time.time()
            src = f"{verify_imports}\n\n{source.rstrip()}\n"
            res = compile_lean(src, timeout_s=args.verify_timeout,
                               debug_dump_path=_dump_path("library"))
            ok = bool(res.ok) and not getattr(res, "used_sorry", False)
            _trace("verify_call", backend="library(full-source)",
                   duration_s=round(time.time() - t0l, 1),
                   ok=ok, errors=res.errors if not ok else None)
            return {"ok": ok, "errors": res.errors}

        def _verify_fn(theorem_header: str, body: str) -> dict:
            if repl_session is None:
                return _compile_verify(theorem_header, body)
            r = repl_session.verify_whole(
                theorem_header, body,
                imports=verify_imports,
                timeout_s=args.verify_timeout,
            )
            if not r.get("ok"):
                if has_header_level_error(
                        r.get("errors") or "",
                        r.get("body_line_offset") or 0):
                    # REPL claims the statement itself is broken. Cross-
                    # check against the compile oracle so a REPL/env
                    # artifact can't poison repair attribution; genuine
                    # header errors reproduce and stop the repair loop.
                    return _compile_verify(theorem_header, body)
                return r
            # REPL accepted — confirm in a fresh session. The compile
            # verifier is the only oracle; one cold compile
            # per CANDIDATE proof instead of one per repair round.
            return _compile_verify(theorem_header, body)

        def _traced_verify(theorem_header: str, body: str) -> dict:
            t0v = time.time()
            # A 900 s compile is the single most expensive step in a
            # pipeline cell; replaying it is most of what resume buys.
            r = replay_holder["log"].step(
                "verify", (theorem_header, body, verify_imports),
                lambda: _verify_fn(theorem_header, body),
                record=_is_real_verdict)
            _trace("verify_call", backend=args.verify_backend,
                   duration_s=round(time.time() - t0v, 1),
                   ok=bool(r.get("ok")), errors=r.get("errors"))
            return r

        def _probe_fn(theorem_header: str, body: str) -> dict:
            """Replay-wrapped satisfaction probe.

            ARM gate probes are full Lean compiles — minutes each, and a
            theory round fires several. Recorded like the real verify;
            the probe is never an oracle either way.
            """
            return replay_holder["log"].step(
                "probe", (theorem_header, body, verify_imports),
                lambda: _probe_uncached(theorem_header, body),
                record=_is_real_verdict)

        def _probe_uncached(theorem_header: str, body: str) -> dict:
            """Errors-only compile that TOLERATES sorry — used solely
            by theory-mode satisfaction probes (assumed lemmas are
            sorry stubs). ok = no error lines. Never counts as a
            solve; the real oracle is _verify_fn."""
            t0p = time.time()
            src = f"{verify_imports}\n\n{theorem_header} := by\n{body}\n"
            # reject_sorry=False: this probe's sources are sorry-STUBBED by
            # design and its verdict comes from the error text, not `ok`.
            # With the default the compile would be skipped and the empty
            # error text read as success — a vacuous gate pass on every
            # theory (observed live on p1963a2_dag_v2).
            res = compile_lean(src, timeout_s=args.verify_timeout,
                               debug_dump_path=_dump_path("probe"),
                               reject_sorry=False)
            errs = res.errors or ""
            has_err = (bool(_ERROR_LINE_RE.search(errs))
                       or errs.startswith("timeout"))
            prefix = f"{verify_imports}\n\n{theorem_header} := by"
            _trace("verify_call", backend="probe(sorry-ok)",
                   duration_s=round(time.time() - t0p, 1),
                   ok=not has_err, errors=errs if has_err else None)
            return {"ok": not has_err, "errors": errs,
                    "body_line_offset": prefix.count("\n") + 1}

        def _circularity_defeq_probe(stmt_a: str, stmt_b: str):
            """Point 7a 'elaborated' mode: decide whether two statements'
            propositions are DEFINITIONALLY equal, via a fresh Lean
            compile of `example : (Pₐ) ↔ (P_b) := Iff.rfl`. Returns True
            (defeq → circular), False (distinct / conversion failed →
            Lean errors), or None (could not run → caller treats as NOT
            circular). Never raises. Reads verify_imports live."""
            from search.goal_fingerprint import statement_to_prop
            from search.proof_dag import split_theorem_header
            try:
                pa = statement_to_prop(stmt_a, split_fn=split_theorem_header)
                pb = statement_to_prop(stmt_b, split_fn=split_theorem_header)
                if not pa or not pb:
                    return None
                src = (f"{verify_imports}\n\n"
                       f"example : ({pa}) ↔ ({pb}) := Iff.rfl\n")
                res = compile_lean(src, timeout_s=args.verify_timeout)
            except Exception as _e:
                _trace("circularity_defeq", ok=False,
                       detail=f"probe error: {type(_e).__name__}")
                return None
            verdict = bool(res.ok)
            _trace("circularity_defeq", ok=True, defeq=verdict,
                   detail=(res.errors or "")[:200])
            return verdict

        # Leaf-fallback ladder: LLM-proposed domain-appropriate closers by
        # default (the static ladder is arithmetic-biased — wrong for
        # topology/analysis); static list as fallback and legacy mode.
        fallbacks: tuple[str, ...] = ()
        fallback_source = "off"
        if args.leaf_fallback:
            fallbacks = DEFAULT_LEAF_FALLBACKS
            fallback_source = "static"
            if args.leaf_fallback_mode == "llm":
                try:
                    raw_fb = _sketch_llm_call(
                        "You are a Lean 4 + Mathlib expert. Given a theorem "
                        "statement, propose up to 6 short finishing tactics "
                        "('closers') likely to discharge SMALL subgoals in "
                        "this statement's mathematical domain. One tactic "
                        "per line, plain text, no arguments containing "
                        "holes (`?_`), no sorry/admit/native_decide, no "
                        "prose.",
                        f"Statement:\n```lean\n{header}\n```")
                    from search.proof_dag import _FORBIDDEN_TAC_RE
                    cand = []
                    for ln in (raw_fb or "").splitlines():
                        ln = ln.strip().strip("`-• ").strip()
                        if (ln and len(ln) <= 60 and "?" not in ln
                                and not _FORBIDDEN_TAC_RE.search(ln)
                                and ln not in cand):
                            cand.append(ln)
                        if len(cand) >= 6:
                            break
                    if cand:
                        fallbacks = tuple(cand)
                        fallback_source = "llm"
                except Exception:
                    pass  # static ladder already in place
        _trace("fallback_ladder", source=fallback_source,
               tactics=list(fallbacks))

        # ---- Pre-flight refutation ------------------------------------
        # Try to prove the goal FALSE before trying to prove it. Believed
        # only on a KERNEL-VERIFIED proof of the negation; a failed search
        # changes nothing and the prover runs exactly as before. A
        # refutation is reported as its own outcome and never counts as a
        # solve. Motivation: lrs_sol_legacyARM_store_v3 burned 8h22m / 37
        # compiles on a statement that is false.
        t0 = time.time()
        refutation = None
        if args.refute_first:
            from search.refute import attempt_refutation
            from search.goal_fingerprint import statement_to_prop
            from search.proof_dag import (split_theorem_header as _split,
                                          _decl_offset)
            _prop = statement_to_prop(header, split_fn=_split)
            if _prop is None:
                _trace("refute", stage="skipped",
                       detail="header does not split into a closed prop")
            else:
                _prelude = header[:_decl_offset(header)]
                refutation = attempt_refutation(
                    _prop,
                    llm_call=_sketch_llm_call,
                    verify_fn=_traced_verify,
                    prelude=_prelude,
                    attempts=args.refute_attempts,
                    max_declines=args.refute_max_declines,
                    premises=premise_names,
                    trace=_trace if args.trace else None,
                )
                if refutation.refuted:
                    dt = time.time() - t0
                    _trace("outcome", verified=False, refuted=True,
                           stage="refuted", wall_s=round(dt, 1),
                           detail=refutation.witness[:300])
                    print(f"[{pid}] REFUTED in {dt:.1f}s — "
                          f"{refutation.witness[:120]}")
                    row = {
                        "id": pid,
                        # NOT "solved": the statement is false. Kept a
                        # distinct outcome so it can never inflate a
                        # solve-rate denominator or numerator.
                        "outcome": "refuted",
                        "wall_s": round(dt, 1),
                        "bench_dir": args.bench_dir or "minif2f",
                        "provider": args.provider,
                        "model": args.model,
                        "verify_imports": verify_imports,
                        "refutation": refutation.to_dict(),
                        "trace_file": (
                            str((trace_dir / f"{pid}.trace.jsonl")
                                .relative_to(ROOT))
                            if args.trace else None),
                    }
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    continue
                _trace("refute", stage="no refutation found",
                       detail=f"{refutation.attempts_used} attempt(s); "
                              f"proceeding to prove")

        result = attempt_dag_proof(
            header,
            sketch_llm_call=_sketch_llm_call,
            verify_fn=_traced_verify,
            sketch_attempts=args.sketch_attempts,
            repair_rounds=args.repair_rounds,
            leaf_fallbacks=fallbacks,
            closer_fallback=args.closer_fallback,
            leaf_closer_call=leaf_closer_call,
            premises=premise_names,
            decompose_depth=args.decompose_depth,
            abduce_lemmas=args.abduce_lemmas,
            abduce_mode=args.abduce_mode,
            # Import refresh is pointless when the whole library is
            # already in scope; any narrow set (llm/inferred/cli) can
            # be missing proof-layer modules the ARM theory needs.
            refresh_imports_call=(
                _refresh_imports
                if verify_imports.strip() != "import Mathlib" else None),
            probe_fn=_probe_fn,
            abduce_theory_rounds=args.abduce_theory_rounds,
            abduce_theory_trigger=args.abduce_theory_trigger,
            abduce_minimize=args.abduce_minimize,
            abduce_minimize_verify=args.abduce_minimize_verify,
            abduce_quality_gate=args.abduce_quality_gate,
            abduce_quality_factor=args.abduce_quality_factor,
            provability_prior_fn=_provability_prior_fn,
            causal_attribution=args.causal_attribution,
            circularity_mode=args.circularity_mode,
            circularity_defeq_probe=(
                _circularity_defeq_probe
                if args.circularity_mode == "elaborated" else None),
            dag_repair_engine=_exec_engine,
            leaf_solvers=_leaf_solvers or None,
            solver_context=(
                SolverContext(theorem_header=header,
                              premises=tuple(premise_names or ()),
                              extra={})
                if _leaf_solvers else None),
            lemma_library_path=args.lemma_library,
            # Constructed HERE, inside the per-problem loop, so the store
            # never outlives the problem: reuse across problems would be
            # cross-problem transfer and would contaminate a matched-budget
            # ablation exactly the way the on-disk bank would.
            sketch_mode=args.sketch_mode,
            blueprint_sequential=args.blueprint_sequential,
            blueprint_rounds=args.blueprint_rounds,
            recognize_leaf_call=(_recognize_leaf
                                 if args.recognize_leaves else None),
            reframe_on_abandon=args.reframe_on_abandon,
            reframe_rounds=args.reframe_rounds,
            reframe_leverage_factor=args.reframe_leverage_factor,
            reframe_build_library=args.reframe_build_library,
            reframe_library_decls=args.reframe_library_decls,
            reframe_library_attempts=args.reframe_library_attempts,
            library_compile_fn=(_library_compile_fn
                                if args.reframe_build_library else None),
            lemma_store=(_LemmaStore()
                         if args.abduce_lemma_store else None),
            store_feedback=args.store_feedback,
            prove_retry_budget=(_retry_budget.DEFAULT_BUDGETS
                                if args.abduce_retry_budget else None),
            lemma_provenance={
                "source_problem": pid,
                "source_split": _source_split,
                "run_id": args.run_id,
                "model": args.model,
                "verified_mathlib_commit": _mathlib_commit,
                "imports": [ln for ln in verify_imports.splitlines()
                            if ln.strip()],
                # Conservative: nothing is eval-eligible until explicitly
                # promoted out-of-band.
                "allowed_for_eval": False,
            },
            trace=_trace if args.trace else None,
        )
        dt = time.time() - t0

        outcome = "solved" if result.verified else "failed"
        if result.verified:
            solved += 1
        _trace("outcome", verified=result.verified,
               stage=result.failure_stage, detail=result.failure_detail,
               wall_s=round(dt, 1),
               sketch_attempts=result.sketch_attempts,
               repair_rounds=result.repair_rounds_used,
               repaired_ids=result.repaired_ids,
               leaf_closer_fixed_ids=result.leaf_closer_fixed_ids,
               decomposed_ids=result.decomposed_ids,
               abduced_ids=result.abduced_ids,
               repair_errors=result.repair_errors)

        # Bound the raw_responses payload — Goedel sometimes emits multi-
        # thousand-token replies and we don't want to balloon the JSONL.
        raws_bounded = [r[:2000] for r in (result.raw_responses or [])][:3]

        sketch_json = None
        if result.sketch is not None:
            sketch_json = {
                "haves": [
                    {"id": h.id, "type": h.type_text,
                     "tactic": h.tactic, "depends": list(h.depends)}
                    for h in result.sketch.haves
                ],
                "closer": result.sketch.closer,
            }

        row = {
            "id": pid,
            "outcome": outcome,
            "wall_s": round(dt, 1),
            "bench_dir": args.bench_dir or "minif2f",
            "leaf_fallback_source": fallback_source,
            "provider": args.provider, "model": args.model,
            "verify_imports": verify_imports,
            "import_source": import_source,
            "sketch_attempts_used": result.sketch_attempts,
            "max_sketch_attempts": args.sketch_attempts,
            "failure_stage": result.failure_stage,
            "failure_detail": result.failure_detail,
            "sketch": sketch_json,
            "assembled_proof": result.assembled_proof,
            "raw_responses_preview": raws_bounded,
            "dag_used": True,
            "leaf_fallback_used": result.leaf_fallback_used,
            "closer_fallback_used": args.closer_fallback,
            "repair_rounds_used": result.repair_rounds_used,
            "repaired_ids": result.repaired_ids,
            "repair_errors": result.repair_errors,
            "leaf_closer_fixed_ids": result.leaf_closer_fixed_ids,
            "leaf_closer_model": (args.leaf_closer_model
                                   if args.leaf_closer_provider else None),
            "verify_backend": args.verify_backend,
            "trace_file": (str((trace_dir / f"{pid}.trace.jsonl")
                               .relative_to(ROOT))
                           if args.trace else None),
            "premises_strategy": args.premises,
            "premises_count": len(premise_names or []),
            "decomposed_ids": result.decomposed_ids,
            "abduce_mode": args.abduce_mode if args.abduce_lemmas else None,
            # An ARM ablation is only readable if the row says when the
            # loop was allowed to fire, not just that it was enabled.
            "abduce_theory_trigger": (args.abduce_theory_trigger
                                      if args.abduce_lemmas else None),
            # What this cell replayed rather than recomputed, and where
            # (if anywhere) the resumed trajectory left the recorded one.
            "replay": replay_holder["log"].summary() if args.resume else None,
            # None here means the API default ("high"), not "no thinking".
            "effort": args.effort,
            "abduced_ids": result.abduced_ids,
            "abduced_lemmas": result.abduced_lemmas,
            # Run-scoped store: `lemma_store_kept` is every lemma this
            # problem ever kernel-proved, INCLUDING those whose theory
            # never committed — the ones the legacy loop discarded
            # silently. Null when the store is disabled.
            "lemma_store_stats": result.lemma_store_stats,
            "lemma_store_kept": result.lemma_store_kept,
            "prove_retry_log": result.prove_retry_log,
            # Set on a SOLVE only: the header the proof was compiled
            # against, defs and store lemmas included. `assembled_proof`
            # alone does not reconstruct a solve — see DagResult.
            "verified_header": result.verified_header,
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{pid}] {outcome} in {dt:.1f}s "
              f"(stage={result.failure_stage or 'none'}, "
              f"sketches={result.sketch_attempts}, "
              f"repairs={result.repair_rounds_used}, "
              f"haves={len(result.sketch.haves) if result.sketch else 0})",
              flush=True)
        tracer_holder["tracer"] = None

    if repl_session is not None:
        repl_session.close()
    print(f"\n{solved}/{attempted} solved -- log: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
