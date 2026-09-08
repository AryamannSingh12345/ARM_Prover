"""Compile-based verifier: write a .lean file, run `lake env lean`, parse errors.

The whole-proof verification backend, and the compile primitive the
tactic-step surrogate builds on. Replaces PyPantograph on this Windows
host.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .sandbox import scrubbed_env

# Default Lean project root used when no override is passed. The runner can
# override per-call via `lean_project_root=...` so a different lake project
# (e.g. a peer Mathlib pin) can be used without editing this file.
LEAN_ROOT = Path(__file__).resolve().parents[2] / "lean"
GEN_DIR = LEAN_ROOT / "Generated"
GEN_DIR.mkdir(exist_ok=True)

# `sorryAx` is the primitive `sorry` elaborates to, and `\bsorry\b` does
# NOT match it (no word boundary before `Ax`). A bare-baseline run closed
# putnam_1981_a1 with `exact sorryAx _ true` and was scored solved. This
# alternation is a cheap PRE-FILTER that rejects the known spellings
# without paying for a compile; the real gate is the axiom check below,
# which asks the kernel what the proof depends on instead of guessing
# from the text. Do not treat this regex as the defence.
_SORRY_RE = re.compile(r"\bsorry\b|\bsorryAx\b|\badmit\b")
_NATIVE_DECIDE_RE = re.compile(r"\bnative_decide\b")
# Match `error:` anywhere on a line — including Lean's named-diagnostic
# form `error(lean.unknownIdentifier):`. Avoids regex pitfalls with Windows
# path absolute prefixes (`C:\...`); whatever precedes `error` we don't
# care about. (Inside compile_lean the returncode check papered over the
# missing `(...)` form; callers that re-derive from the errors TEXT alone
# — run_dag's probe/statement gates — were blind to it.)
_ERROR_LINE_RE = re.compile(r"(^|\n)[^\n]*?\berror(?:\([^)]*\))?:", re.M)
# Lean's marker for implicit / explicit sorry in the resulting declaration.
#
# QUOTING (measured 2026-08-16, toolchain v4.30.0-rc2): Lean emits
#   warning: declaration uses `sorry`
# with BACKTICKS. The original pattern required straight quotes
# ("declaration uses 'sorry'") and therefore never matched on this
# toolchain — this whole layer was inert, which is half of why
# `exact sorryAx _ true` was scored as a solve on putnam_1981_a1.
# Accept either quoting so a future toolchain flipping back still works.
_SORRY_WARNING_RE = re.compile(r"declaration uses [`']sorry[`']", re.M)

# stdout/stderr prefix cap for the JSON sidecar — full lake output can be
# multi-MB on Mathlib errors; the prefix is plenty for triage.
_SIDECAR_OUTPUT_CAP = 4000


@dataclass(slots=True)
class CompileResult:
    ok: bool
    errors: str
    used_sorry: bool
    used_native_decide: bool
    # Replay envelope, populated by compile_lean on every call. cwd is the
    # subprocess working directory; argv is the exact `lake env lean ...`
    # command. `debug_dump_path` is set ONLY when the caller passed
    # `debug_dump_path=` (i.e. when --debug-dump-lean is on); otherwise None.
    # `elapsed_s` is the lake wall time for this single attempt.
    cwd: str = ""
    argv: list[str] = field(default_factory=list)
    debug_dump_path: str | None = None
    elapsed_s: float = 0.0
    # Axiom-gate verdict (backend/axiom_check.py). None when the gate did
    # not run — either it was disabled, or the compile had already failed
    # for another reason and was never probed. True/False otherwise;
    # `ok` is forced False whenever this is False.
    axioms_ok: bool | None = None
    axiom_detail: str = ""


def _write_temp(source: str) -> Path:
    name = f"Try_{uuid.uuid4().hex[:12]}.lean"
    p = GEN_DIR / name
    p.write_text(source, encoding="utf-8")
    return p


def _write_sidecar(debug_dump_path: Path, *, lean_path: Path,
                   source: str, cwd: str, argv: list[str],
                   timeout_s: int, result: CompileResult,
                   stdout: str, stderr: str, status: str) -> None:
    """Write a JSON sidecar next to <debug_dump_path>.lean.

    `status` is "ok" | "errors" | "timeout" so a triage script can grep
    without re-parsing the lake output. Failure to write the sidecar is
    swallowed (never crash a run because of debug plumbing)."""
    sidecar = {
        "status": status,
        "ok": result.ok,
        "outcome": "ok" if result.ok else status,
        "cwd": cwd,
        "argv": argv,
        "timeout_s": timeout_s,
        "elapsed_s": round(result.elapsed_s, 3),
        "used_sorry": result.used_sorry,
        "used_native_decide": result.used_native_decide,
        # Replay command — copy-paste runnable on Windows PowerShell / sh.
        "replay": {
            "cwd": cwd,
            "argv": argv,
            "lean_file": str(lean_path),
        },
        "source_size_bytes": len(source.encode("utf-8")),
        "stdout_prefix": (stdout or "")[:_SIDECAR_OUTPUT_CAP],
        "stderr_prefix": (stderr or "")[:_SIDECAR_OUTPUT_CAP],
        # The Lean source the runner generated, capped only so the sidecar
        # doesn't bloat. The dumped .lean file alongside is the
        # canonical, unbounded source.
        "source_prefix": source[:_SIDECAR_OUTPUT_CAP],
    }
    sidecar_path = debug_dump_path.with_suffix(".json")
    try:
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        # Never let debug plumbing crash a run.
        pass


#: Backoff for a lake spawn that fails WITHOUT OUTPUT. Windows reports
#: this as 0xC0000142 STATUS_DLL_INIT_FAILED — the process starts and its
#: DLLs cannot initialise, which on this host is memory pressure, not a
#: toolchain fault (`lake env lean --version` succeeds when idle).
#:
#: MEASURED on the mf20 pipeline arm (2026-08-28): 184 such exits across
#: two cells; 82% of one cell's 61 verify calls never ran. A single 2s
#: retry was far too short — the pressure window lasts as long as a
#: Mathlib compile holds its memory, i.e. minutes. These delays are
#: chosen to outlast it rather than merely report it.
_INFRA_RETRY_DELAYS = (5.0, 20.0, 60.0)

#: Process-wide tally so a caller can tell a run poisoned by spawn
#: failures from a run that genuinely failed to prove things.
_INFRA_STATE = {"failures": 0, "recovered": 0}


def infra_failure_count() -> int:
    """Spawn failures that survived every retry in this process."""
    return _INFRA_STATE["failures"]


def compile_lean(source: str, timeout_s: int = 300,
                 lean_project_root: Path | str | None = None,
                 debug_dump_path: Path | str | None = None,
                 check_axioms: bool = True,
                 reject_sorry: bool = True) -> CompileResult:
    """Compile a complete Lean source. Returns ok iff no errors and no sorry.

    `reject_sorry` (default True) lets the call return WITHOUT running
    lake when the source already contains `sorry`/`admit`/`sorryAx` or
    `native_decide` — such a source can never be `ok`, so the compile
    buys nothing. Pass False from callers that deliberately compile
    sorry-STUBBED sources and read the ERROR TEXT rather than `ok`
    (theory-mode satisfaction probes); for them the compile is the
    result, not a formality.

    `check_axioms` (default ON) runs the axiom gate — see
    backend/axiom_check.py — on a source that would otherwise be scored
    ok. It costs ONE additional compile, and only ever on a would-be
    SOLVE: a source with errors, sorry or native_decide has already
    failed and is never probed. It is a correctness gate, not a research
    lever, which is why it defaults on; pass False for probe/diagnostic
    compiles where the caller does not use `ok` as a verdict.

    `lean_project_root` overrides the cwd of the `lake env lean` subprocess.
    When None, falls back to the module-level `LEAN_ROOT`. Set this to the
    directory containing `lakefile.toml` for the Mathlib-aware lake project
    you want to compile against — running from anywhere else makes
    `import Mathlib` resolve against the wrong (or no) cache.

    `debug_dump_path` is a path STEM (no extension). When set, this function
    writes:
      - <stem>.lean  — the EXACT source sent to lake, written BEFORE the
                       subprocess so it survives a hang / timeout / crash.
      - <stem>.json  — sidecar with cwd, argv, timeout, elapsed, stdout/
                       stderr prefix, outcome — written after the subprocess
                       returns (or times out).
    Caller is responsible for choosing a unique stem per attempt.
    """
    lean_root = Path(lean_project_root) if lean_project_root else LEAN_ROOT
    used_sorry = bool(_SORRY_RE.search(source))
    used_native_decide = bool(_NATIVE_DECIDE_RE.search(source))

    # Pre-emptive .lean dump for replay. Deliberately ABOVE both the
    # short-circuit and lake: it must survive a hang or timeout, and a
    # source rejected for `sorry` is exactly the kind someone turns
    # --debug-dump-lean on to inspect, so returning early without writing
    # it would leave that case with no artifact at all.
    dump_stem: Path | None = None
    if debug_dump_path is not None:
        dump_stem = Path(debug_dump_path)
        dump_lean = dump_stem.with_suffix(".lean")
        try:
            dump_lean.parent.mkdir(parents=True, exist_ok=True)
            dump_lean.write_text(source, encoding="utf-8")
        except Exception:
            # Never block lake on debug plumbing.
            dump_stem = None

    # SHORT-CIRCUIT: `ok` already requires `not used_sorry` and `not
    # used_native_decide`, so a source carrying either is rejected no
    # matter what lake says. Running the compile anyway pays full Mathlib
    # import cost to learn nothing — MEASURED on this host at 22 minutes
    # to reject a file whose entire proof body was `sorry` (p1963a2
    # baseline, attempt 3), and ~20 more on its final attempt.
    #
    # ONLY valid when the caller treats `ok` as the verdict. Theory-mode
    # SATISFACTION PROBES and the import STATEMENT GATES deliberately
    # compile sorry-terminated sources and derive their verdict from the
    # error text, ignoring `ok` — for them a skipped compile is not a
    # cheap rejection but a LOST RESULT. Measured live on
    # p1963a2_dag_v2: with this unconditional, every probe returned in
    # 0.0s and every gate read the empty error text as "clean", so the
    # satisfaction gate passed VACUOUSLY on every theory and a
    # nonexistent module was accepted into the import set. Such callers
    # must pass `reject_sorry=False`.
    if reject_sorry and (used_sorry or used_native_decide):
        marker = "sorry/admit/sorryAx" if used_sorry else "native_decide"
        # The message carries an `error:` marker DELIBERATELY: since those
        # callers scan for `error:`, a missed opt-out would otherwise FAIL
        # OPEN — accepting what it should reject. With the marker it fails
        # CLOSED (spurious rejection): visible, safe, and fixable.
        sc_result = CompileResult(
            ok=False,
            errors=(f"error: rejected before compile: source contains "
                    f"{marker}, which can never be accepted"),
            used_sorry=used_sorry,
            used_native_decide=used_native_decide,
            cwd=str(lean_root),
            argv=[],
            debug_dump_path=(str(dump_stem.with_suffix(".lean"))
                             if dump_stem is not None else None),
            elapsed_s=0.0,
        )
        # Honour the documented dump contract: <stem>.json accompanies
        # <stem>.lean. `argv` is empty and there is no stdout/stderr
        # because no subprocess ran — the status says exactly that, so a
        # tool reading sidecars finds a file rather than a gap.
        if dump_stem is not None:
            _write_sidecar(
                dump_stem, lean_path=dump_stem.with_suffix(".lean"),
                source=source, cwd=str(lean_root), argv=[],
                timeout_s=timeout_s, result=sc_result,
                stdout="", stderr="", status="rejected_precompile",
            )
        return sc_result

    path = _write_temp(source)
    argv = ["lake", "env", "lean", str(path)]
    cwd_str = str(lean_root)
    stdout = ""
    stderr = ""
    status = "errors"  # default; overwritten below
    t0 = time.monotonic()
    try:
        try:
            proc = subprocess.run(
                argv,
                cwd=lean_root,
                env=scrubbed_env(),
                capture_output=True,
                text=True,
                # Lake/Lean output is UTF-8 (Unicode goal symbols ⊢ ∀ ⟨ ⟩ etc.).
                # On Windows, `text=True` alone decodes with cp1252 and crashes on
                # the first byte > 0x7F that isn't in cp1252. Force utf-8 with
                # errors="replace" — a single � in a diagnostic message is fine;
                # silently swallowing the entire run because of a decode crash is
                # not. Does not change verifier semantics: the markers we look
                # for (`error:`, `unsolved goals`, `declaration uses 'sorry'`,
                # `native_decide`, `sorry`) are pure-ASCII strings.
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )
            elapsed = time.monotonic() - t0
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            combined = stdout + stderr
            # INFRASTRUCTURE FAILURE, NOT A PROOF VERDICT. lake exiting
            # non-zero having said NOTHING means the compile never ran —
            # the process could not start. Left alone this returns
            # ok=False with EMPTY error text in ~0s, which every caller
            # reads as "the proof is wrong": the model is told its lemma
            # failed, ARM burns a revision round, and the trace records a
            # mathematical failure that never happened.
            #
            # MEASURED on mf20_amc12b_2021_p13_scaffold (2026-08-28): 27
            # of 41 verify calls came back in 0.0s with empty errors,
            # while the same code on the preceding cell had none. ARM
            # made 23 prove_lemma calls and banked ZERO lemmas
            # (lemma_store misses=41) — including for `Real.sin (2 * π) =
            # 0`, which Mathlib proves by `Real.sin_two_pi`. The cell
            # then failed for reasons that were never mathematical.
            # Consistent with resource exhaustion on this 7.7 GB host
            # (three lean/lake processes live, hours into a run).
            #
            # One retry, because the condition is transient by nature;
            # then a marked, loud failure so it can never again be
            # mistaken for a verdict.
            # A compile we KILLED is not a compile that failed to start.
            # Segment recycling SIGTERMs the in-flight lake (exit 143 /
            # -15; SIGKILL gives 137 / -9), which lands here looking
            # identical to a spawn failure: non-zero, no output. Left
            # alone it is retried three times inside a dying process and
            # then counted against the spawn-failure budget, corrupting
            # the one metric that tells a poisoned run from a real one.
            # Observed on mfx imo_2001_p6 segment 1: `lake exited 143
            # with NO output after 597.2s` — 597 s is a compile that RAN
            # and was interrupted, not one that never started.
            if proc.returncode in (143, -15, 137, -9) and not combined.strip():
                combined = (f"error: KILLED — lake terminated by signal "
                            f"(exit {proc.returncode}) after "
                            f"{elapsed:.0f}s; not a proof verdict and not "
                            f"a spawn failure.")
                print(f"[verify] {combined}", flush=True)
            for _delay in _INFRA_RETRY_DELAYS:
                if proc.returncode == 0 or combined.strip():
                    break
                print(f"[verify] lake exited {proc.returncode} with NO "
                      f"output after {elapsed:.1f}s — compile did not "
                      f"run. Waiting {_delay:.0f}s before retry.",
                      flush=True)
                time.sleep(_delay)
                try:
                    proc = subprocess.run(
                        argv, cwd=lean_root, env=scrubbed_env(),
                        capture_output=True, text=True,
                        encoding="utf-8", errors="replace",
                        timeout=timeout_s)
                    elapsed = time.monotonic() - t0
                    stdout = proc.stdout or ""
                    stderr = proc.stderr or ""
                    combined = stdout + stderr
                except Exception as _retry_exc:   # pragma: no cover
                    combined = combined + " " + type(_retry_exc).__name__
            if proc.returncode != 0 and not combined.strip():
                _INFRA_STATE["failures"] += 1
                combined = (
                    f"error: INFRASTRUCTURE — lake exited "
                    f"{proc.returncode} with no output after "
                    f"{len(_INFRA_RETRY_DELAYS)} retries; the compile did "
                    f"not run. This is NOT a proof verdict. "
                    f"({_INFRA_STATE['failures']} this process)")
                print(f"[verify] {combined}", flush=True)
            elif _INFRA_STATE["failures"]:
                # A recovered spawn means the pressure window passed.
                _INFRA_STATE["recovered"] += 1
            has_errors = bool(_ERROR_LINE_RE.search(combined)) or proc.returncode != 0
            has_implicit_sorry = bool(_SORRY_WARNING_RE.search(combined))
            ok = (
                (not has_errors)
                and (not used_sorry)
                and (not has_implicit_sorry)
                and (not used_native_decide)
            )
            result = CompileResult(
                # Hard reject: Lean accepting the file is NOT enough. We also reject
                # sorry (explicit or implicit) and native_decide — `native_decide`
                # is a verifier escape hatch (compiles to native code, bypasses the
                # kernel's trust path) and a proof using it must not count as ok.
                ok=ok,
                errors=combined,
                used_sorry=used_sorry or has_implicit_sorry,
                used_native_decide=used_native_decide,
                cwd=cwd_str,
                argv=list(argv),
                debug_dump_path=str(dump_stem.with_suffix(".lean"))
                                  if dump_stem is not None else None,
                elapsed_s=elapsed,
            )
            status = "ok" if ok else "errors"
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            # Timeout is always a reject; native_decide flag preserved for logging.
            result = CompileResult(
                False, f"timeout after {timeout_s}s",
                used_sorry, used_native_decide,
                cwd=cwd_str,
                argv=list(argv),
                debug_dump_path=str(dump_stem.with_suffix(".lean"))
                                  if dump_stem is not None else None,
                elapsed_s=elapsed,
            )
            status = "timeout"
    finally:
        try:
            path.unlink()
        except OSError:
            pass

    if dump_stem is not None:
        _write_sidecar(
            dump_stem, lean_path=dump_stem.with_suffix(".lean"),
            source=source, cwd=cwd_str, argv=argv,
            timeout_s=timeout_s, result=result,
            stdout=stdout, stderr=stderr, status=status,
        )

    # The axiom gate. Runs LAST, only on a source that has otherwise been
    # scored ok, so it costs a compile only on a would-be solve. A proof
    # that depends on anything beyond Lean's three standard axioms is
    # demoted to not-ok with the reason appended to `errors` — the caller
    # sees a failure, never a silent pass.
    if check_axioms and result.ok:
        from backend.axiom_check import ALLOWED_AXIOMS
        from backend.axiom_check import check_axioms as _run_axiom_gate

        def _probe(probe_src: str) -> tuple[bool, str]:
            sub = compile_lean(probe_src, timeout_s=timeout_s,
                               lean_project_root=lean_project_root,
                               check_axioms=False)
            # `#print axioms` writes to stdout; a probe carrying it is
            # expected to be error-free. Report errors as probe failure.
            return (not bool(_ERROR_LINE_RE.search(sub.errors)), sub.errors)

        verdict = _run_axiom_gate(source, _probe)
        result.axioms_ok = verdict.ok
        result.axiom_detail = verdict.detail
        if not verdict.ok:
            reason = ("inconclusive: " if verdict.inconclusive else "")
            result.ok = False
            result.errors = (
                f"{result.errors}\n"
                f"AXIOM GATE REJECTED: {reason}{verdict.detail}\n"
                f"(allowed: {sorted(ALLOWED_AXIOMS)})"
            )
            # A non-standard axiom dependency IS a sorry when it is
            # sorryAx; flag it so callers logging used_sorry see it.
            if any("sorryAx" in ax
                   for axs in verdict.offending.values() for ax in axs):
                result.used_sorry = True

    return result


def verify_proof(theorem_header: str, proof_block: str,
                 imports: str = "import Mathlib.Tactic.NormNum",
                 lean_project_root: Path | str | None = None,
                 debug_dump_path: Path | str | None = None,
                 timeout_s: int = 300,
                 check_axioms: bool = True) -> CompileResult:
    """Verify a (theorem header, proof block) pair as a standalone file.

    `timeout_s` covers the entire `lake env lean` invocation INCLUDING
    Mathlib import time. On this Windows host `import Mathlib.Tactic`
    alone takes ~180s; raise to 600+ for any verify that uses it.

    `check_axioms` is forwarded to compile_lean; see there.
    """
    src = f"{imports}\n\n{theorem_header} := by\n{proof_block}\n"
    return compile_lean(src, timeout_s=timeout_s,
                        lean_project_root=lean_project_root,
                        debug_dump_path=debug_dump_path,
                        check_axioms=check_axioms)
