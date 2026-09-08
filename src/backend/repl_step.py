"""Persistent Lean REPL backend.

Used by `run_dag.py --verify-backend repl`, which keeps one warm Lean
process and a per-import-set environment cache instead of paying a cold
`lake env lean` start per verification.

Why this exists. The compile-file backend (compile_step.step) re-imports
Mathlib on every tactic invocation, which on this Windows host costs
~14-25 minutes per `lake env lean` call once the user-generated source is
non-trivial. That makes per-tactic search wall-clock unworkable, as
measured on `mathd_numbertheory_100`: Lean compilation, not GPU, is the
bottleneck here.

What this is. A long-lived `lake exe repl` subprocess wrapped in a step-level
state machine. Lifecycle:

    session = LeanReplStepSession(lean_project_root=...)
    session.startup()            # launches REPL + sends `import Mathlib` once.
    session.start_problem(hdr)   # opens a tactic state for `<hdr> := by sorry`.
    res = session.step(prefix_tactics=(...), new_tactic="...")  # one tactic.
    ...
    session.close()

The session caches the REPL's per-state `proofState` IDs indexed by the
tuple of tactics that produced the state, so a best-first pop of a
previously-explored prefix is O(1) — no re-application.

Soundness. A REPL DONE (`goals == []`) is NOT a verified proof on its own.
The caller must still re-verify the full proof via
`backend.compile_verify.verify_proof`: the Lean verifier is the only oracle,
in a fresh session, with no `sorry`/`native_decide` shortcuts.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from .compile_step import StepOutcome, StepResult
from .sandbox import scrubbed_env

LEAN_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "lean"
REPL_EXE_DEFAULT = LEAN_DIR_DEFAULT / ".repl" / ".lake" / "build" / "bin" / "repl.exe"

# Per-step-time stats cap. We don't keep the raw list of every step time
# (could be thousands per run); we keep min/mean/max/n which is plenty for
# JSONL diagnostics.
_STEP_TIMES_HISTORY_CAP = 200


class ReplStartupError(RuntimeError):
    """Raised when the REPL subprocess could not start, or imports failed."""


def repl_messages_to_lake_errors(messages: list) -> str:
    """Convert REPL message dicts (severity/pos/data) into lake-style
    `repl.lean:<line>:<col>: error: <msg>` lines. Pure — testable without
    a REPL. Only error-severity messages are emitted; the synthesized
    shape matches what `search.proof_dag.parse_error_locations` expects,
    so REPL-backed verification plugs into leaf error attribution
    unchanged."""
    parts: list[str] = []
    for m in messages or []:
        if not isinstance(m, dict) or m.get("severity") != "error":
            continue
        pos = m.get("pos") or {}
        line = pos.get("line", 0)
        col = pos.get("column", 0)
        data = str(m.get("data", "")).strip()
        parts.append(f"repl.lean:{line}:{col}: error: {data}")
    return "\n".join(parts)


class ReplDeadError(RuntimeError):
    """The REPL process died (typically because we killed it on timeout)."""


class LeanReplStepSession:
    """One long-lived Lean REPL subprocess driven in tactic mode.

    Not threadsafe across multiple callers — use one session per worker.
    Per-call IO IS serialised internally via `_lock` so a stray verbose log
    print can't interleave with a tactic round-trip.
    """

    def __init__(self, lean_project_root: Path | str | None = None,
                 imports: str = "import Mathlib",
                 repl_exe: Path | str | None = None,
                 verbose: bool = False,
                 startup_timeout_s: float = 300.0):
        self._lean_root = Path(lean_project_root) if lean_project_root else LEAN_DIR_DEFAULT
        self._repl_exe = Path(repl_exe) if repl_exe else REPL_EXE_DEFAULT
        self._imports = imports
        self._verbose = verbose
        # Default startup budget when this session lazily restarts the REPL
        # (e.g. inside start_problem after a previous-problem kill). Also the
        # value used by any caller that invokes startup() with no explicit
        # timeout_s, which the constructor takes as `startup_timeout_s`.
        self._startup_timeout_s = float(startup_timeout_s)
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._env_id: int | None = None
        # prefix_tactics tuple → REPL proofState int.
        self._state_cache: dict[tuple[str, ...], int] = {}
        # Cached startup wall-clock; populated by startup().
        self._startup_s: float = 0.0
        # Bounded list of per-step elapsed seconds, for diagnostic summary.
        self._step_times: list[float] = []
        # When set, the REPL subprocess is no longer usable (e.g. we killed
        # it on timeout). The next start_problem will respawn.
        self._dead: bool = True
        # Additional import-set → env id cache for verify_whole. Distinct
        # from _env_id (the session's own imports): whole-proof callers
        # bring per-problem inferred imports, each loaded at most once per
        # REPL process. Invalidated on kill (envs die with the process).
        self._env_cache: dict[str, int] = {}
        # Header of the theorem currently loaded; used by step() to detect
        # a "start_problem was never called" misuse.
        self._current_theorem_header: str | None = None

    # ------------------------------------------------------------------
    # subprocess plumbing
    # ------------------------------------------------------------------

    def _spawn(self) -> None:
        """Launch the REPL subprocess. Idempotent only after a clean close."""
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self._repl_exe.exists():
            raise ReplStartupError(
                f"REPL binary not found at {self._repl_exe}. Build it once "
                f"with `cd {self._lean_root / '.repl'} && lake build` "
                f"(takes a few minutes)."
            )
        self._proc = subprocess.Popen(
            ["lake", "env", str(self._repl_exe)],
            cwd=self._lean_root,
            env=scrubbed_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Same rationale as compile_verify: force utf-8 + replace so a
            # stray byte never crashes the parent. The JSON wire protocol
            # is ASCII; the goal-text payloads inside it are UTF-8.
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._dead = False

    def _read_response(self) -> dict:
        """Read until a blank-line terminator and JSON-parse the buffer."""
        assert self._proc is not None and self._proc.stdout is not None
        buf: list[str] = []
        while True:
            line = self._proc.stdout.readline()
            if not line:
                # EOF — REPL died. Caller decides whether this is recoverable.
                stderr_tail = ""
                if self._proc.stderr is not None:
                    try:
                        stderr_tail = self._proc.stderr.read()
                    except Exception:
                        pass
                raise ReplDeadError(
                    f"REPL closed unexpectedly. stderr tail:\n{stderr_tail[:1500]}"
                )
            if line.strip() == "" and buf:
                return json.loads("".join(buf))
            buf.append(line)

    def _send(self, cmd: dict, *, timeout_s: float | None) -> dict:
        """Send one JSON command and read one JSON response.

        On timeout, KILLS the subprocess and marks the session dead. The
        next call to start_problem will respawn from scratch (which means
        re-importing Mathlib — expensive but necessary because the kill
        loses every proofState).
        """
        if self._dead or self._proc is None or self._proc.poll() is not None:
            raise ReplDeadError("REPL is not running")
        with self._lock:
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(cmd) + "\n\n")
            self._proc.stdin.flush()
            if timeout_s is None:
                return self._read_response()

            result_box: list[dict] = []
            error_box: list[BaseException] = []

            def reader() -> None:
                try:
                    result_box.append(self._read_response())
                except BaseException as e:  # noqa: BLE001
                    error_box.append(e)

            t = threading.Thread(target=reader, daemon=True)
            t.start()
            t.join(timeout_s)
            if t.is_alive():
                # Tactic hung past the budget — kill the REPL and mark dead.
                self._force_kill()
                raise subprocess.TimeoutExpired(cmd="repl-tactic", timeout=timeout_s)
            if error_box:
                raise error_box[0]
            if not result_box:  # pragma: no cover - defensive
                raise ReplDeadError("REPL reader returned no payload")
            return result_box[0]

    def _force_kill(self) -> None:
        """Hard-kill the REPL subprocess TREE and invalidate per-session
        state. The Popen target is `lake env repl.exe` — killing only the
        `lake` parent orphans the `repl.exe` child on Windows (observed
        2026-07-06: multi-GB repl.exe processes surviving their session
        across timeout respawns), so kill the whole tree."""
        if self._proc is None:
            return
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(self._proc.pid), "/T", "/F"],
                    capture_output=True, timeout=10,
                )
            self._proc.kill()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            pass
        self._dead = True
        self._proc = None
        self._env_id = None
        self._state_cache.clear()
        self._env_cache.clear()
        self._current_theorem_header = None

    # ------------------------------------------------------------------
    # public lifecycle
    # ------------------------------------------------------------------

    def startup(self, timeout_s: float | None = None) -> float:
        """Launch REPL + send the imports command. Returns wall-clock time.

        When `timeout_s` is None, falls back to `self._startup_timeout_s`
        (configured by the constructor / CLI --repl-startup-timeout). The
        explicit-argument path is preserved so tests and ad-hoc callers
        can pin a per-call budget independent of session defaults.
        """
        if timeout_s is None:
            timeout_s = self._startup_timeout_s
        t0 = time.monotonic()
        self._spawn()
        if self._verbose:
            print(f"[repl] starting up (cwd={self._lean_root})", flush=True)
        try:
            resp = self._send({"cmd": self._imports}, timeout_s=timeout_s)
        except subprocess.TimeoutExpired:
            raise ReplStartupError(
                f"REPL import `{self._imports}` exceeded {timeout_s}s — "
                f"check that `lake exe cache get` and `lake build Mathlib` "
                f"have been run in {self._lean_root}."
            )
        msgs = resp.get("messages", []) if isinstance(resp, dict) else []
        errors = [m for m in msgs
                  if isinstance(m, dict) and m.get("severity") == "error"]
        if errors:
            raise ReplStartupError(
                f"REPL import failed: {json.dumps(errors)[:1500]}"
            )
        env_id = resp.get("env") if isinstance(resp, dict) else None
        if env_id is None:
            raise ReplStartupError(
                f"REPL import returned no `env` id: {json.dumps(resp)[:500]}"
            )
        self._env_id = int(env_id)
        self._startup_s = time.monotonic() - t0
        if self._verbose:
            print(f"[repl] startup {self._startup_s:.1f}s "
                  f"(env={self._env_id})", flush=True)
        return self._startup_s

    @property
    def startup_s(self) -> float:
        return self._startup_s

    @property
    def imports(self) -> str:
        """The exact import string this session was constructed with — used
        by the caller to decide whether the next problem needs a new
        session (different imports require a fresh REPL because the env
        id from the original imports cannot serve a different one)."""
        return self._imports

    @property
    def startup_timeout_s(self) -> float:
        return self._startup_timeout_s

    # ------------------------------------------------------------------
    # whole-proof verification against a warm env
    # ------------------------------------------------------------------

    def _env_for_imports(self, imports: str | None, *,
                         timeout_s: float) -> int:
        """Env id for an import set, loaded at most once per REPL process.

        None or the session's own imports reuse the startup env; any other
        set is sent as its own `cmd` and cached. Raises ReplStartupError
        on import failure."""
        if imports is None or imports.strip() == self._imports.strip():
            if self._env_id is None:
                self.startup()
            assert self._env_id is not None
            return self._env_id
        key = imports.strip()
        cached = self._env_cache.get(key)
        if cached is not None:
            return cached
        try:
            resp = self._send({"cmd": key}, timeout_s=timeout_s)
        except subprocess.TimeoutExpired:
            raise ReplStartupError(
                f"REPL import set exceeded {timeout_s}s: {key[:200]}"
            )
        msgs = resp.get("messages", []) if isinstance(resp, dict) else []
        errors = [m for m in msgs
                  if isinstance(m, dict) and m.get("severity") == "error"]
        if errors:
            raise ReplStartupError(
                f"REPL import failed: {json.dumps(errors)[:1000]}"
            )
        env_id = resp.get("env") if isinstance(resp, dict) else None
        if env_id is None:
            raise ReplStartupError("REPL import returned no `env` id")
        self._env_cache[key] = int(env_id)
        return int(env_id)

    def verify_whole(self, theorem_header: str, proof_block: str, *,
                     imports: str | None = None,
                     timeout_s: float = 240) -> dict:
        """Check `<header> := by <proof>` against a WARM env.

        This is the fast inner-loop check — repair rounds cost seconds
        instead of a cold `lake env lean` compile. It is NOT the final
        oracle: a proof counts only if it re-verifies in a
        fresh session, so callers MUST confirm any ok=True result via
        `backend.compile_verify.verify_proof` before reporting solved.

        Returns {"ok": bool, "errors": str, "body_line_offset": int} —
        the same contract run_dag's verify_fn speaks, with REPL messages
        synthesized into lake-style `<file>:<line>:<col>: error:` text so
        line-based error attribution works unchanged.
        """
        offset = theorem_header.count("\n") + 1  # header lines before body
        if self._dead or self._proc is None or self._proc.poll() is not None:
            self._spawn()
            self.startup()
        try:
            env_id = self._env_for_imports(
                imports, timeout_s=self._startup_timeout_s)
            src = f"{theorem_header} := by\n{proof_block}\n"
            resp = self._send({"cmd": src, "env": env_id},
                              timeout_s=timeout_s)
        except subprocess.TimeoutExpired:
            return {"ok": False,
                    "errors": f"repl timeout after {timeout_s}s",
                    "body_line_offset": offset}
        except (ReplDeadError, ReplStartupError) as e:
            return {"ok": False,
                    "errors": f"repl error: {str(e)[:800]}",
                    "body_line_offset": offset}
        msgs = resp.get("messages", []) if isinstance(resp, dict) else []
        sorries = resp.get("sorries") or []
        err_text = repl_messages_to_lake_errors(msgs)
        ok = (not err_text) and (not sorries)
        if not ok and not err_text:
            err_text = "declaration uses 'sorry'"
        return {"ok": ok, "errors": err_text, "body_line_offset": offset}

    def start_problem(self, theorem_header: str, *,
                      timeout_s: float = 120) -> None:
        """Open the initial tactic state for a new theorem.

        Sends `<header> := by sorry` so the REPL hands back a fresh
        `proofState`. Resets the per-problem prefix → state cache and
        seeds it with the initial state at the empty prefix.

        If the REPL died (e.g. previous timeout killed it), respawns and
        re-imports before proceeding — this is the slow path; in steady
        state we just start_problem against the existing session.
        """
        if self._dead or self._proc is None or self._proc.poll() is not None:
            self._spawn()
            # Re-import after a kill — necessary because every proofState
            # the previous session held is gone with it. Use the session's
            # configured startup budget; the default is intentionally large
            # because Mathlib imports can take minutes on a cold cache.
            self.startup(timeout_s=self._startup_timeout_s)
        # Reset per-problem state.
        self._state_cache = {}
        self._current_theorem_header = theorem_header
        cmd_text = f"{theorem_header} := by sorry"
        resp = self._send(
            {"cmd": cmd_text, "env": self._env_id},
            timeout_s=timeout_s,
        )
        if not isinstance(resp, dict):
            raise ReplStartupError(
                f"unexpected REPL response opening theorem: {resp!r}"
            )
        msgs = resp.get("messages", []) or []
        # The "declaration uses 'sorry'" warning is expected — that's why we
        # added the sorry in the first place. Errors aren't.
        errors = [m for m in msgs
                  if isinstance(m, dict) and m.get("severity") == "error"]
        if errors:
            raise ReplStartupError(
                f"REPL rejected theorem header `{theorem_header[:200]}`: "
                f"{json.dumps(errors)[:1500]}"
            )
        sorries = resp.get("sorries", []) or []
        if not sorries:
            raise ReplStartupError(
                f"REPL response had no `sorries` entry for theorem "
                f"`{theorem_header[:200]}`: {json.dumps(resp)[:500]}"
            )
        proof_state = sorries[0].get("proofState")
        if proof_state is None:
            raise ReplStartupError(
                f"REPL sorry entry had no `proofState`: "
                f"{json.dumps(sorries[0])[:500]}"
            )
        self._state_cache[()] = int(proof_state)
        if self._verbose:
            print(f"[repl] start_problem [{theorem_header[:80]}] "
                  f"proofState={proof_state}", flush=True)

    def validate_header(self, theorem_header: str, *,
                         timeout_s: float = 60) -> dict:
        """Parse-check `theorem_header` without raising on Lean errors.

        Sends `<header> := by sorry` to the REPL and returns a structured
        result so the caller can distinguish "Lean rejected the syntax"
        from "REPL is broken / Mathlib not imported". Lazy-starts the
        REPL the same way `start_problem` does, so callers don't need
        a separate startup() invocation.

        Returns:
          {
            "ok":           bool,        # True iff a sorry-proofState was produced
            "errors":       list[dict],  # severity == "error" messages from Lean
            "first_error":  str | None,  # `data` field of the first error, truncated
            "first_error_pos": dict|None,# {"line": int, "column": int} of first error
            "normalised_header": str,    # the exact header that was sent
          }

        Side effects: on success, the session is now "loaded" for this
        header — `is_loaded_for` returns True and the cached proofState
        for `()` is populated, so a follow-up `step_for(header, …)`
        does NOT re-send the declaration. This avoids double-startup
        when the runner validates the header and then enters search.
        """
        if self._dead or self._proc is None or self._proc.poll() is not None:
            self._spawn()
            self.startup(timeout_s=self._startup_timeout_s)
        # Send the header with a sorry'd proof — same call shape as
        # start_problem, but DO NOT raise on errors.
        cmd_text = f"{theorem_header} := by sorry"
        try:
            resp = self._send(
                {"cmd": cmd_text, "env": self._env_id},
                timeout_s=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "errors": [{
                    "severity": "error",
                    "data": f"REPL timed out parsing header after {timeout_s}s",
                    "pos": None,
                }],
                "first_error": (
                    f"REPL timed out parsing header after {timeout_s}s"
                ),
                "first_error_pos": None,
                "normalised_header": theorem_header,
            }
        if not isinstance(resp, dict):
            return {
                "ok": False,
                "errors": [{
                    "severity": "error",
                    "data": f"unexpected REPL response: {resp!r}",
                    "pos": None,
                }],
                "first_error": f"unexpected REPL response: {str(resp)[:200]}",
                "first_error_pos": None,
                "normalised_header": theorem_header,
            }
        msgs = resp.get("messages", []) or []
        errors = [m for m in msgs
                   if isinstance(m, dict) and m.get("severity") == "error"]
        sorries = resp.get("sorries", []) or []
        if errors or not sorries:
            first = errors[0] if errors else {}
            return {
                "ok": False,
                "errors": errors,
                "first_error": (
                    str(first.get("data", ""))[:500]
                    if errors else
                    f"REPL accepted header but produced no `sorries`: "
                    f"{json.dumps(resp)[:200]}"
                ),
                "first_error_pos": first.get("pos") if errors else None,
                "normalised_header": theorem_header,
            }
        # Header parsed cleanly — register the proofState so a follow-up
        # step_for() skips re-sending the declaration.
        proof_state = sorries[0].get("proofState")
        if proof_state is not None:
            self._state_cache = {(): int(proof_state)}
            self._current_theorem_header = theorem_header
        return {
            "ok": True,
            "errors": [],
            "first_error": None,
            "first_error_pos": None,
            "normalised_header": theorem_header,
        }

    def is_loaded_for(self, theorem_header: str) -> bool:
        """Cheap predicate for the lazy-step wrapper: are we already in
        the tactic state of this theorem (i.e. start_problem was called
        for it AND the session hasn't died since)?"""
        return (
            (not self._dead)
            and self._proc is not None
            and self._proc.poll() is None
            and self._current_theorem_header == theorem_header
        )

    def step_for(self, theorem_header: str,
                 prefix_tactics: tuple[str, ...],
                 new_tactic: str, **kw) -> StepResult:
        """Lazy-start convenience wrapper: same signature as step(), but
        if the session hasn't been started for this theorem yet (or the
        REPL died and needs respawning), do that first.

        This is the entry point a caller wires in as `step_fn`. Letting
        the session lazy-start matters for tests that mock the step
        function: they never invoke step_for, so the
        real REPL is never spawned in test environments without a built
        `repl.exe`.
        """
        if not self.is_loaded_for(theorem_header):
            self.start_problem(theorem_header)
        return self.step(theorem_header, prefix_tactics, new_tactic, **kw)

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        finally:
            self._proc = None
            self._env_id = None
            self._state_cache = {}
            self._current_theorem_header = None
            self._dead = True

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def _ensure_state_for_prefix(self, prefix_tactics: tuple[str, ...],
                                 timeout_s: float) -> int | None:
        """Return the REPL proofState that follows `prefix_tactics`.

        Fast path: cache hit. Slow path: replay from root, applying each
        missing tactic. Slow path should never trigger in normal best-first
        operation — every state we ever push is cached when it was created.
        """
        cached = self._state_cache.get(prefix_tactics)
        if cached is not None:
            return cached
        if () not in self._state_cache:
            return None
        state = self._state_cache[()]
        for i, tac in enumerate(prefix_tactics):
            partial = tuple(prefix_tactics[:i + 1])
            if partial in self._state_cache:
                state = self._state_cache[partial]
                continue
            resp = self._send(
                {"tactic": tac, "proofState": state},
                timeout_s=timeout_s,
            )
            proof_state = resp.get("proofState")
            if proof_state is None:
                return None
            state = int(proof_state)
            self._state_cache[partial] = state
        return state

    def step(self, theorem_header: str,
             prefix_tactics: tuple[str, ...],
             new_tactic: str, *,
             imports: str = "import Mathlib",
             timeout_s: float = 30,
             lean_project_root: Path | str | None = None,
             debug_dump_path: Path | str | None = None) -> StepResult:
        """One tactic application — same signature as compile_step.step.

        `imports` and `lean_project_root` are accepted for signature parity
        with the compile backend but are intentionally ignored: the REPL
        was started with the project's imports once at startup. Override
        either requires a fresh session, which the caller should manage.

        `debug_dump_path` is honoured: when set, dumps the synthetic
        whole-file source (header + prefix + tactic) and a sidecar — same
        artefact a compile-backend run would produce, so triage scripts
        keep working across backends.
        """
        if self._current_theorem_header is None:
            return StepResult(
                StepOutcome.FAIL,
                error="repl: start_problem was not called before step",
            )
        # Cache lookup / replay.
        try:
            state = self._ensure_state_for_prefix(prefix_tactics, timeout_s)
        except subprocess.TimeoutExpired:
            elapsed = float(timeout_s)
            self._step_times.append(elapsed)
            self._trim_step_times()
            return StepResult(
                StepOutcome.FAIL,
                error=f"timeout after {timeout_s}s (replay)",
                elapsed_s=elapsed,
            )
        except ReplDeadError as e:
            return StepResult(
                StepOutcome.FAIL,
                error=f"repl-dead: {e}",
            )
        if state is None:
            return StepResult(
                StepOutcome.FAIL,
                error=f"repl: no cached state for prefix len={len(prefix_tactics)}",
            )

        # Optional debug dump — mimic the compile backend's artefact so a
        # triage script can grep across both backends uniformly.
        dump_stem: Path | None = None
        if debug_dump_path is not None:
            dump_stem = Path(debug_dump_path)
            body = "\n".join("  " + t for t in
                             list(prefix_tactics) + [new_tactic])
            src = f"{imports}\n\n{theorem_header} := by\n{body}\n"
            try:
                dump_stem.parent.mkdir(parents=True, exist_ok=True)
                dump_stem.with_suffix(".lean").write_text(src, encoding="utf-8")
            except Exception:
                dump_stem = None

        # Send the tactic.
        t0 = time.monotonic()
        try:
            resp = self._send(
                {"tactic": new_tactic, "proofState": state},
                timeout_s=timeout_s,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            self._step_times.append(elapsed)
            self._trim_step_times()
            res = StepResult(
                StepOutcome.FAIL,
                error=f"timeout after {timeout_s}s",
                elapsed_s=elapsed,
                debug_dump_path=str(dump_stem.with_suffix(".lean"))
                                  if dump_stem is not None else None,
            )
            self._write_sidecar_if_dumping(dump_stem, res, status="timeout",
                                           timeout_s=timeout_s)
            return res
        except ReplDeadError as e:
            return StepResult(
                StepOutcome.FAIL, error=f"repl-dead: {e}",
                debug_dump_path=str(dump_stem.with_suffix(".lean"))
                                  if dump_stem is not None else None,
            )
        elapsed = time.monotonic() - t0
        self._step_times.append(elapsed)
        self._trim_step_times()

        # Parse.
        msgs = resp.get("messages", []) if isinstance(resp, dict) else []
        errors = [m for m in msgs
                  if isinstance(m, dict) and m.get("severity") == "error"]
        if errors:
            err_text = "; ".join(str(m.get("data", "")) for m in errors)
            res = StepResult(
                StepOutcome.FAIL,
                error=f"error: {err_text}"[:500],
                elapsed_s=elapsed,
                debug_dump_path=str(dump_stem.with_suffix(".lean"))
                                  if dump_stem is not None else None,
            )
            self._write_sidecar_if_dumping(dump_stem, res, status="errors",
                                           timeout_s=timeout_s)
            return res
        proof_state = resp.get("proofState")
        if proof_state is None:
            res = StepResult(
                StepOutcome.FAIL,
                error=f"unexpected repl response: {json.dumps(resp)[:300]}",
                elapsed_s=elapsed,
                debug_dump_path=str(dump_stem.with_suffix(".lean"))
                                  if dump_stem is not None else None,
            )
            self._write_sidecar_if_dumping(dump_stem, res, status="errors",
                                           timeout_s=timeout_s)
            return res

        goals = resp.get("goals") or []
        new_handle = int(proof_state)
        new_prefix = tuple(prefix_tactics) + (new_tactic,)
        self._state_cache[new_prefix] = new_handle

        if not goals:
            # No goals → tactic closed the proof. NOT a verified proof —
            # the search engine must still verify_proof via compile_lean.
            res = StepResult(StepOutcome.DONE, elapsed_s=elapsed,
                             debug_dump_path=str(dump_stem.with_suffix(".lean"))
                                              if dump_stem is not None else None)
            self._write_sidecar_if_dumping(dump_stem, res, status="ok",
                                           timeout_s=timeout_s)
            return res
        goal_text = "\n".join(str(g) for g in goals) if isinstance(goals, list) \
                    else str(goals)
        res = StepResult(
            StepOutcome.PROGRESS,
            new_goal_text=goal_text[:1500],
            elapsed_s=elapsed,
            debug_dump_path=str(dump_stem.with_suffix(".lean"))
                              if dump_stem is not None else None,
        )
        self._write_sidecar_if_dumping(dump_stem, res, status="ok",
                                       timeout_s=timeout_s)
        return res

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def _trim_step_times(self) -> None:
        if len(self._step_times) > _STEP_TIMES_HISTORY_CAP:
            # Keep the most recent N — older values are less useful for
            # diagnosing the current run's behaviour.
            del self._step_times[:-_STEP_TIMES_HISTORY_CAP]

    def step_times_summary(self) -> dict:
        """Return {count, min, mean, max} of recent step elapsed seconds.

        Returns an empty dict if no steps have run yet — easier to
        serialise into JSONL without conditionals."""
        if not self._step_times:
            return {}
        n = len(self._step_times)
        return {
            "count": n,
            "min": round(min(self._step_times), 3),
            "mean": round(sum(self._step_times) / n, 3),
            "max": round(max(self._step_times), 3),
        }

    def _write_sidecar_if_dumping(self, dump_stem: Path | None,
                                   res: StepResult, *,
                                   status: str, timeout_s: float) -> None:
        if dump_stem is None:
            return
        try:
            sidecar = {
                "backend": "repl",
                "status": status,
                "outcome": res.outcome.value,
                "elapsed_s": round(res.elapsed_s, 3),
                "timeout_s": timeout_s,
                "error_prefix": (res.error or "")[:2000],
                "new_goal_text_prefix": (res.new_goal_text or "")[:2000],
            }
            dump_stem.with_suffix(".json").write_text(
                json.dumps(sidecar, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
