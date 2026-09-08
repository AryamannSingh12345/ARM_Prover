"""Compile-based tactic-step surrogate.

Also the home of the `StepOutcome` / `StepResult` vocabulary that the
persistent REPL session reuses.

Real Pantograph backend would expose `run(state, tactic) -> successor state`.
We approximate by:
  body = previous tactics + new tactic [+ sorry]
  invoke `lake env lean` and parse the output:
    - clean compile          -> proof complete (DONE)
    - "unsolved goals" only  -> tactic accepted, more goals remain (PROGRESS)
    - other error            -> tactic failed (FAIL)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .compile_verify import compile_lean


class StepOutcome(str, Enum):
    DONE = "done"          # proof is complete
    PROGRESS = "progress"  # tactic accepted, goals remain
    FAIL = "fail"          # tactic itself errored


@dataclass(slots=True)
class StepResult:
    outcome: StepOutcome
    new_goal_text: str = ""
    error: str = ""
    # Replay envelope inherited from CompileResult. cwd + argv let a
    # post-mortem reconstruct the exact `lake env lean ...` invocation;
    # debug_dump_path is the .lean file that was sent to lake when
    # --debug-dump-lean was on (None otherwise); elapsed_s is the lake
    # wall time for this attempt.
    cwd: str = ""
    argv: list[str] = field(default_factory=list)
    debug_dump_path: str | None = None
    elapsed_s: float = 0.0


_UNSOLVED_RE = re.compile(r"error:\s*unsolved goals\s*\n(.*?)(?=\n\S|\Z)", re.S)
_HAS_NON_UNSOLVED_RE = re.compile(r"error:(?!\s*unsolved goals)", re.M)


def step(theorem_header: str, prefix_tactics: tuple[str, ...],
         new_tactic: str, imports: str = "import Mathlib",
         timeout_s: int = 180,
         lean_project_root: Path | str | None = None,
         debug_dump_path: Path | str | None = None) -> StepResult:
    body_lines = list(prefix_tactics) + [new_tactic]
    body = "\n".join("  " + t for t in body_lines)
    src = f"{imports}\n\n{theorem_header} := by\n{body}\n"
    res = compile_lean(src, timeout_s=timeout_s,
                       lean_project_root=lean_project_root,
                       debug_dump_path=debug_dump_path)
    # Replay envelope carried by every return branch.
    diag = {
        "cwd": res.cwd,
        "argv": list(res.argv),
        "debug_dump_path": res.debug_dump_path,
        "elapsed_s": res.elapsed_s,
    }
    # Honest DONE only — `res.ok` already enforces no sorry / no implicit sorry
    # / no native_decide / no errors (see backend.compile_verify).
    if res.ok:
        return StepResult(StepOutcome.DONE, **diag)
    txt = res.errors
    # Timeout doesn't contain "error:" — treat as FAIL.
    if txt.startswith("timeout"):
        return StepResult(StepOutcome.FAIL, error=txt[:300], **diag)
    # Any error other than "unsolved goals" means the tactic itself is bad.
    if _HAS_NON_UNSOLVED_RE.search(txt):
        return StepResult(StepOutcome.FAIL, error=txt[:300], **diag)
    # Only "unsolved goals" -> tactic was accepted but goals remain.
    m = _UNSOLVED_RE.search(txt)
    if m:
        return StepResult(StepOutcome.PROGRESS,
                          new_goal_text=m.group(1).strip()[:1500],
                          **diag)
    # Conservative classification: anything else (res.ok=False with no error
    # line and no unsolved-goals marker — e.g. native_decide rejection, an
    # implicit `sorry` warning, or some other non-standard verifier reject)
    # is FAIL. The old fallback used to return DONE here when no sorry was
    # detected, which masked native_decide and other escape-hatch rejections.
    reason = (
        "non-classifiable verifier rejection"
        + (" (native_decide)" if res.used_native_decide else "")
        + (" (used_sorry)" if res.used_sorry else "")
    )
    return StepResult(StepOutcome.FAIL, error=(txt[:300] or reason), **diag)
