"""Mine a corpus proof-prior from Mathlib source.

Spec: PARTS 2 / 6 / 7 / 9 / 11 of the Mathlib-prior series.

Walk a Mathlib source tree, extract tactic-mode theorem/lemma proofs,
classify each tactic, record both:
  - the tactic transition itself, keyed on theorem features and the
    previous tactic class (PART 6);
  - a synthesised premise-citation move (`have h := <Premise>`) for
    each premise the tactic cites (PART 7), so the search can suggest
    a premise application even when no exact `have ... := P` template
    has been seen.

The output JSONL is read directly by `ProofPriorIndex.load(...)` and is
mergeable with the solved-log prior via `scripts/merge_proof_priors.py`.

Important: theorem NAMES from Mathlib are NEVER stored as features.
They are used only for `--exclude-subset` / `--exclude-theorem-id` /
`--exclude-name-regex` filtering and for diagnostic counters.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Make src/ importable without installing the package.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from search.corpus_features import extract_theorem_features  # noqa: E402
from search.proof_prior import (  # noqa: E402
    ProofPriorIndex, ProofPriorMove, ProofStateFeatures,
)
from search.proof_script_extract import (  # noqa: E402
    LeanDeclaration, extract_lean_declarations, split_tactic_script,
)
from search.tactic_classify import (  # noqa: E402
    canonicalise_for_namespace, canonicalise_tactic_text, classify_tactic,
    extract_rw_simp_lemmas, extract_used_premises, tactic_shape,
    _is_local_projection,
)


# Mathlib subdirectories that almost never contain reusable tactic
# scripts and that bloat scan time. The miner skips these by default;
# the user can override via --include-dirs.
_DEFAULT_SKIP_DIRS = frozenset({
    "Deprecated", "Archive", "Counterexamples", "Generated",
    ".lake", "test", "Tactic",  # Tactic/ is metaprograms, not proofs
})


def _resolve_skip(path: Path, repo_root: Path) -> bool:
    """True if `path` lies under any default-skip directory."""
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        rel = path
    return any(part in _DEFAULT_SKIP_DIRS for part in rel.parts)


def _iter_lean_files(
    root: Path,
    *,
    include_dirs: list[str],
    exclude_dirs: list[str],
    max_files: int | None,
) -> list[Path]:
    """Enumerate .lean files under `root` honouring include/exclude filters."""
    inc = [d.replace("\\", "/") for d in include_dirs]
    exc = [d.replace("\\", "/") for d in exclude_dirs]
    out: list[Path] = []
    for p in sorted(root.rglob("*.lean")):
        rel = str(p.relative_to(root)).replace("\\", "/")
        if inc and not any(rel.startswith(d) or f"/{d}/" in f"/{rel}" for d in inc):
            continue
        if any(rel.startswith(d) or f"/{d}/" in f"/{rel}" for d in exc):
            continue
        if _resolve_skip(p, root) and not inc:
            continue
        out.append(p)
        if max_files is not None and len(out) >= max_files:
            break
    return out


def _load_subset_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"exclude-subset file not found: {path}")
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s)
    return out


def _record_move(
    idx: ProofPriorIndex,
    features: ProofStateFeatures,
    move: ProofPriorMove,
) -> None:
    """Wrap add_observation so all corpus moves count as "solved" (the
    proof closed, since we mined from a published Mathlib proof)."""
    idx.add_observation(features, move, result="solved")


def _active_namespace_hint(decl: LeanDeclaration) -> str | None:
    """Return 'Nat' iff this declaration sits under a Mathlib Nat path.

    The bare-name canonicalisation map only fires for Nat for now (PART
    B fix). The detection is intentionally narrow: only file paths that
    contain a `/Nat/` segment OR the leaf `Nat.lean` count.
    """
    pp = (decl.file_path or "").replace("\\", "/")
    if "/Data/Nat/" in pp or "/Nat/" in pp or pp.endswith("/Nat.lean"):
        return "Nat"
    return None


def _mine_decl(
    decl: LeanDeclaration,
    idx: ProofPriorIndex,
    *,
    source_name: str,
    min_proof_lines: int,
    max_proof_lines: int | None,
) -> dict:
    """Mine one declaration. Returns per-decl counters."""
    counters = {"tactics": 0, "premise_moves": 0, "transitions": 0}
    if decl.proof_kind != "tactic":
        return counters
    tactics = split_tactic_script(decl.proof_text)
    if len(tactics) < min_proof_lines:
        return counters
    if max_proof_lines is not None and len(tactics) > max_proof_lines:
        # A pathologically long proof would dominate the prior. Cap by
        # truncating; we still get the prefix's transitions.
        tactics = tactics[:max_proof_lines]

    base_features = extract_theorem_features(decl.statement_text)
    namespace_hint = _active_namespace_hint(decl)
    previous_class: str | None = None
    for tactic in tactics:
        tactic_raw = tactic.strip().rstrip(";")
        if not tactic_raw:
            continue
        # PART E fix-up: rewrite bare Nat lemma names inside the FULL
        # tactic text so the mined `tactic_template` (e.g.
        # `rw [Nat.gcd_mul_lcm]`) is replayable verbatim outside the
        # Mathlib/Data/Nat namespace. The premise extractor below also
        # benefits — it sees the canonical names directly.
        tactic_clean = canonicalise_tactic_text(tactic_raw, namespace_hint)
        cls = classify_tactic(tactic_clean)
        # Per-step features inherit the theorem-level bag and slot in
        # the previous tactic's class. This is the only state-dependent
        # signal we can recover without a real Lean tracer.
        features = ProofStateFeatures(
            symbols=base_features.symbols,
            namespaces=base_features.namespaces,
            target_shape=base_features.target_shape,
            hypothesis_shapes=base_features.hypothesis_shapes,
            constants=base_features.constants,
            previous_tactic_class=previous_class,
            retrieved_premises=(),
        )
        raw_premises = extract_used_premises(tactic_clean)
        # PART B: canonicalise bare names against the active Nat namespace.
        # Local projections (`h.foo`, `a.gcd`) are unaffected because
        # they contain a `.` and the canon map only rewrites bare names.
        premises = [
            canonicalise_for_namespace(p, namespace_hint) for p in raw_premises
        ]
        # PART C: track which (raw) premises came from inside an rw/simp
        # bracket so we can tag the synthesised premise-prior row with
        # `rw_lemma`. Membership is checked against the RAW name (before
        # canonicalisation) because the bracket scanner extracts the
        # in-source spelling.
        rw_simp_set = set(extract_rw_simp_lemmas(tactic_clean))
        # Provenance for the audit: the Mathlib declaration this tactic
        # was mined from. Stored as metadata, never used for scoring.
        # Mathlib decl names live in a separate namespace from
        # miniF2F evaluation problem IDs, so the clean-eval audit
        # against an eval target will never match Mathlib provenance.
        provenance: tuple[str, ...] = (
            (decl.declaration_name,) if decl.declaration_name else ()
        )
        # 1) Record the tactic as a move.
        move = ProofPriorMove(
            tactic_template=tactic_clean,
            tactic_class=cls,
            premise=premises[0] if premises else None,
            source=source_name,
            tags=(source_name, "transition"),
            requires_instantiation=False,
            provenance=provenance,
        )
        _record_move(idx, features, move)
        counters["tactics"] += 1
        if previous_class is not None:
            counters["transitions"] += 1
            # PART D: corpus-derived abstract transition. The previous
            # class + this tactic's shape feed the runtime instantiator.
            idx.add_abstract_observation(
                previous_class, tactic_shape(tactic_clean),
                next_class=cls,
                source="abstract_continuation_prior",
                tags=("mathlib",),
                provenance=provenance,
                result="solved",
            )
        # 2) Premise priors: one synthesised `have h := Premise` move
        #    per cited premise. Lets the search suggest a citation even
        #    when no exact tactic template was mined.
        for raw_prem, prem in zip(raw_premises, premises):
            if not prem or _is_local_projection(prem):
                continue
            tags: tuple[str, ...] = (source_name, "premise_prior")
            if raw_prem in rw_simp_set:
                tags = tags + ("rw_lemma",)
            synth = ProofPriorMove(
                tactic_template=f"have h := {prem}",
                tactic_class="have_premise",
                premise=prem,
                source=source_name,
                tags=tags,
                requires_instantiation=True,
                provenance=provenance,
            )
            _record_move(idx, features, synth)
            counters["premise_moves"] += 1
        previous_class = cls
    return counters


def build(
    *,
    mathlib_root: Path,
    out_path: Path,
    max_files: int | None,
    max_theorems: int | None,
    include_dirs: list[str],
    exclude_dirs: list[str],
    min_proof_lines: int,
    max_proof_lines: int | None,
    source_name: str,
    excluded_ids: set[str],
    excluded_regexes: list[re.Pattern[str]],
    progress_every: int,
    smoothing_alpha: float,
    write_sidecar: bool,
) -> dict:
    """Mine the corpus. Returns stats dict; writes JSONL + optional sidecar."""
    idx = ProofPriorIndex()
    # Allow CLI to override the index's smoothing constant.
    if smoothing_alpha and smoothing_alpha > 0:
        idx.EPSILON = smoothing_alpha

    files = _iter_lean_files(
        mathlib_root,
        include_dirs=include_dirs,
        exclude_dirs=exclude_dirs,
        max_files=max_files,
    )
    n_files = len(files)
    n_decls = 0
    n_tactic_proofs = 0
    n_tactics_extracted = 0
    n_premise_moves = 0
    n_excluded = 0
    seen_names: set[str] = set()
    t0 = time.monotonic()

    for fi, file_path in enumerate(files, 1):
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        decls = extract_lean_declarations(text, str(file_path))
        for decl in decls:
            n_decls += 1
            # Exclusion (PART 9).
            if decl.declaration_name in excluded_ids:
                n_excluded += 1
                continue
            if any(rx.search(decl.declaration_name) for rx in excluded_regexes):
                n_excluded += 1
                continue
            if max_theorems is not None and len(seen_names) >= max_theorems:
                break
            if decl.proof_kind != "tactic":
                continue
            n_tactic_proofs += 1
            seen_names.add(decl.declaration_name)
            counters = _mine_decl(
                decl, idx,
                source_name=source_name,
                min_proof_lines=min_proof_lines,
                max_proof_lines=max_proof_lines,
            )
            n_tactics_extracted += counters["tactics"]
            n_premise_moves += counters["premise_moves"]
        if progress_every > 0 and fi % progress_every == 0:
            dt = time.monotonic() - t0
            print(f"[mathlib-miner] {fi}/{n_files} files  "
                  f"decls={n_decls} tactic_proofs={n_tactic_proofs} "
                  f"tactics={n_tactics_extracted} excluded={n_excluded} "
                  f"({dt:.1f}s)", file=sys.stderr, flush=True)
        if max_theorems is not None and len(seen_names) >= max_theorems:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx.save(out_path)

    sidecar_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    sidecar = {
        "mathlib_root": str(mathlib_root),
        "source_name": source_name,
        "max_files": max_files,
        "max_theorems": max_theorems,
        "min_proof_lines": min_proof_lines,
        "max_proof_lines": max_proof_lines,
        "smoothing_alpha": smoothing_alpha,
        "include_dirs": list(include_dirs),
        "exclude_dirs": list(exclude_dirs),
        "clean_holdout": bool(excluded_ids or excluded_regexes),
        "excluded_theorem_count": len(excluded_ids),
        "excluded_subset": sorted(excluded_ids),
        "excluded_name_regexes": [rx.pattern for rx in excluded_regexes],
    }
    if write_sidecar:
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    return {
        "total_files_seen": n_files,
        "declarations_seen": n_decls,
        "tactic_proofs_seen": n_tactic_proofs,
        "tactics_extracted": n_tactics_extracted,
        "premise_moves_recorded": n_premise_moves,
        "rows_excluded": n_excluded,
        "unique_feature_keys": len(idx._by_key),
        "moves_recorded": sum(len(b) for b in idx._by_key.values()),
        "output": str(out_path),
        "sidecar": str(sidecar_path) if write_sidecar else None,
        "elapsed_s": round(time.monotonic() - t0, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mathlib-root", required=True,
                    help="Path to the Mathlib source directory "
                         "(the one containing Data/, Algebra/, ...).")
    ap.add_argument("--out",
                    default="data/proof_prior/mathlib_proof_prior_moves.jsonl")
    ap.add_argument("--max-files", type=int, default=200,
                    help="Cap on .lean files scanned. v1 default is "
                         "deliberately small; raise to 2000 for a "
                         "broader pass, omit (=None) for full Mathlib.")
    ap.add_argument("--max-theorems", type=int, default=50000,
                    help="Hard cap on distinct theorems mined.")
    ap.add_argument("--max-proof-lines", type=int, default=80,
                    help="Per-proof tactic-line cap. Long proofs are "
                         "truncated, not skipped — the prefix is still "
                         "useful for the prior.")
    ap.add_argument("--min-proof-lines", type=int, default=1)
    ap.add_argument("--include-dirs", action="append", default=[],
                    help="Restrict to these top-level Mathlib subdirs "
                         "(repeatable, e.g. 'Data/Nat', 'Algebra').")
    ap.add_argument("--exclude-dirs", action="append", default=[],
                    help="Skip these top-level Mathlib subdirs.")
    ap.add_argument("--source-name", default="mathlib_corpus",
                    help="Value of ProofPriorMove.source on emitted rows.")
    ap.add_argument("--smoothing-alpha", type=float, default=1e-3,
                    help="Additive smoothing constant for the prior "
                         "probability calculation. Default 1e-3 matches "
                         "ProofPriorIndex.EPSILON.")
    ap.add_argument("--progress-every", type=int, default=50,
                    help="Print one diagnostic line every N files. 0 = silent.")
    ap.add_argument("--write-sidecar", action="store_true", default=True,
                    help="Write <out>.meta.json alongside the JSONL.")
    ap.add_argument("--no-write-sidecar", dest="write_sidecar",
                    action="store_false")
    # PART 9: holdout exclusion.
    ap.add_argument("--exclude-subset", default=None,
                    help="Path to a subset file (one theorem name per "
                         "line, '#' comments). Decls whose declaration_name "
                         "matches are skipped from the mining pass.")
    ap.add_argument("--exclude-theorem-id", action="append", default=[],
                    metavar="ID", help="Skip decls with this exact name. "
                                        "Repeatable.")
    ap.add_argument("--exclude-name-regex", action="append", default=[],
                    metavar="REGEX", help="Skip decls whose name matches "
                                          "this regex. Repeatable.")
    args = ap.parse_args()

    mathlib_root = Path(args.mathlib_root).resolve()
    if not mathlib_root.is_dir():
        print(f"[mathlib-miner] not a directory: {mathlib_root}",
              file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()

    excluded_ids: set[str] = set()
    if args.exclude_subset:
        sub_path = Path(args.exclude_subset)
        if not sub_path.is_absolute():
            sub_path = (repo_root / sub_path).resolve()
        excluded_ids |= _load_subset_ids(sub_path)
    for tid in args.exclude_theorem_id:
        tid = (tid or "").strip()
        if tid:
            excluded_ids.add(tid)
    excluded_regexes = [re.compile(p) for p in args.exclude_name_regex]

    stats = build(
        mathlib_root=mathlib_root,
        out_path=out_path,
        max_files=args.max_files if args.max_files > 0 else None,
        max_theorems=args.max_theorems if args.max_theorems > 0 else None,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
        min_proof_lines=args.min_proof_lines,
        max_proof_lines=args.max_proof_lines if args.max_proof_lines > 0 else None,
        source_name=args.source_name,
        excluded_ids=excluded_ids,
        excluded_regexes=excluded_regexes,
        progress_every=args.progress_every,
        smoothing_alpha=args.smoothing_alpha,
        write_sidecar=args.write_sidecar,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
