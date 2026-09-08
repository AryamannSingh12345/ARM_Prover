# ARM Prover

A theorem prover for **Lean 4 + Mathlib**. A language model writes the proof
steps, this code organises and checks them, and Lean decides whether the proof
is real. Nothing counts as solved unless Lean compiles it from scratch, with no
`sorry` and no shortcuts.

The repo contains two ways of proving a theorem, so they can be compared:

| Mode | Command | What it does |
|---|---|---|
| **ARM prover** | `src/eval/run_dag.py` | The model breaks the theorem into a list of smaller `have` steps. Each step is proved on its own. Steps that fail get repaired one at a time, and if repair keeps failing, the ARM loop asks the model to invent new helper lemmas. |
| **Baseline** | `src/eval/run_minif2f.py` | The same model writes the whole proof in one go. You can ask for several tries, and you can feed the Lean error back and let it try again. |

The baseline is there to answer one question: when the ARM prover solves a
problem, could the model have solved it on its own anyway?

## How the ARM loop works

ARM is the part of the prover that invents new lemmas. It runs when a step is
stuck and normal repair is not fixing it. Six stages:

1. **Restate the stuck step.** The failing step becomes a small theorem of its
   own, with the original assumptions and any earlier `have` steps it needs.
2. **Ask for helper lemmas.** The model replies with lemma *statements* only
   (no proofs yet), plus the tactic that would finish the stuck step if those
   lemmas were available.
3. **Reject circular lemmas.** If a proposed lemma just restates the goal, it
   is thrown out before anything is compiled. Otherwise the model could
   "prove" the theorem by assuming it.
4. **Check the lemmas would be enough.** Each lemma is filled in with `sorry`
   and the file is compiled once. This is a cheap question: *if these lemmas
   were true, would the stuck step close?* A compile with `sorry` in it can
   never be reported as a solve.
5. **Prove each lemma.** Only now is each lemma proved separately, and each one
   has to pass the real Lean check. Earlier proved lemmas can be used by later
   ones. If a lemma proof fails, the failure is sent back to the model and it
   gets a limited number of retries.
6. **Keep all of them or none.** The proof is only changed once every lemma has
   been proved. Unused lemmas are dropped, the proved ones are added to the top
   of the file, and the normal proof loop carries on.

## What you need

* Python 3.11–3.13
* [`elan`](https://github.com/leanprover/elan), which installs Lean
* A built copy of Mathlib inside `lean/` (downloaded, not compiled)
* An API key for Anthropic or OpenAI, or the address of a server that speaks
  the OpenAI API

The repo holds code only. The problem sets and the Mathlib index are downloaded
or built on your machine. The steps below are the full list.

## Setup

The commands below are written for a bash shell. On Windows, use Git Bash or
WSL, or translate them to PowerShell.

**On Windows, clone into a short path such as `C:\ARM_Prover`.** Mathlib's
build files have long names, and Windows cuts paths off at 260 characters by
default. From a deep folder, `lake build` fails part-way with
`failed to create file ... .olean`. A short folder avoids it. (You can instead
turn on long path support in Windows, but the short folder is easier.)

### 1. Install the Python package

```bash
python -m pip install -e ".[dev]"
```

### 2. Put your API key in the OS keychain

```bash
python -c "import keyring; keyring.set_password('prover','anthropic','<key>')"
python -c "import keyring; keyring.set_password('prover','openai','<key>')"
```

The code reads keys from the keychain, or from `ANTHROPIC_API_KEY` /
`OPENAI_API_KEY` if you prefer environment variables. It never reads `.env`
files. Keys are also stripped out of the environment before Lean is started
(`src/backend/sandbox.py`), because Lean code written by a model could
otherwise print them with `#eval IO.getEnv`.

### 3. Install Lean and Mathlib

```bash
cd lean
lake exe cache get     # downloads the prebuilt Mathlib; do not compile it yourself
lake build
mkdir -p Generated     # the prover writes each attempt into this folder
cd ..
```

Lean is pinned to `leanprover/lean4:v4.30.0-rc2` in `lean/lean-toolchain`, and
Mathlib to commit `ab9605ce3663` in `lean/lake-manifest.json`. Leave both as
they are.

`lean/Generated/` is not in git, but it has to exist. The prover writes files
there and will not create the folder for you.

`lake build` compiles a file that imports all of Mathlib, which needs several
GB of free memory. If it is killed or runs out of memory, you can carry on: the
prover only needs the files that `lake exe cache get` downloaded. The check
below is what actually tells you the setup works.

Check that Lean works before going further:

```bash
python -c "import sys; sys.path.insert(0,'src'); from backend.compile_verify import verify_proof; print(verify_proof('theorem t : 2 + 2 = 4', '  norm_num'))"
```

This takes a few minutes because Lean has to load the library. You should see
`ok=True` and `axioms_ok=True`.

### 4. Get the problems

These come from other projects with their own licences, so you download them:

```bash
git clone https://github.com/yangky11/miniF2F-lean4 data/miniF2F
# PutnamBench: download the Lean 4 files and put them in data/putnambench/Test/
```

The prover looks for one file per problem, named `<problem_id>.lean`, inside
whatever folder you pass to `--bench-dir`:

| `--bench-dir` | problems |
|---|---|
| `data/miniF2F/MiniF2F/Test` | miniF2F test set |
| `data/putnambench/Test` | PutnamBench |
| `data/conjectures` | the nine conjectures included in this repo |

### 5. Build the Mathlib index

By default the prover searches Mathlib for lemma names to suggest to the model
(`--premises bm25`). That search needs an index, which is about 40 MB and is
not in git. **A default run will crash without it:**

```bash
python scripts/build_mathlib_graph.py --mathlib-root lean/.lake/packages/mathlib --out-dir data/mathlib_graph
```

This writes `data/mathlib_graph/nodes.jsonl` (about 214,000 declarations) and
`edges.jsonl` (about 155,000 edges). It takes a couple of minutes. Rebuild
it if you ever change the Mathlib version. To skip it, run with
`--premises none`.

Two more indexes are optional and the prover works fine without them:
`scripts/build_statement_index.py` (only used by `--recognize-leaves`), and
`data/premise_scores.sqlite`, which is created as needed.

## Running it

### ARM prover

```bash
python src/eval/run_dag.py \
  --subset data/dev_subset.txt \
  --bench-dir data/miniF2F/MiniF2F/Test \
  --abduce-lemmas --abduce-mode theory \
  --abduce-theory-trigger stuck --abduce-theory-rounds 2 \
  --verify-backend compile --verify-timeout 1200 \
  --run-id my_arm_run
```

What the flags do:

| Flag | Meaning |
|---|---|
| `--subset` | a text file listing which problem ids to try |
| `--bench-dir` | the folder holding those problems |
| `--abduce-lemmas` | turn lemma invention on (off by default) |
| `--abduce-mode theory` | use the six-stage loop above. The other option, `eager`, makes the model send lemmas together with their proofs in one go |
| `--abduce-theory-trigger` | when to start inventing lemmas. `stuck` waits until a step has failed twice, so you need `--repair-rounds 2` or more. `always` starts on the first failure, which is what you want if you are measuring ARM itself |
| `--abduce-theory-rounds` | how many times the model may revise its lemmas |
| `--verify-backend` | `compile` starts Lean fresh each time. `repl` keeps one Lean process running and is much faster over many problems |
| `--run-id` | names the output file |

Without `--abduce-lemmas` the same command runs the prover with lemma
invention switched off, which is the comparison run for measuring what ARM
adds.

There is also a launcher that picks a model from a menu and fills in sensible
flags. It prints the full command before running it:

```bash
python scripts/run.py gpt            # pick an OpenAI model and run
python scripts/run.py claude --list  # just show the model list
```

### Baseline

```bash
python src/eval/run_minif2f.py \
  --subset data/dev_subset.txt \
  --bench-dir data/miniF2F/MiniF2F/Test \
  --k 8 --run-id my_baseline_run
```

`--k` is how many proofs to ask for per problem (default 1).
`--self-correct-rounds 2` sends the Lean error back and asks for a fix.

### A quick, cheap run

```bash
python src/eval/run_dag.py --subset data/dev_subset.txt \
  --bench-dir data/miniF2F/MiniF2F/Test \
  --premises none --sketch-attempts 1 --repair-rounds 0 --run-id smoke
```

### Where the output goes

Both commands write one line of JSON per problem to
`results/<run-id>.jsonl`, and a longer record of the run to
`results/traces/<run-id>/<problem_id>.trace.jsonl`.

You can stop a run and start it again with the same command: problems already
in the results file are skipped.

* `python scripts/summarize.py` — totals across runs
* `python scripts/spend.py` — how many tokens each run used

Be aware that Lean, not the model, takes most of the time. One compile with
`import Mathlib` takes minutes. Use `--verify-backend repl` for anything longer
than a few problems.

## What is in each folder

```
src/backend/    Talking to Lean: run a file, read the errors, check the axioms,
                keep a Lean process warm, hide API keys from Lean
src/policy/     Talking to the model: Anthropic, OpenAI and vLLM clients, and
                the prompts
src/search/     The prover itself. proof_dag.py holds the ARM loop and the
                step-by-step proof structure. The rest is lemma search,
                retrieval, import guessing and tactic templates
src/eval/       The two run commands, problem loading, replay
src/diag/       Writes the per-problem trace file
lean/           The Lean project, with the pinned Lean and Mathlib versions
data/           Lists of problem ids, and nine conjectures written out in Lean
scripts/        Index builders, problem-set preparation, the launcher, and
                small tools for reading results
tests/          The test suite. No Lean and no network needed
```

The `.txt` files in `data/` are just lists of problem names, one per line. The
actual theorem statements come from `--bench-dir`. The exception is
`data/conjectures/`, which holds nine open problems (Frankl, Cramér and three
from number theory) written out in Lean, so you can run the prover on something
real without downloading anything.

## Tests

```bash
pytest -m "not live"     # fast: Lean is faked, nothing is downloaded
pytest -m live           # slow: really runs Lean
```

## Rules the code keeps

* **Only Lean decides.** A proof counts if, and only if, Lean compiles it in a
  fresh session with no `sorry`, no `admit`, no `native_decide`, no errors and
  no timeout. Do not loosen `src/backend/compile_verify.py`.
* **`sorry` is only used to ask "would this be enough?"** Those compiles can
  never produce a solve.
* **Helper lemmas are all kept or all dropped.** If even one of them fails to
  prove, the proof and the file are left exactly as they were.
* **A compile that never started is not a failed proof.** If Lean exits without
  printing anything, that is recorded as a machine problem, not as a wrong
  theorem.
* **Changing `src/search/` or `src/backend/` makes old results out of date.**
  Run them again rather than reusing old numbers.
