# CLAUDE.md

Project notes for Claude Code. See [README.md](README.md) for the user-facing intro.

## What this is

`mysca` — Statistical Coupling Analysis (SCA) pipeline for identifying co-evolving
amino acid sectors in protein multiple sequence alignments. Exposes six CLIs:
`sca-prealign`, `sca-preprocess`, `sca-core`, `sca-pymol`, `sca-plots` (replay
plotter that regenerates diagnostic figures from persisted results), and
`sca-project` (project primary-sequence records onto an existing SCA result).

## Layout

- `src/mysca/` — package source. Core pipeline in `preprocess.py` + `core.py`;
  CLIs in `run_prealign.py` / `run_preprocessing.py` / `run_sca.py` /
  `run_pymol.py` / `run_plots.py` / `run_project.py`. On-disk result
  containers in `results.py`; out-of-sample projection lives in
  `project/` (library) + `run_project.py` (CLI).
- `tests/` — pytest suite. Fixtures in `conftest.py`; shared test MSAs/PDBs in
  `tests/_data/`; tmp output in `tests/_tmp/` (gitignored).
- `docs/.claude_sessions/` — per-session notes (prior agent work, context).
- `demo/SH3/` — worked example; `demo/SH3/out/` contents are gitignored.

## Running

```bash
conda activate ./env
python -m pip install -e '.[dev]'   # editable install
pytest tests                         # full suite; uses pytest config in pyproject.toml
```

Tests pass with `PYTHONPATH=src python -m pytest tests/` when running from
an ad-hoc worktree.

## Conventions

- Python 3.12+; no `# what-the-code-does` comments — only non-obvious *why*.
- Logging via `mysca.logging_config.configure_logging(verbosity, logfile)` —
  entrypoints call it once; library modules use `logging.getLogger("mysca.<mod>")`.
  tqdm progress bars bypass the logger; `--pbar` is orthogonal to `--verbosity`.
- Canonical AA alphabet is `AA_STD20` + `-` gap; non-canonical symbols trigger
  WARNING-level log entries (see `io.load_msa`, `run_preprocessing.py --syms`).
- Prefer editing existing files over creating new ones; no speculative abstractions.
- When adding / renaming / removing / changing default of an `argparse`
  argument in any `run_*.py` entrypoint, update **all three** surfaces in
  the same commit:
  1. the `help=` string on the argument itself,
  2. the `COMMAND LINE ARGUMENTS` / `OUTPUTS` section of the module's top
     docstring,
  3. the matching section in [docs/cli_reference.md](docs/cli_reference.md).
  If the change affects what the entrypoint writes to disk, also update
  the docstring of the corresponding container in
  [src/mysca/results.py](src/mysca/results.py).
