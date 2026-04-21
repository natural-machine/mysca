# 2026-04-21 — Logger migration for preprocessing + SCA-core pipelines

## Summary

Replaced bare `print()`-based output across the `mysca` package with a proper Python `logging` setup. Entrypoints now call a single `configure_logging(verbosity, logfile)` at startup; all modules use `logging.getLogger(...)` to emit at `DEBUG`/`INFO`/`WARNING` levels. Output goes to stderr (terminal) AND a logfile in the run's output directory — tqdm progress bars stay terminal-only and never touch the logfile because they bypass the logging framework. Three soft warnings around canonical-symbol filtering are now explicitly `WARNING`-level so they stand out from routine INFO chatter. Work done on branch `addison-dev-logging` in worktree `.claude/worktrees/logging-migration` (off `addison-dev`).

## Pre-session state

Session started at commit `b0247ee changed readme to refer to conda not mamba` on `addison-dev`.

To reach pre-session state:

```sh
git checkout addison-dev
```

Pre-session inventory (for context): 81 `print()` calls across 8 files, all gated on an integer `verbosity` kwarg (0/1/2); one `warnings.warn()` site at [run_sca.py:353](src/mysca/run_sca.py#L353); two unconditional prints at [preprocess.py:572](src/mysca/preprocess.py#L572) (GPU fallback) and [core.py:161](src/mysca/core.py#L161) (non-convergence). No `import logging` anywhere in `src/` or `tests/`. A closure helper `get_printv` at [run_sca.py:632-637](src/mysca/run_sca.py#L632-L637) was used as an ad-hoc level gate. `conftest.py` had no stdout-capture fixtures and no existing tests asserted on stdout content.

## Design decisions

1. **Dual output, always**: `configure_logging` attaches a `StreamHandler(sys.stderr)` and, when `logfile` is given, a `FileHandler`. Every record goes to both. Formats differ — stderr is terse (`%(levelname)s %(name)s: %(message)s`), logfile includes timestamps.
2. **tqdm left alone**: tqdm writes to `sys.stderr` directly with `\r`; it never goes through logging. So `FileHandler` never sees progress-bar frames. `--pbar` flag and verbosity stay orthogonal.
3. **Level mapping**: `--verbosity 0 → WARNING`, `1 → INFO` (default), `2+ → DEBUG`.
4. **Library hygiene**: [src/mysca/__init__.py](src/mysca/__init__.py) adds a `NullHandler` so bare `import mysca` is silent. Each module uses `logging.getLogger("mysca.<module>")`.
5. **`verbosity` kwarg kept on library functions** as a no-op compatibility shim — notebook callers passing `verbosity=2` don't break, but level is now globally controlled by `configure_logging`.
6. **`get_printv` deleted** — not exported, only used inside `run_sca.py`.
7. **`run_sca.py:353` → `logger.warning` AND `warnings.warn`** (both, so `-W error` still fires for programmatic callers).
8. **Third-party loggers (Biopython etc.) untouched** — `Bio.PDB` already silenced via `QUIET=True`; `captureWarnings(True)` funnels `warnings.warn` into our handlers.

## Changes made

### New files

- **`src/mysca/logging_config.py`** — defines `configure_logging(verbosity, logfile, *, capture_warnings=True)` and `verbosity_to_level(v)`. Idempotent; clears existing non-Null handlers on re-call so `run_full_pipeline` could reconfigure per-stage without duplicates.
- **`tests/test_logging.py`** — 18 tests covering level mapping, handler setup, idempotency, reconfiguration, `captureWarnings` integration, silent `import mysca` in a subprocess, NullHandler presence, plus the three canonical soft warnings.

### Modified files (library)

- [src/mysca/__init__.py](src/mysca/__init__.py) — added `NullHandler` on the `"mysca"` logger.
- [src/mysca/io.py](src/mysca/io.py) — `load_msa`: drops `verbosity > 1` gate for "Removed N seqs" (now unconditional `logger.warning` when N>0, `logger.debug` when N==0). Added new WARNING in the auto-detect branch when inferred alphabet has non-canonical symbols (`set(aa_syms) - set(AA_STD20)`).
- [src/mysca/preprocess.py](src/mysca/preprocess.py) — all ~20 status prints → `logger.info`. Unconditional GPU-fallback print at [line 499](src/mysca/preprocess.py#L499) → `logger.warning`.
- [src/mysca/core.py](src/mysca/core.py) — `fi0=0` diagnostic → `logger.debug`; ICA convergence status → `logger.info`; unconditional non-convergence print at [line 161](src/mysca/core.py#L161) → `logger.warning`.
- [src/mysca/tools.py](src/mysca/tools.py) — `remove_sequences_with_X` status → `logger.info`.

### Modified files (entrypoints)

- [src/mysca/run_preprocessing.py](src/mysca/run_preprocessing.py) — calls `configure_logging` right after `os.makedirs(outdir)`; all status prints → `logger.info`. Added WARNING for `--syms none` branch.
- [src/mysca/run_sca.py](src/mysca/run_sca.py) — deleted `get_printv` helper; all `printv(...)` call sites → `logger.info` (or `logger.warning` for ICA retries). `warnings.warn` at [line 353](src/mysca/run_sca.py#L353) now paired with `logger.warning`. `configure_logging` wired up with `scarun.log` in outdir.
- [src/mysca/run_full_pipeline.py](src/mysca/run_full_pipeline.py) — same pattern; logs to `full_pipeline.log`.
- [src/mysca/run_pymol.py](src/mysca/run_pymol.py) — logs to `pymol.log` in outdir (or no logfile when `--outdir` not given); removed per-function `verbosity` kwargs in `plot_scaffold_by_sectors` / `plot_scaffold_with_multiple_sectors`.

### Logger-name bug found during spot-check

Initially each entrypoint used `logger = logging.getLogger(__name__)`. That resolves to `"__main__"` when the module is run via `python -m mysca.run_preprocessing`, which does NOT propagate to the `"mysca"` parent — so the entrypoint's own WARNINGs silently skipped the configured handlers. Fix: hard-coded logger names in each entrypoint (`"mysca.run_preprocessing"`, `"mysca.run_sca"`, `"mysca.run_full_pipeline"`, `"mysca.run_pymol"`). Library modules still use `__name__` since they are always imported, never run as `__main__`.

### Test fixture fix

`tests/test_logging.py::TestLoadMsaSoftWarnings` tests passed in isolation but failed when the full suite ran. Cause: prior entrypoint tests invoked `configure_logging` which sets `propagate=False` on the `"mysca"` logger, preventing pytest's `caplog` from capturing records (caplog attaches at the root logger). Fixed by making the `reset_package_logger` fixture reset state BEFORE yielding too (clear non-Null handlers, set `propagate=True`, reset level), and applying the fixture to the `TestLoadMsaSoftWarnings` tests.

## Three canonical soft warnings (explicit WARNING level)

1. **Auto-detect branch** in [io.py](src/mysca/io.py) `load_msa`: when `mapping is None`, inferred `aa_syms` is checked against `AA_STD20`. If non-canonical symbols are present, logs `"Auto-detected alphabet contains non-canonical symbols: [...] — no filtering applied."`.
2. **Sequences removed** in [io.py](src/mysca/io.py) `load_msa`: when explicit `mapping` filters out any records, logs `"Removed N sequences containing excluded symbols."`. No longer gated on verbosity.
3. **`--syms none`** in [run_preprocessing.py](src/mysca/run_preprocessing.py): logs `"--syms none disables excluded-symbol filtering; using auto-detected alphabet."`.

## Post-session state

- All 370 tests pass (`PYTHONPATH=src python -m pytest tests/`). This includes the existing 352 tests plus 18 new logger tests.
- Spot-checked a real preprocessing run on `tests/_data/msas/msa02.faa` with `--syms none -v 1`:
  - Both canonical-symbol WARNINGs fire with correct `WARNING mysca.{io,run_preprocessing}` prefixes in stderr.
  - `outdir/preprocessing.log` written with timestamped lines, same content as stderr.
  - No tqdm progress output in the logfile.
- Git status at session end: all changes committed on `addison-dev-logging` (see commit below).

## Related commits

- Session end: commit on `addison-dev-logging` containing the logger module, NullHandler setup, all `print → logger` migrations, the three soft-warning additions, and the new `tests/test_logging.py`.

## Lingering items

- **`verbosity` kwargs are now vestigial** on `load_msa`, `preprocess_msa`, `run_sca`, `run_ica`, and `apply_ica`. They no longer affect behavior. Left in place for compatibility with any external notebook callers. Consider removing in a future cleanup pass once notebooks/external callers are audited.
- **Branch not merged**: work lives on `addison-dev-logging`. No PR opened yet.
- **Worktree retained** at `.claude/worktrees/logging-migration` — remove with `git worktree remove .claude/worktrees/logging-migration` once the branch lands.
