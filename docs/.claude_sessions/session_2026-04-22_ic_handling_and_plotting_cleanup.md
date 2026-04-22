# 2026-04-22 — IC handling, dead-code cleanup, and plotting review

## Summary

Follow-on session to [session_2026-04-22_gap_value_default.md](session_2026-04-22_gap_value_default.md).
Three distinct workstreams, shipped as ten focused commits:

1. **Independent-component handling.** Three new knobs on the sca-core
   pipeline: decoupled IC count from kstar (`--n_components`), overlap-
   vs-exclusive position assignment (`--assignment`), and graceful
   handling of the empty-group edge case when the t-distribution cutoff
   exceeds every position's IC projection.
2. **Dead-code cleanup.** Deleted the never-wired `run_full_pipeline`
   orchestrator, the two buggy `_compute_weights_v1/_v2` stubs, and the
   commented-out `get_groups()`. Renamed the remaining weight
   implementations so the CLI exposes only production methods.
3. **Plotting review + fixes.** Audited [src/mysca/pl](../../src/mysca/pl),
   then fixed four issues the audit surfaced: a missing `scipy.stats`
   import, lost `filter_history` (non-replayable plots), silent skips
   when axis indices are out of bounds, and an empty `pl/__init__.py`.

Test count: 857 → 883 (+26 new) across the session. No regressions at
any point.

## Pre-session state

```sh
git checkout f992b3f
```

Branch `addison-dev`. Ten commits below are all on top of that.

## Workstream 1 — IC handling (three commits)

### `d38d119` — Add `--n_components` to decouple IC count from kstar

Before: `run_sca` ran ICA on exactly `kstar` eigenvectors, so only kstar
ICs ever existed. After: `--n_components` accepts a positive integer or
`"all"`, defaults to `None` (= kstar, preserves prior behavior), and is
clamped to `[kstar, L]`. `kstar_identified` still records bootstrap
significance independently.

New on-disk artifact: `sca_results/n_components.txt` and an
`SCAResults.n_components` attribute. 9 tests in
[tests/test_n_components.py](../../tests/test_n_components.py) cover
default, `"all"`, explicit int, both clamp directions, and argparse
type validation.

### `400a5ac` — `--assignment {overlap,exclusive}` with overlap as default

Extracted the position→IC assignment step into
`assign_positions_to_groups()` (shared helper). `overlap` (new default)
lets a position belong to every IC group whose cutoff it clears;
`exclusive` preserves the previous max-projection behavior, and
`--weak_assignment` is only meaningful under `exclusive`.

6 tests in [tests/test_assignment.py](../../tests/test_assignment.py):
unit tests on a synthetic 3-IC fixture, a superset invariant (overlap
⊇ exclusive), and an end-to-end entrypoint parity check.

Key subtlety I got wrong and then corrected: `weak_assignment` does
*not* push the weak IC out of the result. It excludes the weak IC
from the max-projection tie-break — which means non-weak ICs can now
claim contested positions even when the weak IC's projection was
higher. Under `weak_assignment=(1,)`, a contested position typically
lands in both IC 0 *and* IC 1.

### `48dbd58` — Handle empty IC groups from t-distribution cutoff

Introduced `_safe_concat_int()`: `np.concatenate` that returns an empty
int array instead of raising when every input is empty. Routed the
three call sites in `run_sca` and the `msapos_to_groupidx` assembly in
[results.py](../../src/mysca/results.py) through it. Per-IC warning
when a group ends up empty; summary warning when all ICs are empty.

6 tests in [tests/test_empty_groups.py](../../tests/test_empty_groups.py)
including an entrypoint run at `pstar=100` (→ ppf(1.0)=+inf →
guaranteed-empty cutoff) that must complete with a 0×0 sector subset
on disk and empty group files.

## Workstream 2 — Dead code removal (two commits plus rename)

### `006c2ee` — Delete `run_full_pipeline.py`

900-line deletion. [pyproject.toml:49-53](../../pyproject.toml#L49-L53)
registers only `sca-prealign` / `sca-preprocess` / `sca-core` /
`sca-pymol`; `run_full_pipeline` was neither exposed as a CLI entry
point nor imported nor tested. Its logic duplicated
`run_preprocessing` + `run_sca` and was drifting on every feature
change (workstream 1 had to maintain parallel edits in both). Updated
[CLAUDE.md](../../CLAUDE.md) to drop the stale reference and list all
four live CLIs.

### `7a716ab` — Remove dead weight-computation versions and unused get_groups

Deleted:

- `_compute_weights_v1` / `_compute_weights_v2` in
  [preprocess.py](../../src/mysca/preprocess.py) — both had
  `raise RuntimeError("BUGGY VERSION OF WEIGHT COMPUTATION")` as their
  first line.
- `get_groups()` in [run_sca.py](../../src/mysca/run_sca.py) — the
  commented-out greedy-assignment path superseded by
  `assign_positions_to_groups()`.

Dropped the matching `"v1"` / `"v2"` dispatch cases.

### `ead4060` — Rename weight methods; restrict CLI to production choices

```text
_compute_weights_v5    → _compute_weights_sparse
_compute_weights_torch → _compute_weights_gpu
```

`compute_weights()` now accepts `"sparse"` / `"gpu"` for production use
and `"_v3"` / `"_v4"` / `"_v6"` for benchmark variants (the leading
underscore in the version string signals "not for routine use").

`sca-preprocess --weight_method` CLI choices shrank from
`{v3, v4, v5, gpu}` (default `v5`) to `{sparse, gpu}` (default
`sparse`). `gpu` still falls back to `sparse` when no torch
accelerator is detected. Callers wanting a benchmark variant can use
`preprocess_msa(weight_computation_version="_v3")` directly from
Python.

Tests ([test_preprocess.py](../../tests/test_preprocess.py),
[test_sparse_overflow.py](../../tests/test_sparse_overflow.py))
updated to the new version strings. No argstring fixtures referenced
`--weight_method`, so those stayed intact.

## Workstream 3 — Plotting review (one audit + four commits)

Audit scope: `src/mysca/pl/plotting.py` (seven exported functions) plus
the 24 inline plot sites in [run_sca.py](../../src/mysca/run_sca.py)
`make_plots`. Report covered correctness, live-usage coherence, and
post-pipeline replayability. All seven `pl/` functions have live
callers; the replayability blocker was `filter_history`. Proposed a
six-item list (1-6); user approved 1-4 as four separate commits.

### `018634b` — Explicitly import `scipy.stats`

One-line fix. `plot_t_distributions` calls `scipy.stats.t.pdf` but the
module only did `import scipy`. Worked today only because
[run_sca.py](../../src/mysca/run_sca.py) imports `scipy.stats` (via
`fit_t_distributions`) before plotting runs, registering it in
`sys.modules`. Replay contexts that import `pl` in isolation would
hit AttributeError.

### `01b0acd` — Persist `filter_history` in `PreprocessingResults`

Before: the per-stage filter diagnostic (retained counts, threshold
used, and the stat distribution that fed the filter) was returned in
`preprocess_msa`'s results dict but never written to disk, so the two
filter plots (`plot_filter_history`, `plot_filter_distributions`)
could only be regenerated during the original run.

After: new `filter_history.json` alongside the other preprocessing
artifacts. Serialization helpers
`_filter_history_to_jsonable` / `_filter_history_from_jsonable` handle
the numpy `stat_values` arrays (tolist on save, `np.asarray` on load).
Two new round-trip tests including a mixed-None-and-ndarray case.

### `ced81d5` — DEBUG log when plot axes are out of bounds

`plot_data_2d` / `plot_data_3d` had hard-coded axis indices up to 6
(e.g. `(5,6)`). With `--n_components` landing at, say, 3, those plots
silently disappeared. Now emit a DEBUG line identifying which plot was
skipped and why.

Tests ([tests/test_plotting_guards.py](../../tests/test_plotting_guards.py))
needed a local-handler capture fixture because pytest's `caplog`
doesn't see these records — [logging_config.py:51](../../src/mysca/logging_config.py#L51)
sets `mysca.propagate = False`, so once any entrypoint test runs
`configure_logging`, DEBUG records from the `mysca.pl.plotting` child
never reach the root logger caplog attaches to. Standalone caplog
worked; full-suite ordering broke it. The local `addHandler` + restore
pattern is robust to ordering.

### `d59457d` — Make `mysca.pl` the public surface

`pl/__init__.py` was empty; callers reached into the implementation
module directly (`from mysca.pl.plotting import ...`). Now re-exports
the seven plotters with an explicit `__all__`, and the two entrypoint
call sites use the flat `from mysca.pl import ...` form.

## Verification

Full suite at end of session: **883 passed** (857 → 878 after
workstream 1; 878 unchanged after workstream 2 because it was dead
code; 878 → 880 → 883 across the plotting commits).

## Deferred / lingering items

- **Plotting items 5 and 6 from the audit**:
  - (5) Extract the 24 inline plots in `run_sca.make_plots` into
    named functions in `pl/plotting.py` so they're individually
    callable and replayable.
  - (6) `sca-plots <results_dir>` CLI that reads a saved results
    directory and regenerates every plot. Depends on (5) and the
    `filter_history` persistence from this session.
- **Miscellaneous audit flags not acted on**: `plot_dendrogram`
  linkage-index check worth reviewing when next touched; the 24
  inline plots in `make_plots` hardcode colors/titles and share no
  style infrastructure.
- **`run_full_pipeline` deletion implications**: nothing imports it
  and the README/demo never used it, but if anyone has a local
  script that did, they'd need to rewire to `sca-preprocess` +
  `sca-core`. Flagged only in the commit message, not in a
  user-facing changelog.
- **Interrupted user message**: user started "Two more tasks, each
  its own commit:" but only listed item 1 (tasks 1-4 of the
  plotting audit). Item 2 is still pending.

## Queued for next session

User added three items just before end of session:

1. **Prealign-phase filter diagnostic plot** mirroring
   `plot_filter_history` / `plot_filter_distributions` for the
   sca-prealign stage.
2. **Plotting API refactor**: every plot function should *return*
   its figure/axis so callers using `from mysca.pl import …` can
   manipulate it. Saving should be opt-in — default filename per
   plot, overridable, and `outdir` parameter defaulting to `.`
   (entrypoints pass the image directory).
3. **IC summary in log output.** After the assignment step, log the
   top `N_logged_comps=20` ICs with:
   - significance marker (`*` if index < kstar, `-` otherwise)
   - eigenvalue
   - count of processed MSA positions
   - processed MSA position indices
   - corresponding unprocessed (pre-filter) MSA positions
   - corresponding reference-sequence positions (if a reference is
     specified), with a gap marker when the reference has no AA at
     that MSA column.

   Format per user's sketch:

   ```text
   IC 0: * λ_0=<1.234> | 4 (processed) MSA positions: [4, 6, 9, 13]
     -> (unprocessed) [<corresponding>]
     -> (reference structure) [<corresponding>]
   ```
