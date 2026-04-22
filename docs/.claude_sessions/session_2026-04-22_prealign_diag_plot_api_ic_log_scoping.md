# 2026-04-22 — Prealign diagnostics, plot API refactor, IC summary log, kstar scoping

## Summary

Fourth session on 2026-04-22 (follow-on to
[session_2026-04-22_ic_handling_and_plotting_cleanup.md](session_2026-04-22_ic_handling_and_plotting_cleanup.md)).
Four focused commits, each addressing a follow-up the user queued during
the previous session plus one issue surfaced from running the demo:

1. **Prealign filter diagnostic.** `sca-prealign` now records a
   per-stage sequence-count history and (optionally) emits a one-panel
   waterfall plot, mirroring what the preprocessing stage already did.
2. **Plot API refactor.** Every plotter in `mysca.pl` now returns its
   `Figure` (multi-axis) or `Axes` (single-axis) and treats saving as
   opt-out via `save=True` / `outdir="."` / `filename=<default>`. Scripts
   keep working because `outdir` kept its positional slot; library users
   can get a figure back without touching disk.
3. **Top-N IC summary log.** After assignment, the sca-core pipeline
   writes a human-readable summary of the top ICs to the log:
   significance marker (`*` / `-`), eigenvalue, and MSA positions in
   processed / unprocessed / reference coordinates. A new
   `--n_logged_comps` flag (default 10) caps how many lines are
   printed.
4. **kstar scoping of expensive downstream steps.** Prompted by the
   question "what else depends on n_components?". When a user requests
   `--n_components` larger than `kstar`, the extra ICs still get
   computed and their first-class artifacts persisted, but two
   scaling hotspots (per-sequence sector mappings and
   t-distribution subplots) stop at `kstar`.

Test count: 891 → 895 (+26 new across the four commits; baseline
grew from 887 as earlier commits added their own tests).

## Pre-session state

```sh
git checkout 80a98f1
```

Branch `addison-dev`. All four commits below are linear on top of that.

## Commit 1 — `136f0ea` Prealign filter diagnostic

### Motivation

After the preprocessing-phase `plot_filter_history` /
`plot_filter_distributions` plots, the prealign phase (which clusters
and aligns raw FASTA) had no equivalent view. The per-stage sequence
count is a useful sanity check — did clustering remove 99% of your
sequences? Did alignment write the expected number of rows?

### Code changes

1. **[src/mysca/pl/plotting.py](../../src/mysca/pl/plotting.py)** — new
   `plot_prealign_filter_history(filter_history, imgdir)`. Single-panel
   waterfall of per-stage sequence counts with labeled deltas and
   colored bars (lightgray for the initial row, steelblue for filter
   stages). Adjacent siblings use the same conventions as
   `plot_filter_history`.

2. **[src/mysca/pl/__init__.py](../../src/mysca/pl/__init__.py)** —
   re-export `plot_prealign_filter_history`.

3. **[src/mysca/run_prealign.py](../../src/mysca/run_prealign.py)** —
   Build a `filter_history` list as the pipeline runs (initial →
   optional cluster → align), writing it to `filter_history.json` in
   the output dir. New `--plot` flag renders the diagnostic to
   `outdir/images/prealign_filter_history.png`.

### Tests

- **[tests/test_entrypoint_prealign.py](../../tests/test_entrypoint_prealign.py)** —
  `test_align_only` now also asserts `filter_history.json` contains
  `["initial", "align"]` stages with matching counts; new
  `test_align_plot_emits_filter_history_png` verifies `--plot`
  produces the expected file.

## Commit 2 — `ce2b187` Plot API refactor

### Motivation

From the audit in the previous session: plotting functions in
`mysca.pl` hard-coded `plt.savefig(...)` + `plt.close()` with a
positional `imgdir` argument. Library callers couldn't import a
plotter and do their own manipulation — they got a closed figure on
disk whether they wanted it or not.

### Code changes

Every function in [pl/plotting.py](../../src/mysca/pl/plotting.py) now
takes:

```python
def plot_X(..., outdir=".", *, filename=None, save=True):
```

- `outdir` — positional (keeps existing script calls working; default
  `.` for library use).
- `filename` — keyword-only; per-plot conventional default
  (`t_distributions.png`, `dendrogram.png`, ...). Data-dependent
  plots (`plot_data_2d/3d`) synthesize one from the axis indices and
  selected groups when not given.
- `save` — keyword-only, default `True`. When `True`, write to
  `{outdir}/{filename}` and close the figure. When `False`, return
  an open figure without touching disk.

Returns: `Axes` for single-axis plots (`plot_data_2d`, `plot_data_3d`,
`plot_prealign_filter_history`), `Figure` for multi-axis plots
(`plot_dendrogram`, `plot_sequence_similarity`, `plot_filter_history`,
`plot_filter_distributions`, `plot_t_distributions`). Matches the
user's spec.

A new `_maybe_save(fig, save, outdir, filename)` helper centralizes
"mkdir + savefig + close" so individual plotters don't duplicate it.

### Call-site impact

All existing positional callers in `run_sca.py`, `run_preprocessing.py`,
`run_prealign.py`, and `tests/test_plotting_guards.py` kept working
because `outdir` retained its positional slot. Only
`plot_dendrogram(Cij, nclusters=kstar, imgdir=IMGDIR)` (which used
`imgdir=` as a kwarg) needed rewriting to
`plot_dendrogram(Cij, IMGDIR, nclusters=kstar)`.

### Tests

Three new tests in
[tests/test_plotting_guards.py](../../tests/test_plotting_guards.py):

- `test_plot_data_2d_returns_axes_and_skips_save` — `save=False`
  returns an `Axes` and does not write.
- `test_plot_data_2d_custom_filename` — explicit `filename=` overrides
  the default data-derived name.
- `test_plot_data_3d_returns_axes_and_skips_save` — same contract for
  the 3D variant.

## Commit 3 — `ca499bf` Top-N IC summary log

### Motivation

After assignment, nothing summarized the IC → position mapping in a
form a user could scan in the terminal. Prior output was just "X
important positions (with repeats) / Y (without repeats)." The user's
sketch:

```text
IC 0: * λ_0=<1.234> | 4 (processed) MSA positions: [4, 6, 9, 13]
  -> (unprocessed) [<corresponding>]
  -> (reference structure) [<corresponding>]
```

### Code changes

**[src/mysca/run_sca.py](../../src/mysca/run_sca.py)** —
new module-scope `log_top_ic_summary(groups, kstar, evals_sca,
retained_positions, msa_obj_orig, reference_id, *, n_logged_comps=10)`.
For each of the top `n_logged_comps` ICs (capped by `len(groups)`):

- Significance marker: `*` if `i < kstar`, else `-`.
- `λ_i = float(evals_sca[i])` (the `i`-th sorted SCA eigenvalue).
- Processed MSA positions: `groups[i]` as a list of ints.
- Unprocessed MSA positions: `retained_positions[groups[i]]`.
- Reference residue indices (only when `reference_id` is specified
  and the id is present in the MSA): uses
  `get_rawseq_indices_of_msa` to derive a row of raw residue indices
  per MSA column; values are mapped to `'-'` where the reference has
  a gap at that column.

`--n_logged_comps` (default 10) controls the cap; 0 disables. Recorded
in `results.args` for reproducibility.

### Tests

Four unit tests in
[tests/test_ic_summary_log.py](../../tests/test_ic_summary_log.py) —
significance marker + position mappings, reference mapping with gap
handling, `n_logged_comps` cap behavior, and the `n_logged_comps=0`
no-op. A local-handler capture fixture is used (instead of `caplog`)
because `configure_logging` sets `mysca.propagate=False`, so caplog's
root-attached handler doesn't see these records in full-suite runs.

## Commit 4 — `502d7d7` kstar scoping of expensive downstream work

### Motivation

After running the demo, the user observed that only 2 ICs appeared in
the new summary log. Clarified that `len(groups) == n_components`
defaults to `kstar`. Follow-on question: "what else depends on
n_components?". Audit surfaced two hot spots where scaling with
n_components is costly without proportional analytical value for
non-significant ICs:

- **Per-sequence sector mappings** in
  `statsectors_msa` / `statsectors_seq` expanded every (IC group ×
  sector_seqid) pair. With `--sectors_for all` and
  `--n_components all` this balloons to `n_components × n_sequences`
  dict entries serialized to `statsectors_msa.npz`.
- **`plot_t_distributions`** draws one subplot per IC column.

### Code changes

**[src/mysca/run_sca.py](../../src/mysca/run_sca.py)** — the
per-sequence loop uses `for gidx in range(min(kstar, len(groups))):`
with an INFO line noting how many IC groups were not expanded per
sequence.

**[src/mysca/pl/plotting.py](../../src/mysca/pl/plotting.py)** —
`plot_t_distributions` gains an optional `max_plots` kwarg; the
entrypoint passes `kstar` so only significant ICs are rendered.
Library callers (import + call directly) still get all columns by
default.

Unchanged / still scales with n_components (by design):
ICA rotation, `fit_t_distributions`, `assign_positions_to_groups`,
`v_ica_normalized.npy` (full L × n_components), per-IC
`sector_{i}_msapos.npy` files, `sca_matrix_sector_subset.npy`.

### Tests

- **[tests/test_n_components.py](../../tests/test_n_components.py)** —
  new `test_per_sequence_sector_mapping_scoped_to_kstar` runs the
  entrypoint with `--kstar 2 --n_components all` and asserts:
  `statsectors_msa.npz` only contains `group_0_*` and `group_1_*`
  keys; per-IC `sector_i_msapos.npy` files still exist for every IC.
- **[tests/test_plotting_guards.py](../../tests/test_plotting_guards.py)** —
  three tests for the new `max_plots` semantics: cap at N, None means
  all, 0 returns None.

## Verification

```sh
env/bin/python -m pytest tests/
```

End-of-session: **895 passed**. No regressions introduced at any
intermediate commit. The test count climbed commit-by-commit: 884
(after prealign plot) → 887 (plot API) → 891 (IC log) → 895 (kstar
scoping).

## Notes for next session

- **Demo** still defaults `n_components = kstar`. If the user wants
  the SH3 demo to visibly exercise the top-N summary, the
  `scripts/step2_scacore.sh` / `scripts/step3_scacore_raw.sh` files
  need `--n_components all`. Offered in passing but not done.
- **Replay CLI** (item 6 from the earlier audit) — still deferred.
  With prealign's `filter_history.json` and all IC artifacts now
  persisted, a `sca-plots <results_dir>` command that regenerates
  every replayable figure is straightforward.
- **Inline plots in `make_plots`** — the 24 `plt.subplots` sites in
  `run_sca.py::make_plots` (item 5 from the audit) were not touched;
  they still hardcode colors/titles and duplicate save/close boilerplate.
