# 2026-04-22 — sca-plots replay CLI, CLI doc audit, position-mapping tests, results.info, preprocess log comparators

## Summary

Fifth session on 2026-04-22 (follow-on to
[session_2026-04-22_prealign_diag_plot_api_ic_log_scoping.md](session_2026-04-22_prealign_diag_plot_api_ic_log_scoping.md)).
Work falls into five pieces. No commits yet — all changes are in the
working tree; commit boundaries noted below for when the user is ready
to split this up.

1. **`sca-plots` replay CLI + demo scripts.** New `mysca/run_plots.py`
   module and `sca-plots` entrypoint. Regenerates diagnostic plots
   from persisted results (prealign / preprocessing / scacore) without
   rerunning the pipeline. Demo scripts bumped to `--n_components 10`
   so the SH3 demo exercises the top-N IC summary.
2. **Position-mapping audit and gap-closing tests (flagged critical).**
   Audited every helper that bridges original MSA ↔ processed MSA ↔
   raw sequence. Added five tests for previously-uncovered transforms:
   `get_rawseq_scores_in_groups`,
   `get_group_rawseq_scores_by_entry`,
   `log_top_ic_summary` with dropped columns AND a reference gap, and
   a round-trip consistency check across two fixtures.
3. **CLI documentation audit.** `docs/cli_reference.md` was out of
   sync: missing `--plot` on prealign, a stale `--weight_method`
   choice list on preprocess (claimed `v3,v4,v5,gpu`; actual
   `sparse,gpu`), and four missing `sca-core` flags
   (`--background`, `--n_components`, `--n_logged_comps`,
   `--assignment`). Fixed all of it, added a new `sca-plots` section,
   fixed `run_pymol.py`'s stale top docstring (referenced the removed
   `--groups_dir` flag), and filled in `run_preprocessing.py`'s
   placeholder docstring plus missing `help=` text on
   `--syms` / `--gapsym` / `--plot` / `--pbar` and the vague SCA
   parameter helps.
4. **`results.info` feature.** Added class-level `FIELD_DESCRIPTIONS`
   constant and `info()` method on both `PreprocessingResults` and
   `SCAResults`. `print(results.info())` renders a table: field name,
   current-instance value summary (shape/dtype/`(none)`), description.
   Covered by five new tests including a coverage invariant (every
   `__init__` attribute must have a description entry, no extras).
5. **Preprocess threshold log messages.** The four log lines in
   `preprocess_msa` that report filtered rows/columns now state the
   comparator explicitly ("Filtered N positions with gap frequency
   ≥ τ (0.4)") instead of the ambiguous "at threshold".

Test count: 895 → 911 (+16 new across the five pieces).

## Pre-session state

```sh
git checkout 507f0e0
```

Branch `addison-dev`. All work below is in the uncommitted working
tree on top of that commit.

## Piece 1 — `sca-plots` replay CLI + demo scripts

### Motivation

From the prior session's "Notes for next session": a `sca-plots
<results_dir>` command that regenerates every replayable figure is
straightforward now that prealign's `filter_history.json` and all IC
artifacts are persisted. Demo scripts still defaulted
`n_components = kstar`, which on the SH3 demo meant 2 ICs — not
enough to make the new top-N IC summary useful in the output.

The user rejected autodetection of stage directories — they wanted
explicit flags per stage. They also changed `--n_components all` to
`--n_components 10` for the demo ("10 is enough to get the point
across").

### Code changes

1. **[demo/SH3/scripts/step2_scacore.sh](../../demo/SH3/scripts/step2_scacore.sh)**,
   **[demo/SH3/scripts/step3_scacore_raw.sh](../../demo/SH3/scripts/step3_scacore_raw.sh)** —
   added `--n_components 10`.

2. **[src/mysca/run_plots.py](../../src/mysca/run_plots.py)** (new) —
   entrypoint with `--prealign DIR` / `--preprocessing DIR` /
   `--scacore DIR` flags (at least one required), optional shared
   `--imgdir`, and per-stage replay helpers. Each stage reads from its
   persisted artifacts via `PreprocessingResults.load()` /
   `SCAResults.load()` (plus `filter_history.json` for prealign) and
   calls the existing `mysca.pl.*` plotters. Default output goes into
   each stage's own `images/` subdirectory.

3. **[src/mysca/__main__.py](../../src/mysca/__main__.py)** and
   **[pyproject.toml](../../pyproject.toml)** — registered the
   `sca-plots` console script.

4. **[CLAUDE.md](../../CLAUDE.md)** — bumped the "four CLIs" line to
   five.

### Scope decision

`sca-plots` covers the 7 functions in `mysca.pl` only. The four
inline matplotlib figures currently living in
`run_sca.py::make_plots` (conservation, SCA-matrix imshow, spectrum
vs null, sector-subset) are deliberately out of scope — they belong
to deferred item (3) from the prior session (`make_plots` refactor
into `mysca.pl`). Once those are extracted, `sca-plots` picks them
up automatically.

### Tests

- **[tests/test_entrypoint_plots.py](../../tests/test_entrypoint_plots.py)** (new)
  — 6 tests covering each stage flag, `--imgdir` override, missing-dir
  error, and argparse rejection of empty args. Prealign test gated on
  `mafft` being on PATH.

  One subtlety: the `--scacore` test forces `--kstar 3 --n_components 3`
  because the msa06 fixture is tiny (5 positions) and would otherwise
  produce `kstar=1`, which makes all the EV/IC scatter sweeps (which
  need at least 2 columns) skip silently.

## Piece 2 — Position-mapping audit and gap-closing tests

### Motivation

The user flagged this as critical: "we need to once again ensure that
we have coverage for mapping input MSA positions to processed MSA
positions and finally to raw (unaligned) original sequences." Earlier
sessions had refactored this plumbing multiple times (gap-value flip,
IC summary log, kstar scoping) and regressions would be easy to miss.

### Audit findings

Three coordinate systems:
1. **Original MSA columns** — indices `[0..L_orig)` into the input
   alignment. Bridged to (2) by `retained_positions`.
2. **Processed MSA columns** — indices `[0..L_proc)` after
   preprocessing. `retained_positions[j]` is the original column of
   processed column `j`. Consumed by
   `run_sca.py:log_top_ic_summary` (line 760) for the processed →
   unprocessed map.
3. **Raw (unaligned) sequence positions** — 0-based residue indices
   in a specific ungapped input sequence. Bridged to (1) by
   `get_rawseq_indices_of_msa()` in `helpers.py` (returns `-1` for
   columns where the sequence has a gap).

Existing direct test coverage:

| Helper | Covered with both dropped cols AND gaps? |
|---|---|
| `get_rawseq_indices_of_msa` | yes |
| `get_conserved_rawseq_positions` | yes |
| `get_rawseq_positions_in_groups` | yes |
| `get_group_rawseq_positions_by_entry` | yes |
| `get_rawseq_scores_in_groups` | **no** (never tested directly) |
| `get_group_rawseq_scores_by_entry` | **no** (never tested directly) |
| `log_top_ic_summary` reference mapping | partial (had gap in ref but NO dropped columns) |
| Round-trip: processed → original → raw | no explicit test |

`sector_*_msapos.npy` / `group_*_msapos.npy` files on disk: confirmed
written as **processed-MSA coordinates** (via `groups[i]` from
`assign_positions_to_groups`, which indexes into `v_ica_normalized`).

### Code changes

None to production code. Four gap-closing tests added:

1. **[tests/test_helpers.py](../../tests/test_helpers.py)** —
   - `test_get_rawseq_scores_in_groups` — exercises the
     `retained_positions=[0,1,3,4]` + internal-gap fixture used by
     the existing `*_positions_in_groups` test, but with scores.
     Verifies scores are filtered wherever the corresponding raw
     index is a gap sentinel.
   - `test_get_group_rawseq_scores_by_entry` — parallel to
     `*_positions_by_entry`, exercising sub-selection of
     retained sequences (drops `sequence1`).
   - `test_processed_to_raw_mapping_consistency` — round-trip
     property check across two fixtures with dropped cols + internal
     gaps. For every retained `(sequence, processed_col)`, derives
     the raw residue index via the two-step mapping and
     cross-checks it against the aligned sequence directly
     (counting non-gap symbols strictly before the original column).

2. **[tests/test_ic_summary_log.py](../../tests/test_ic_summary_log.py)** —
   - `test_log_top_ic_summary_with_dropped_cols_and_reference_gap`
     — the combined case (`retained_positions=[0,2,3,5]` drops
     columns 1 and 4; reference `"AB-DEF"` has a gap at original
     column 2, which IS retained). Asserts the expected
     `[0, '-', 2, 4]` rendering of the reference line.

### Correctness note

While writing the test I initially got the expected reference line
wrong (`[0, '-', 1, 2]`). The helper was right; the test was. For
reference "AB-DEF" the raw residue indices at original columns 0..5
are `[0, 1, -, 2, 3, 4]`, and `retained_positions=[0,2,3,5]` selects
`[0, -, 2, 4]`. Mental model fix: raw indices are 0-based in the
ungapped sequence, NOT the position in the list of retained columns.

## Piece 3 — CLI documentation audit

### Motivation

User asked to verify that every argparse argument's `help=` text,
each entrypoint's top docstring, and `docs/cli_reference.md` are all
in sync and accurate, and to note in CLAUDE.md that changes to args
must ripple through all three locations.

### Findings and fixes

- **[docs/cli_reference.md](../../docs/cli_reference.md)**:
  - `sca-prealign`: added missing `--plot` and an "Output" list.
  - `sca-preprocess`: stale `--weight_method` choices (`v3,v4,v5,gpu`
    → now correct `sparse,gpu`); added missing `--gap_value` and
    `--plot`; moved `--input_format` from Required to Optional (it
    has a default); expanded every description with the actual
    semantics; removed the obsolete Weight Methods table.
  - `sca-core`: added `--background`, `--n_components`,
    `--n_logged_comps`, `--assignment` (all new in prior sessions);
    clarified the `assignment/weak_assignment` relationship;
    rewrote the Output list to match what `SCAResults.save()`
    actually produces.
  - Added a whole new `sca-plots` section.

- **[src/mysca/run_pymol.py](../../src/mysca/run_pymol.py)** — the
  example usage block at the top referenced a `--groups_dir` flag
  that was commented out in `parse_args` (the CLI moved to
  `--modes <npz>`). Replaced with a current-shape example.

- **[src/mysca/run_preprocessing.py](../../src/mysca/run_preprocessing.py)** —
  the top docstring's `COMMAND LINE ARGUMENTS` section was six
  placeholder lines (`"--gap_truncation_thresh : "` with nothing
  after the colon). Filled them in; added `help=` text to `--pbar`,
  `--plot`, `--syms`, `--gapsym`; replaced the vague
  "SCA parameter sequence_gap_thresh γ_{seq}" helps with real
  descriptions.

- **[CLAUDE.md](../../CLAUDE.md)** — added a "conventions" bullet:
  on any argparse change, update (a) the `help=` text, (b) the
  module top docstring, (c) `docs/cli_reference.md`, and (d)
  `results.py` container docstring if it affects on-disk format.

### Verified

All four `sca-*` console scripts (prealign, preprocess, core, plots)
produce non-erroring `--help` output after the edits. `sca-pymol` not
smoke-tested because it imports `pymol` at module load (not
installed in the default env).

## Piece 4 — `results.info` feature

### Motivation

User wanted to be able to query a results object for "a description
of each return value," with descriptions stored as a class-level
constant.

### Code changes

**[src/mysca/results.py](../../src/mysca/results.py)**:

- Two module-scope helpers: `_describe_value(val)` produces a
  compact `ndarray(3, 4) int64` / `dict (n=5)` / `(none)` summary;
  `_format_info_table(header, descriptions, value_fn)` renders the
  three-column table.
- `PreprocessingResults.FIELD_DESCRIPTIONS` — 11 entries, one per
  `__init__` attribute. Entry for `retained_positions` documents
  its bridge role: `original_col = retained_positions[processed_col]`.
- `PreprocessingResults.info()` — returns the formatted string.
- Same pattern on `SCAResults` — 28 entries. Several call out
  relationships explicitly (e.g. "Cij_raw is not persisted to
  disk", "sca_matrix_sector_subset shape is sum of group
  lengths squared").

### Design notes

- `info()` returns a string rather than printing. Caller does
  `print(results.info())`. This is testable without `capsys` and
  follows Python idiom.
- Descriptions are class-level, not instance-level — saves memory
  and makes `SCAResults.FIELD_DESCRIPTIONS["kstar"]` introspectable
  without constructing an instance.

### Tests

**[tests/test_results.py](../../tests/test_results.py)** — new
`TestFieldDescriptions` class with five tests:

- `test_preprocessing_field_descriptions_cover_all_init_args` and
  `test_sca_field_descriptions_cover_all_init_args` — coverage
  invariants. Iterates `vars(instance).keys()` and asserts the
  set equals `FIELD_DESCRIPTIONS.keys()`. No missing, no extras.
  This is what protects future authors from adding a new attribute
  and forgetting to describe it.
- `test_preprocessing_info_marks_populated_vs_none` — populated
  ndarrays render as `ndarray(3,)`, None fields render as `(none)`.
- `test_sca_info_shows_scalars_and_none` — scalar ints render as
  `int=3`.
- `test_field_descriptions_are_class_level_constants` — asserts
  `FIELD_DESCRIPTIONS` is in `vars(Class)` but not in
  `vars(instance)`.

## Piece 5 — Preprocess log comparator messages

### Motivation

Inline user request: "Instead of 'removed n sequences at threshold y'
we should say 'removed n sequences greater than/less than/greater or
equal to/etc. y'."

### Code changes

**[src/mysca/preprocess.py](../../src/mysca/preprocess.py)** —
rewrote the four `logger.info("Filtered %d … at threshold …")` lines
to state the comparator explicitly, derived from the direction of
each screen:

- `screen = x < τ` → "Filtered N positions with gap frequency ≥ τ (0.4)."
- `screen = x < γ_seq` → "Filtered N sequences with gap frequency ≥ γ_seq (0.2)."
- `screen = x >= Δ` → "Filtered N sequences with similarity to reference < Δ (0.2)."
- `screen = x < γ_pos` → "Filtered N positions with weighted gap frequency ≥ γ_pos (0.2)."

The comparator direction was already encoded in `filter_history`
entries as `"filter_direction": "above" | "below"`; the log lines
just weren't exposing it. Now they are.

No test changes needed — existing tests don't assert on these strings
(confirmed via `grep "at threshold" tests/`).

## Verification

```sh
env/bin/python -m pytest tests/
```

End-of-session: **911 passed**. Baseline at start of session: 895.
No regressions at any intermediate edit.

## Notes for next session

- **`mysca.project` subpackage** (planned in
  [plan_mysca_project_and_structure.md](plan_mysca_project_and_structure.md))
  — out-of-sample primary-sequence projection onto an existing SCA
  result. First target before `mysca.structure`. Default alignment
  via `mafft --add`; dispatch table to let HMMER/hmmalign be added
  without touching callers.
- **`mysca.structure`** — PDB / tertiary integration, depends on
  `mysca.project`. Planned after the `project` subpackage lands.
- **`make_plots` refactor** (still deferred from two sessions ago) —
  24 inline `plt.subplots` sites in `run_sca.py`. Extracting these
  into `mysca.pl` would give `sca-plots` full coverage and remove
  the scope carve-out noted above.
- **Uncommitted state.** Everything above is in the working tree. A
  reasonable commit split is one commit per piece (five commits),
  so the bundle can be reviewed in isolation:
  1. demo scripts + `sca-plots` CLI + entrypoint registration.
  2. Position-mapping audit tests (no production changes).
  3. CLI doc audit (`docs/cli_reference.md`, `run_pymol.py` and
     `run_preprocessing.py` docstrings, CLAUDE.md rule).
  4. `results.info` feature + tests.
  5. Preprocess threshold log comparator messages.
