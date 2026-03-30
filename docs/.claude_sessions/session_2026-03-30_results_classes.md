# Session: Result Container Classes for SCA Outputs

**Date:** 2026-03-30

## Motivation

Preprocessing and SCA core results were saved via scattered `np.savez`, `json.dump`,
`sparse.save_npz`, and `np.save` calls spread across `run_preprocessing.py` and
`run_sca.py`. Loading was similarly ad-hoc. The goal was to introduce container
classes that:

1. Provide clean attribute-based access to results
2. Encapsulate save/load logic in one place
3. Keep the on-disk format documented and usable without mysca installed

## Changes Made

### New file: `src/mysca/results.py`

Two container classes plus shared file-name constants:

**`PreprocessingResults`**
- Attributes: `msa`, `msa_binary3d`, `retained_sequences`, `retained_positions`,
  `retained_sequence_ids`, `sequence_weights`, `fi0_pretruncation`, `args`,
  `sym_map`, `msa_obj_orig`
- Computed properties: `n_sequences`, `n_positions`
- `save(outdir)` — writes preprocessing_results.npz, preprocessing_args.json,
  sym2int.json, msa_binary2d_sp.npz, msa_orig.fasta-aln
- `load(dirpath)` — classmethod, reconstructs from directory
- `from_preprocess_output(msa, results_dict, sym_map, msa_obj_orig)` — factory
  that unpacks the `(msa, dict)` tuple returned by `preprocess_msa()`

**`SCAResults`**
- Attributes grouped by stage: core (Dia, conservation, sca_matrix, phi_ia, fi0,
  fia), optional large (Cijab_raw, fijab, Cij_raw), eigen (evals_sca, evecs_sca,
  significant variants), bootstrap (kstar, kstar_identified, cutoff, evals_shuff),
  ICA (v_ica, w_ica), sectors (groups, group_scores, t_dists_info, statsectors_msa,
  statsectors_seq, sca_matrix_sector_subset), args
- Computed properties: `n_sectors`, `n_positions`
- `save(outdir, save_all=False)` — writes all files matching original on-disk layout
- `load(dirpath)` — classmethod, loads whatever files exist (missing -> None)
- `from_core_output(sca_results_dict, args)` — factory that maps core key names
  (Di -> conservation, Cij_corr -> sca_matrix)

### Refactored: `src/mysca/run_preprocessing.py`

- Replaced ~40 lines of scattered save calls with:
  ```python
  results = PreprocessingResults.from_preprocess_output(...)
  results.save(outdir)
  ```
- Removed unused imports (json, sparse, AlignIO)
- Removed commented-out legacy save code
- File-name constants now imported from `results.py` (aliased to preserve
  existing names for backward compat with `run_sca.py` imports)

### Refactored: `src/mysca/run_sca.py`

- Loading preprocessed data now uses `PreprocessingResults.load(indir)` instead
  of manual np.load + sparse.load_npz + AlignIO.read
- Core SCA results built via `SCAResults.from_core_output(sca_results)`
- Eigen, bootstrap, ICA, and sector fields populated incrementally on the
  `results` object as computation progresses
- Single `results.save(OUTDIR, save_all=SAVE_ALL)` at the end replaces all
  scattered save calls
- File-name constants imported from `results.py`
- Removed unused imports (Bio.AlignIO, scipy.sparse)
- Note: the intermediate `np.save(evals_shuff)` after bootstrapping is
  intentionally preserved as a checkpoint for long runs

### Updated: `src/mysca/__init__.py`

- Exports `PreprocessingResults` and `SCAResults`

### New file: `tests/test_results.py`

18 tests covering both classes:

PreprocessingResults (5 tests):
- Computed properties (n_sequences, n_positions)
- save() creates all expected files
- Round-trip save/load preserves all fields
- from_preprocess_output() factory
- On-disk format readable with plain numpy/json (no mysca needed)

SCAResults (13 tests):
- Computed properties (n_sectors, n_positions)
- from_core_output() maps key names correctly, optional fields are None
- save() creates all expected files and subdirectories
- Round-trip for core, eigen, bootstrap, ICA, and sector fields (separate tests)
- Round-trip for statsectors_msa and statsectors_seq dicts
- save_all=True includes Cijab_raw/fijab; False excludes them
- Partial results: loading core-only gives None for optional fields
- On-disk format readable with plain numpy/json

## Design Decisions

- Plain attributes (not @property for every field) — simple, no immutability needed
- No computation in the classes — strictly data containers + persistence
- Same on-disk format as before — backward compatible
- `load()` is graceful — missing optional files result in None
- Key name mapping (Di -> conservation, Cij_corr -> sca_matrix) handled in
  `from_core_output()` factory
- numpy int64/float64 values cast to Python int/float before JSON serialization
  (fixed a TypeError discovered during testing)

## Verification

- All 325 existing tests pass
- All 18 new tests pass
