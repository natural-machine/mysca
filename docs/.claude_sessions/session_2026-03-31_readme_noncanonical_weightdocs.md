# Session: README Rewrite, Non-canonical Symbol Filtering, Weight Method Docs

**Date:** 2026-03-31

## Changes

### README.md rewrite (committed as 0c9aac2)

Comprehensive rewrite of the project README:
- Replaced placeholder description with actual project summary
- Fixed `conda activate` typo in global env setup
- Added PyMOL subsection under Setup
- Documented all three CLI tools with key options, linking to the full CLI reference
- Added Python API section showing `PreprocessingResults` and `SCAResults` usage
- Added Demo section
- Cleaned up references formatting

### New file: `docs/cli_reference.md` (committed as 0c9aac2, updated later)

Full CLI reference with complete argument tables for `sca-preprocess`, `sca-core`,
and `sca-pymol`, including all arguments, defaults, and descriptions. Later updated
to link to weight_methods.md.

### Non-canonical symbol filtering in `load_msa`

**Problem:** When an MSA contains amino acids outside the SymMap's known symbols
(e.g. "X" with the standard 20 AA map), `load_msa()` raised a `KeyError` at the
matrix construction step (io.py line 78). The existing `exclude_syms` mechanism
only filtered explicitly listed symbols, not arbitrary unknown characters.

**Fix:**

`src/mysca/mappings.py`:
- Added `NONCANONICAL` sentinel as the default for `exclude_syms`
- When active (default), any symbol not in `sym_list` (aa_syms + gap) is excluded
- Added `is_excluded(sym)` method that handles both modes
- Passing `exclude_syms=""` explicitly disables filtering (old behavior)
- Updated `DEFAULT_MAP` to use the new default

`src/mysca/io.py`:
- Filter logic now uses `mapping.is_excluded()` instead of checking
  `mapping.exclude_syms` directly
- Removed redundant second filter in matrix construction list comprehension

`tests/test_io.py`:
- Updated test: `SYMMAP1` (default) now drops X-containing sequences instead of
  raising KeyError → expects `(2, 10)` shape
- Added `SYMMAP1_NO_EXCLUDE` with `exclude_syms=""` to verify KeyError still occurs
  when filtering is explicitly disabled

**Index safety:** `retained_sequences` indices from `preprocess_msa` are relative
to the MSA matrix it receives (post-`load_msa` filtering). Since `msa_obj_orig` is
also the post-filter MSA object, indices remain consistent through `run_sca.py`.

### New file: `docs/weight_methods.md`

Documentation of all sequence weight computation methods:
- Status table (v1-v6, gpu) with CLI exposure and test coverage
- Per-method descriptions
- Recommendations (v5 default, gpu for large MSAs)
- Resolved uint8 overflow issue

### Updated: `docs/cli_reference.md`

Added links to `weight_methods.md` from the `--weight_method` argument row
and the Weight Methods section.

## Verification

All 352 tests pass.
