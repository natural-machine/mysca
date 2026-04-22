# 2026-04-22 — Gap-integer default flipped to 0; gap_value CLI flag

## Summary

Made the SCA pipeline gap-position-agnostic and flipped the default gap
integer from `len(aa_list)` (historically 20 for `AA_STD20`) to `0`. Fixed
several code paths that silently assumed gap-at-end, generalized the sparse
one-hot encoders, and added a `--gap_value` CLI flag to `sca-preprocess`.
Existing Excel-derived test ground truth (which was computed at gap-at-end)
is preserved by pinning the entrypoint-preprocessing argstrings to an
explicit `--gap_value <N>`.

## Pre-session state

```
git checkout c8ac674
```

Branch: `addison-dev`. No staged changes at session start; these edits
started from an audit of how `SymMap.gapint` was used in `src/`.

## Motivation

`SymMap` already exposed `gapint` as an attribute, and in principle the gap
integer could be anywhere in `[0, len(aa_list)]`. In practice several places
in `src/mysca/` hardcoded "gap is last":

- `np.eye(NUM_SYMS)[msa][:,:,:-1]` in three files — drops the last column
  regardless of where the gap actually is.
- `get_onehotmsa_sparse` / `get_onehotmsa_sparse_nogap` explicitly raised
  `ValueError` unless `gap == num_aa`.
- `background_freq_array` was sized `num_aas` and indexed via
  `sym_map[aa]`, which overflows when gap isn't at the end.

The immediate trigger was the user's preference for gap=0 as the default.
Before flipping the default, the silent assumptions had to go.

## Code changes

### Gap-agnostic one-hot construction

1. **[src/mysca/preprocess.py](../../src/mysca/preprocess.py)** — new
   `onehot_without_gap(msa, num_syms, gapint)` helper: `np.eye(num_syms)[msa]`
   then `np.delete(..., gapint, axis=-1)`. The resulting axis-2 ordering
   always matches `sym_map.aa_list`, regardless of where the gap sits in
   `sym_list`. `preprocess_msa` uses it at the initial xmsa construction.

2. **[src/mysca/run_full_pipeline.py:419](../../src/mysca/run_full_pipeline.py#L419)**,
   **[src/mysca/run_sca.py:317](../../src/mysca/run_sca.py#L317)** — shuffled
   one-hot construction for the null-distribution bootstrap routed through
   `onehot_without_gap`. Fixed an existing latent bug in `run_sca.py` where
   `NSYMS = msa_binary3d.shape[-1]` misleadingly called `num_aas` "NSYMS"
   and then built `np.eye(NSYMS + 1)`; renamed `NSYMS = len(sym_map)`.

### Generalized sparse encoders

3. **[src/mysca/preprocess.py](../../src/mysca/preprocess.py)** —
   `get_onehotmsa_sparse` drops its `gap == num_aa` assertion; its body
   already worked for any gap position (each `(pos, symbol)` pair maps to a
   unique sparse column).
   `get_onehotmsa_sparse_nogap` was genuinely broken for non-end gap: the
   `mask = a < num_aa` filter both excluded a valid AA (`a == num_aa`) and
   included the gap when `gap == 0`. Replaced with `mask = a != gap` and
   `aa_idx = a - (a > gap)` to remap to consecutive AA indices.

### Background-frequency alignment

4. **[src/mysca/run_sca.py:262](../../src/mysca/run_sca.py#L262)**,
   **[src/mysca/run_full_pipeline.py:254](../../src/mysca/run_full_pipeline.py#L254)**,
   **[src/mysca/core.py:76](../../src/mysca/core.py#L76)** — build the
   length-`num_aas` background array by iterating `sym_map.aa_list`, so the
   axis ordering of `qa` matches the AA-only xmsa axis. Prior code used
   `sym_map[aa]` which returned gap-indexed integers and could overflow.

### SymMap API

5. **[src/mysca/mappings.py](../../src/mysca/mappings.py)** — added
   `gap_value: int = 0` kwarg. Valid range `[0, len(aa_list)]`. `sym_list`
   is now `aa_list[:gap_value] + [gapsym] + aa_list[gap_value:]`. Default
   flipped from "append at end" to `0`. Also added a comment on the
   `NONCANONICAL = "noncanonical"` sentinel explaining its identity-compare
   contract.

6. **[src/mysca/results.py](../../src/mysca/results.py)** —
   `PreprocessingResults.load()` now reconstructs a real `SymMap` from the
   on-disk flat `sym2int.json` dict via the new `_symmap_from_sym2int`
   helper. On-disk format is unchanged (still a flat `{symbol: int}` dict
   for external consumers); convention "-" identifies the gap. If `"-"`
   isn't present, the raw dict is returned unchanged so `sym_map[sym]`
   lookups still work.

### CLI

7. **[src/mysca/run_preprocessing.py](../../src/mysca/run_preprocessing.py)**
   — added `--gap_value INT` (default 0), plumbed through to both
   `--syms default` and `--syms <custom>` code paths.

## Test changes

1. **[tests/test_sym_maps.py](../../tests/test_sym_maps.py)** — updated
   `test_default_mapping` to `exp_gapint = 0`; updated the
   `mapping_and_expecteds` fixture to `exp_gapint = 0`; added parametric
   `test_gap_value_parameter` and `test_gap_value_out_of_range_raises`.

2. **[tests/test_core.py](../../tests/test_core.py)**,
   **[tests/test_preprocess.py](../../tests/test_preprocess.py)** — wrapped
   the existing Excel-ground-truth parametrize with
   `@pytest.mark.parametrize("gap_value", [0, "mid", "end"])`. Ground truth
   is invariant under gap repositioning: `fia`, `Dia`, `fijab`, `Cijab`,
   `Di` are indexed by AA in `aa_list` order; `retained_sequences`,
   `retained_positions`, and `sequence_weights` depend only on
   `msa == GAP` comparisons. `"end"` preserves the pre-default-flip layout
   exactly — the Excel ground truth was computed at that layout and is not
   re-derived.

3. **[tests/test_gap_value.py](../../tests/test_gap_value.py)** (new) —
   targeted tests for gap-position invariance:
   - `test_onehot_without_gap_matches_aa_list_order` — dense helper's axis-2
     column `i` always encodes `aa_list[i]`.
   - `test_sparse_nogap_matches_dense` — sparse encoder (gap excluded)
     agrees with the dense helper after reshape.
   - `test_sparse_with_gap_preserves_similarity` — sparse encoder (gap
     included) produces the same pairwise-similarity dot-product matrix as
     the reference (gap-at-end) layout, confirming permutation-invariance.
   - `test_sparse_rejects_out_of_range_gap`.

4. **[tests/test_entrypoint_preprocessing.py](../../tests/test_entrypoint_preprocessing.py)**
   — added `test_main_gap_value_flag` parametrized over
   `gap_value ∈ {0, 3, 4, 8}` on the 8-AA `msa04` fixture. Asserts:
   - `sym2int.json` places gap at the requested integer and keeps AAs in
     input order;
   - `msa_binary3d.shape == (20, 19, 8)` regardless of gap position;
   - `retained_sequences`, `retained_positions`, `sequence_weights` are
     byte-identical across gap values;
   - the filtered integer MSA encodes AAs consistently with `sym2int` and
     never contains the gap integer (msa04 has no gaps after filtering).

5. **[tests/test_results.py](../../tests/test_results.py)** — updated
   `test_round_trip` to compare `loaded.sym_map.sym2int` against the raw
   dict the test saved, since `load()` now returns a `SymMap`.

6. **Argstring fixtures pinned to gap-at-end** to preserve Excel-derived
   integer-MSA ground truth (files are under
   [tests/_data/entrypoint_tests/preprocessing/argstrings/](../../tests/_data/entrypoint_tests/preprocessing/argstrings/)):
   `argstring4a.txt` and `argstring5a.txt` gained `--gap_value 8`;
   `argstring6a.txt`, `argstring7a.txt`, `argstring8a.txt` gained
   `--gap_value 4`.

## Verification

Full suite: `env/bin/python -m pytest tests/` — **857 passed**. Previous
suite was 233 passed (test_sym_maps + test_core + test_preprocess only; the
earlier session snapshot had fewer collection targets). Growth
(233 → 857) is primarily from the `gap_value` parametrization layer
tripling `test_core` + `test_preprocess` and from the new entrypoint
parametrizations.

## Naming choice

Initially called the SymMap kwarg and CLI flag `gap_position`, then
renamed to `gap_value` after the user pointed out that "position" is
ambiguous with MSA column position. The rename touches the SymMap
signature, the CLI flag, the `_symmap_from_sym2int` helper, all test
parametrize IDs and argstring fixtures, and the renamed test file
`test_gap_value.py` (was `test_gap_position.py`).

## Lingering items

- **`run_full_pipeline` CLI doesn't expose `--gap_value`.** It builds its
  SymMap via `load_msa(mapping=None)`, which picks up the new library
  default (gap=0). No tests depend on ground-truth integer MSAs from that
  entrypoint, so nothing is broken today. Flagged in the conversation as
  a loose end; no change made.
- **Memory file** (`~/.claude/.../memory/project_gap_integer_default.md`)
  documents the default flip and the Excel-ground-truth convention so
  future sessions don't re-derive expected MSA integers under the new
  encoding.
