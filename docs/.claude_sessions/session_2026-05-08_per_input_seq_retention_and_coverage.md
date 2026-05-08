# 2026-05-08 — Per-input-sequence retention statistics

## Summary

Added two per-input-MSA-sequence diagnostics so users can ask, for any
sequence in the original input FASTA:

1. **`seq_retained_fraction`** (always computed, in `sca-preprocess`)
   — what fraction of this sequence's non-gap residues survived the
   column-filtering steps. Sequence-side normalization:
   `kept_AAs / total_non_gap_AAs_in_input`. Indexed by post-`load_msa`
   input order, so sequences later dropped by sequence-level filters
   still have a recorded value (NaN when the input row has zero
   non-gap residues).

2. **`component_coverage_per_seq`** (gated by new `--coverage_for`
   flag, in `sca-core`) — for each selected input sequence, the
   fraction of every IC's high-load positions where the sequence has
   a non-gap residue. Stored as a dict keyed by `seq_id` →
   length-`n_components` float vector. NaN entries flag ICs with
   empty position sets.

The two fractions are complementary: (1) summarizes how much of a
sequence's signal made it past column filtering; (2) tells you how
well that sequence covers the positions that *define* each sector.

## Why this design

Plan: [.claude/plans/i-want-to-enable-tidy-lighthouse.md](../../.claude/plans/i-want-to-enable-tidy-lighthouse.md).

Three substantive design decisions, all settled with the user via
AskUserQuestion before drafting the plan:

- **Where each fraction lives.** Fraction (1) is intrinsic to
  preprocessing — it depends only on `retained_positions` and the
  post-`load_msa` MSA — so it's computed unconditionally inside
  `preprocess_msa` rather than gated by a flag. Fraction (2) needs
  `ic_positions` and lives in `sca-core` next to the existing
  `--sectors_for` block.

- **`--coverage_for` semantics deliberately diverge from
  `--sectors_for`.** `--sectors_for all` means "every retained
  sequence" because per-sequence sector residue mappings are
  undefined for filtered sequences. But coverage is meaningful for
  filtered sequences too — in fact a low coverage fraction on a
  dropped sequence helps explain *why* it was dropped. So
  `--coverage_for all` (the default) resolves against the full input
  MSA, including pre-filtered sequences. The same applies to the
  file-path mode and the explicit `reference` literal.

- **Storage format: dict keyed by sequence id.** Mirrors
  `ic_residues_per_seq` rather than a 2D matrix + parallel id list.
  Per-sequence storage cost is small (one float per IC) so the dict
  overhead is acceptable; per-seq lookup is the natural access
  pattern for downstream analysis. Compute is done over **all**
  `n_components` (not just `kstar`) because the storage cost stays
  small and it's more useful for diagnosing borderline ICs.

## Changes

Eight files touched.

**Library:**
- [src/mysca/preprocess.py](../../src/mysca/preprocess.py): vectorized
  computation of `seq_retained_fraction` over the post-`load_msa`
  integer MSA, inserted just before the `preprocessing_results` dict
  is assembled. Added to the dict and the docstring.
- [src/mysca/results.py](../../src/mysca/results.py):
  - `PreprocessingResults`: new `seq_retained_fraction` field
    (init / `from_preprocess_output` / save / load /
    `FIELD_DESCRIPTIONS` / on-disk-format docstring). Loads as `None`
    for legacy bundles missing the key.
  - `SCAResults`: new `component_coverage_per_seq` dict field
    (same five surfaces). New on-disk file constant
    `COMPONENT_COVERAGE_PER_SEQ_FNAME = "component_coverage_per_seq.npz"`,
    persisted via `np.savez_compressed(**dict)` to match
    `ic_residues_per_seq.npz`.
- [src/mysca/run_sca.py](../../src/mysca/run_sca.py):
  - New `--coverage_for` argparse entry (default `"all"`, accepting
    `"reference"`, `"all"`, or a path to a newline-delimited id
    file).
  - Resolver block placed parallel to the existing `--sectors_for`
    block but resolving against `msa_obj_loaded` (the input MSA),
    not `retained_sequences`. Missing-id warning is logged like the
    sectors_for path.
  - Computation builds a single `(n_requested, L_orig)` non-gap byte
    mask once (numpy `frombuffer` per row), then per-IC slices via
    `retained_positions[ic_positions[i]]` and divides by
    `len(ic_positions[i])`. NaN when the IC's position set is empty.
  - `coverage_for` and `sectors_for` added to the persisted args
    dict. Top docstring (`COMMAND LINE ARGUMENTS` + `OUTPUTS`)
    updated per the CLAUDE.md three-surface rule.
- [src/mysca/run_preprocessing.py](../../src/mysca/run_preprocessing.py):
  top-docstring `OUTPUTS` section documents the new
  `seq_retained_fraction` array.

**Docs:** `docs/cli_reference.md` updates (the `sca-preprocess` output
bullet for `seq_retained_fraction` and the `sca-core` `--coverage_for`
row + `component_coverage_per_seq.npz` output bullet) shipped in the
prior commit's `docs/cli_reference.md` snapshot
(commit 3a2c106 — coincident edits got bundled, not by intent).

**Tests:**
- [tests/test_preprocess.py](../../tests/test_preprocess.py): four new
  tests for fraction (1) — a hand-computable check across 5 input
  sequences with one all-gap row that hits the sequence-gap filter,
  a partial-retention variant, a save/load round-trip, and a
  legacy-bundle compatibility check that loads as `None`.
- [tests/test_results.py](../../tests/test_results.py): round-trip
  test for `component_coverage_per_seq` including a NaN entry to
  guard against silent dtype coercion.
- [tests/test_entrypoint_scarun.py](../../tests/test_entrypoint_scarun.py):
  three end-to-end sca-core tests on the existing `msa06.faa` /
  `argstring6a` fixtures — default `--coverage_for=all` (one key per
  *input* MSA sequence, including the three the preprocessing chain
  drops); `--coverage_for=reference` (single-key result); and
  `--coverage_for=<path>` with a deliberately-missing id silently
  skipped. Uses `n_boot=2` rather than `-1` because `-1` triggers a
  pre-existing `evals_shuff[:,1]` indexing bug when no prior
  bootstrap file exists.

## Verification

`pytest tests/test_preprocess.py tests/test_results.py
tests/test_entrypoint_scarun.py -k "not gpu"` → **669 passed, 135
deselected**. The deselected tests are GPU-weight parametrizations
that were already failing before this change (host has NVIDIA GB10 /
CUDA cap 12.1, above the installed PyTorch's declared 12.0 ceiling)
— confirmed via `git stash && pytest` baseline by the user.

User ran the full suite locally and reported pass.

## Notes for follow-up

- `--coverage_for=reference` requires a `reference_id` recorded in
  the prior `sca-preprocess` args; without one, the resolver logs and
  skips. Consider folding this into a downstream filter-history /
  diagnostics view since `seq_retained_fraction` is now a natural
  signal for the filter-distribution plot.
- The default of `"all"` is cheap (M_input × n_components floats) but
  is technically a behavior change — previous runs would not have
  produced `component_coverage_per_seq.npz`. Existing downstream
  consumers should treat the file as optional (the `SCAResults.load`
  path already does).
