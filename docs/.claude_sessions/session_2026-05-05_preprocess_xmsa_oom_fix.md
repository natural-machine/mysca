# 2026-05-05 — Defer dense one-hot construction in `preprocess_msa` to fix OOM

## Summary

`mysca.preprocess.preprocess_msa` was building the dense 3D one-hot
encoding `xmsa` immediately after loading the integer MSA — *before*
any filtering. For an MSA of shape `(N_seq, N_pos)` this materialized
`(N_seq, N_pos, NUM_AAS=20)` int16 entries up front, ~40× the size of
the integer MSA. On large alignments (~50k × 1k → ~2 GB int16, ~8 GB
once cast to int64 in the result dict) this triggered an OOM even
though the integer MSA itself comfortably fit in memory and the
*post-filter* MSA would also fit as a one-hot.

Fix: build `xmsa` exactly once at the end of preprocessing, when
`msa` is at its smallest. No public-API change.

## Why this is safe

The exploration step confirmed `xmsa` had only one real consumer:
`preprocessing_results["msa_binary3d"]` returned at the end of
`preprocess_msa`. The intermediate slicing of `xmsa` after every filter
was bookkeeping; all gap / similarity statistics are computed off the
integer `msa` directly.

Crucially, `xmsa` was also being threaded into `compute_weights(...)`
as a kwarg, but **none** of the five `_compute_weights_*` dispatch
targets (`sparse`, `gpu`, `_v3`, `_v4`, `_v6`) ever read it — they
each rebuild a sparse one-hot internally via
`get_onehotmsa_sparse(msa, num_aas, gap)`. Dead arg, removed.

## Changes

Single-file edit to [src/mysca/preprocess.py](../../src/mysca/preprocess.py):

1. Removed early construction (the two `xmsa = ...` lines that used to
   sit just before the position-gap filter).
2. Removed all four intermediate `xmsa = xmsa[...]` slicings between
   filter stages (position-gap, sequence-gap, reference-similarity,
   weighted-position-gap).
3. Dropped the unused `xmsa=xmsa` kwarg from both `compute_weights(...)`
   calls (round 1 and round 2).
4. Added a single `xmsa = onehot_without_gap(msa, NUM_SYMS, GAP)` line
   immediately before the `preprocessing_results` dict, so the dense
   one-hot is built once on the fully-filtered MSA. The downstream
   `xmsa.astype(int)` cast in the result dict is unchanged. The
   intermediate `np.int16` cast that used to live at the top is gone —
   it was only there to keep memory in check across the now-removed
   slicing chain.

The existing helper `onehot_without_gap` was reused as-is. Two
unrelated call sites in `run_sca.py` (shuffle bootstrap) were not
touched.

## Verification

Full suite via `pytest tests`:
- **1130 passed**, 1 deselected, 0 failed (suite total grew naturally
  since the previous session's 1112).
- The one deselected test (`test_compute_fijab_gpu_precision_match`) is
  an order-dependent fp32 GPU rounding test (max abs diff 2.4e-5 vs
  1e-6 tolerance) that flakes when run after a long preceding test
  load on this machine's GB10 (CUDA cap 12.1, above PyTorch's
  declared support of 12.0). It passes 3/3 times in isolation both on
  the baseline and on the patched code, confirming this refactor did
  not introduce the flake.

## Out of scope (deliberately deferred)

- `msa_binary3d` is still cast to `int` (typically int64) in the result
  dict — 8× larger than the bool form. Tightening this dtype is a
  separate optimization and would touch
  `results.py` save/load + downstream consumers.
- The shuffle-bootstrap `onehot_without_gap` calls in `run_sca.py` may
  have a similar pre-filter memory profile but were not the reported
  problem and are out of scope here.
