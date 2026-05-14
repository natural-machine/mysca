# 2026-05-14 — Restrict `get_rawseq_indices_of_msa` to needed rows (sca-core OOM fix)

## Summary

A colleague's `sca-core` run on a large preprocessed MSA was being
killed by the OOM-killer right after the log line
`Generating per-sequence sectors for reference sequence: ...`. Her
command was the default `sca-core -i ... -o ... --regularization 0.03
--seed 0 -nb 2` — no `--sectors_for`, so per-sequence sector mapping
is supposed to cover only the reference row.

Root cause: two call sites in `run_sca.py` were calling
`get_rawseq_indices_of_msa(msa_obj_loaded)` on the *full* loaded MSA,
which allocates an `int64` matrix of shape
`(n_total_seqs, n_total_positions)` — tens of GB on full-scale inputs —
and then immediately threw all but one row away. The first call
(inside `log_top_ic_summary`) printed the `reference res:` line and
returned; the second call (the sector-mapping block) tipped the
process over.

Fix: teach the helper to materialize only the rows the caller asks
for, and have both call sites request exactly the rows they consume.

## Why this is safe

- `log_top_ic_summary` already had `ref_row` and only ever indexed
  `all_raw[ref_row]`; nothing else used `all_raw`.
- The sector-mapping block immediately did
  `rawseq_idxs[retained_sequences,:][:,retained_positions][np.isin(retained_sequences, sector_seqidxs)]`.
  `sector_seqidxs` is always a subset of `retained_sequences` (it is
  constructed by iterating `retained_sequences`), so materializing
  only those rows up front gives an equivalent result with identical
  ordering — and the order-preservation is exercised by the new
  parametrized test (single row, out-of-order subset, empty subset).
- The helper's default `seqidxs=None` preserves the original
  full-matrix behavior, so `get_conserved_rawseq_positions` and the
  tests that call it positionally are unaffected.

## Changes

1. [src/mysca/helpers.py](../../src/mysca/helpers.py) —
   `get_rawseq_indices_of_msa` gains an optional
   `seqidxs: NDArray | None` kwarg. When given, the function builds
   only `(len(seqidxs), npos)` rows, in `seqidxs` order. Docstring
   spells out the OOM motivation.
2. [src/mysca/run_sca.py](../../src/mysca/run_sca.py) —
   - `log_top_ic_summary`: `get_rawseq_indices_of_msa(msa_obj_loaded,
     seqidxs=np.array([ref_row]))[0]` instead of computing the full
     matrix and indexing one row.
   - Sector-mapping block (was lines 998–1005): now
     `get_rawseq_indices_of_msa(msa_obj_loaded, seqidxs=sector_seqidxs)[:, retained_positions]`,
     replacing the
     `full → retained_sequences → retained_positions → np.isin`
     chain. A short comment explains the memory rationale so the
     waste doesn't get reintroduced.
3. [tests/test_helpers.py](../../tests/test_helpers.py) — new
   parametrized `test_get_rawseq_indices_of_msa_seqidxs_subset`
   covering single row, out-of-order subset, and empty subset.

## Verification

- `pytest tests/test_helpers.py` — 18 passed.
- Focused subset matching `-k "sca or run_sca or core or rawseq or
  log_top"` — 189 passed across `test_core.py`,
  `test_entrypoint_scarun.py`, `test_helpers.py`,
  `test_ic_summary_log.py`, `test_project.py` (~4 min).
- User confirmed full-suite pass on their machine.

## Out of scope (deliberately deferred)

- The same helper is still typed as returning `int64`. Position
  indices comfortably fit in `int32`, which would halve memory for
  any future full-matrix users — not pursued here since the
  call-site change already eliminates the offending allocations.
- `get_conserved_rawseq_positions` in `helpers.py` still calls the
  helper without `seqidxs`. It is only used in tests today, so this
  was left alone.

## Pre-session state

```bash
git checkout c1932bd  # "Bump version to 0.1.3"
```

## Commit

See the commit that follows this session note.
