# Session: Add --sectors_for argument to sca-core

**Date:** 2026-03-30

## Motivation

The files `statsectors_msa.npz` and `statsectors_seq.npz` can be extremely
large when the number of retained sequences is large, because they contain
per-sequence sector position mappings for every retained sequence. Most use
cases only need mappings for a small number of sequences (e.g. the reference
or a few structures of interest).

## Changes

### `src/mysca/run_sca.py`

Added `--sectors_for` CLI argument controlling which sequences get per-sequence
sector mappings in `statsectors_msa.npz` and `statsectors_seq.npz`:

- **`None` (default)** — only the reference sequence (read from preprocessing
  args). If no reference was specified, or if it was filtered out during
  preprocessing, produces empty statsectors files with a log message.
- **`"all"`** — every retained sequence (previous default behavior).
- **`<filepath>`** — a text file with one sequence ID per line. Only matching
  retained sequences are included.

Missing IDs are handled gracefully:
- If the reference was filtered out during preprocessing, a message is printed
  and empty statsectors files are saved.
- If a file is provided, IDs not found among retained sequences are logged by
  name, and the remaining matches are processed. If none match, empty files
  are saved.

The per-sequence raw sequence index computation (`group_rawseq_positions`,
`group_rawseq_scores`, and the `_by_entry` helpers) now operates on the
subset `sector_rawseq_idxs` rather than all retained sequences, so only
the necessary rows are computed.

## Files Modified

- `src/mysca/run_sca.py` — new argument, subset logic, graceful missing-ID
  handling

## Verification

All 351 tests pass.
