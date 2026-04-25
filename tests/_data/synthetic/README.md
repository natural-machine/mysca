# Synthetic projection-test fixture

Hand-crafted MSA + query sequences + PDB residue offsets used by
[`tests/test_project_synthetic.py`](../../test_project_synthetic.py)
to verify mapping correctness across every combination of raw
input length × preprocessing column drops × aligner.

Designed in
[`docs/.claude_sessions/plan_synthetic_projection_fixture.md`](../../../docs/.claude_sessions/plan_synthetic_projection_fixture.md);
the predictions in that plan matched what the pipeline produces
**exactly** for both mafft_add and hmmalign.

## Files

- [`synthetic_msa.fasta`](synthetic_msa.fasta) — 10 training
  sequences in aligned FASTA (gaps at cols 2, 5, 8 for clade A rows).
- [`queries.fasta`](queries.fasta) — 5 query sequences, one per
  length regime.
- [`expected.json`](expected.json) — machine-readable index tables;
  ground truth for test assertions.
- `README.md` — this file; human-readable copy.

## Training MSA shape

`L_orig = 10`, `L_proc = 7`. Columns **2**, **5**, **8** are heavily
gapped (50% gaps with default thresholds; dropped when preprocessing
is run with `--gap_truncation_thresh 0.5 --sequence_gap_thresh 0.5
--position_gap_thresh 0.6`).

`retained_positions = [0, 1, 3, 4, 6, 7, 9]`.

```
                        col:  0 1 2 3 4 5 6 7 8 9
synth_clade_A_0               A C - D E - F G - I
synth_clade_A_1               A C - D E - F G - I
synth_clade_A_2               A C - D K - F G - I
synth_clade_A_3               A C - D E - F G - I
synth_clade_A_4               A C - D K - F G - I
synth_clade_B_0               A C T D E V F G W I
synth_clade_B_1               A C T D K V F G W M
synth_clade_B_2               A C T D E V F G W I
synth_clade_B_3               A C T D K V F G W M
synth_clade_B_4               A C T D K V F G W I
```

## Per-training-sequence index tables

### `synth_clade_A_0` — raw `ACDEFGI` (length 7 = L_proc)

| raw idx | residue | orig col | processed col |
|---|---|---|---|
| 0 | A | 0 | 0 |
| 1 | C | 1 | 1 |
| 2 | D | 3 | 2 |
| 3 | E | 4 | 3 |
| 4 | F | 6 | 4 |
| 5 | G | 7 | 5 |
| 6 | I | 9 | 6 |

All other clade A rows have the same structure (differing only in
E↔K at col 4).

### `synth_clade_B_0` — raw `ACTDEVFGWI` (length 10 = L_orig)

| raw idx | residue | orig col | processed col |
|---|---|---|---|
| 0 | A | 0 | 0 |
| 1 | C | 1 | 1 |
| 2 | T | 2 | *(dropped)* |
| 3 | D | 3 | 2 |
| 4 | E | 4 | 3 |
| 5 | V | 5 | *(dropped)* |
| 6 | F | 6 | 4 |
| 7 | G | 7 | 5 |
| 8 | W | 8 | *(dropped)* |
| 9 | I | 9 | 6 |

Raw indices 2, 5, 8 are residues whose original column was dropped
by preprocessing. They are **present in `raw_sequence`** but never
appear in `residue_by_processed_col` or any `ic_residues[i]`.

## Query index tables

### `query_shorter_than_proc` — input `ACDE` (4 residues)

Aligned: `AC-DE-----`. `raw_sequence = ACDE`.

| proc col | orig col | aligned char | raw idx |
|---|---|---|---|
| 0 | 0 | A | 0 |
| 1 | 1 | C | 1 |
| 2 | 3 | D | 2 |
| 3 | 4 | E | 3 |
| 4 | 6 | - | -1 |
| 5 | 7 | - | -1 |
| 6 | 9 | - | -1 |

### `query_equal_to_proc` — input `ACDEFGI` (7 residues)

Aligned: `AC-DE-FG-I`. `raw_sequence = ACDEFGI`. Same shape as clade A rows.

| proc col | orig col | aligned char | raw idx |
|---|---|---|---|
| 0 | 0 | A | 0 |
| 1 | 1 | C | 1 |
| 2 | 3 | D | 2 |
| 3 | 4 | E | 3 |
| 4 | 6 | F | 4 |
| 5 | 7 | G | 5 |
| 6 | 9 | I | 6 |

### `query_between_proc_and_orig` — input `ACTDEFGI` (8 residues)

Aligned: `ACTDE-FG-I`. `raw_sequence = ACTDEFGI`.

T at raw idx 2 lands at original col 2 (a dropped column). It is
retained in `raw_sequence` but has no processed column.

| proc col | orig col | aligned char | raw idx |
|---|---|---|---|
| 0 | 0 | A | 0 |
| 1 | 1 | C | 1 |
| 2 | 3 | D | 3 |
| 3 | 4 | E | 4 |
| 4 | 6 | F | 5 |
| 5 | 7 | G | 6 |
| 6 | 9 | I | 7 |

### `query_equal_to_orig` — input `ACTDEVFGWI` (10 residues)

Aligned: `ACTDEVFGWI` (no gaps). `raw_sequence = ACTDEVFGWI`.
Same mapping as `synth_clade_B_0`.

### `query_longer_than_orig` — input `ACTDEVNSFGWI` (12 residues)

Aligned: `ACTDEVFGWI`. **Before the fix at commit `693415b`:**
`raw_sequence` was taken from the input (length 12), violating
`raw_sequence == aligned_sequence.replace("-", "")`. After the
fix, `raw_sequence = ACTDEVFGWI` (length 10); the N and S that
the aligner dropped (neither appears anywhere in the training MSA,
so they score poorly at every match state) are not in
`raw_sequence`.

| proc col | orig col | aligned char | raw idx |
|---|---|---|---|
| 0 | 0 | A | 0 |
| 1 | 1 | C | 1 |
| 2 | 3 | D | 3 |
| 3 | 4 | E | 4 |
| 4 | 6 | F | 6 |
| 5 | 7 | G | 7 |
| 6 | 9 | I | 9 |

## PDB residue offsets

`_write_minimal_pdb` produces a single-chain PDB whose sequence
matches the input. The offsets below give each fixture a
distinctive residue-number range so `pdb.residue_ids` is clearly
distinct from `raw_idx` and so `project_groups_to_pdb` exercises a
non-trivial offset.

| fixture | start | length | residue numbers |
|---|---|---|---|
| synth_clade_A_0 | 10 | 7 | 10..16 |
| synth_clade_A_2 | 20 | 7 | 20..26 |
| synth_clade_B_0 | 50 | 10 | 50..59 |
| synth_clade_B_1 | 60 | 10 | 60..69 |
| query_shorter_than_proc | 201 | 4 | 201..204 |
| query_equal_to_proc | 300 | 7 | 300..306 |
| query_between_proc_and_orig | 401 | 8 | 401..408 |
| query_equal_to_orig | 501 | 10 | 501..510 |
| query_longer_than_orig | 601 | 12 | 601..612 |

For `query_longer_than_orig` the PDB's 12 residues will not match
the post-alignment `raw_sequence` of length 10;
`project_groups_to_pdb` deliberately raises `ValueError` on this
mismatch. The synthetic test exercises this error path.

## Aligner agreement

For every query, `mafft_add` and `hmmalign` produced **identical**
aligned rows and identical `residue_by_processed_col` values. The
expected values in `expected.json` apply to both aligners.

## Invariant

Across every training and query case under both aligners:

```
raw_sequence == aligned_sequence.replace("-", "")
```

This is what makes `ic_residues[i]` safely dereferenceable into
`raw_sequence[r]` downstream.
