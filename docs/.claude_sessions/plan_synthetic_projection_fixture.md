# Plan — synthetic projection-test fixture

Planning doc drafted 2026-04-23, to be executed in a follow-up session.
Not yet built; this doc specifies the synthetic training MSA, the
query sequences, the PDB fixtures, and the full set of expected index
mappings we'll assert against.

## Motivation

The existing `tests/test_project.py` invariant tests
([test_project.py:278-460](../../tests/test_project.py#L278)) use the
`msa07` fixture, which has a synthetic 4-letter alphabet (ACDE) and
no columns dropped by preprocessing. That is sufficient to catch the
`raw_sequence` / `aligned_sequence` inconsistency I just fixed, but
doesn't exercise:

1. **Preprocessing column drops** — so
   `retained_positions != range(L_orig)` — any bug where
   projection code confuses original-MSA coordinates with processed-
   MSA coordinates would escape.
2. **Query sequences whose length is between L_proc and L_orig** —
   the most confusing regime, because the aligner must decide
   whether to land a residue at a dropped-column orig position
   (i.e. an insert column that will be stripped anyway) or at a
   retained orig position.
3. **Query sequences strictly longer than L_orig** — exercises the
   invariant fix path end-to-end with a known-correct expected
   output.
4. **PDB residue numbering ≠ raw residue index** — the
   tests/_data/structs fixtures are either malformed
   ([msa07_sequence*.pdb](../../tests/_data/structs/structs07)) or
   disconnected from the SCA MSA
   ([Soil3.scaffold…](../../tests/_data/structs/Soil3.scaffold_414071996_c1_8.pdb)).
   No test currently composes a full projection pipeline with a
   non-trivial PDB residue offset end-to-end.

This plan designs a hand-crafted MSA, a family of query sequences of
controlled lengths, and PDB fixtures with deliberate residue-number
offsets, with explicit expected mappings so every transform has a
traceable ground truth.

## Synthetic training MSA

### Design

**L_orig = 10, L_proc = 7** (columns 2, 5, 8 dropped).

Ten sequences, two "clades" of five. Clade A rows have gaps at
columns 2, 5, 8. Clade B rows fill those columns with different
residues.

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

### Per-column gap frequency

| col | gaps (of 10) | kept at `gap_truncation_thresh=0.5`? |
|---|---|---|
| 0 | 0 | yes (0% < 50%) |
| 1 | 0 | yes |
| 2 | 5 | **no** (50% fails strict `<`) |
| 3 | 0 | yes |
| 4 | 0 | yes |
| 5 | 5 | **no** |
| 6 | 0 | yes |
| 7 | 0 | yes |
| 8 | 5 | **no** |
| 9 | 0 | yes |

**`retained_positions = [0, 1, 3, 4, 6, 7, 9]`.** L_proc = 7.

### Required threshold overrides

Defaults will misbehave on this fixture:

- `--gap_truncation_thresh 0.5` — default 0.4 would also drop
  borderline columns we want retained.
- `--sequence_gap_thresh 0.5` — clade_A sequences have gap freq
  3/10 = 30%; default 0.2 would drop them all. 0.5 keeps them.
- `--position_gap_thresh 0.6` — weighted gap threshold after the
  weighting step. With two ~5-seq clusters at `δ=0.8`, weighted
  freq at dropped cols lands around 50%; default 0.2 would also
  drop some retained cols. 0.6 keeps the seven we want.
- No `--reference` — clade A and clade B diverge enough at cols
  2/5/8 that a reference-similarity filter is a nuisance for a
  test this small.

### Ungapped (raw) primary sequences

| MSA id | raw sequence | length | equals L_proc? equals L_orig? |
|---|---|---|---|
| synth_clade_A_0 | `ACDEFGI` | 7 | **L_raw == L_proc** |
| synth_clade_A_1 | `ACDEFGI` | 7 | same |
| synth_clade_A_2 | `ACDKFGI` | 7 | same |
| synth_clade_A_3 | `ACDEFGI` | 7 | same |
| synth_clade_A_4 | `ACDKFGI` | 7 | same |
| synth_clade_B_0 | `ACTDEVFGWI` | 10 | **L_raw == L_orig** |
| synth_clade_B_1 | `ACTDKVFGWM` | 10 | same |
| synth_clade_B_2 | `ACTDEVFGWI` | 10 | same |
| synth_clade_B_3 | `ACTDKVFGWM` | 10 | same |
| synth_clade_B_4 | `ACTDKVFGWI` | 10 | same |

Each raw sequence represents either "only the retained columns" (clade A)
or "all columns including the dropped ones" (clade B). Both cases are
needed so the test sees raw-index → processed-col mappings for
sequences that natively have exactly the processed columns AND for
sequences that natively have the full original-column count.

## Per-training-sequence index tables

### synth_clade_A_0 = `ACDEFGI` (clade A, raw length 7)

| raw idx | residue | aligned col (L_orig) | processed col (L_proc) |
|---|---|---|---|
| 0 | A | 0 | 0 |
| 1 | C | 1 | 1 |
| 2 | D | 3 | 2 |
| 3 | E | 4 | 3 |
| 4 | F | 6 | 4 |
| 5 | G | 7 | 5 |
| 6 | I | 9 | 6 |

No raw residue lands in any dropped column (clade A's only
non-gap columns are the retained ones). So for a clade A row,
`raw_idx ↔ processed_col` is the identity map.

### synth_clade_B_0 = `ACTDEVFGWI` (clade B, raw length 10)

| raw idx | residue | aligned col (L_orig) | processed col (L_proc) |
|---|---|---|---|
| 0 | A | 0 | 0 |
| 1 | C | 1 | 1 |
| 2 | T | 2 | **— (dropped)** |
| 3 | D | 3 | 2 |
| 4 | E | 4 | 3 |
| 5 | V | 5 | **— (dropped)** |
| 6 | F | 6 | 4 |
| 7 | G | 7 | 5 |
| 8 | W | 8 | **— (dropped)** |
| 9 | I | 9 | 6 |

Three raw residues (indices 2, 5, 8) have no processed-column home.
The projection should silently skip them; they won't appear in any
`ic_memberships[i]` because processed cols 2, 5, 8 don't exist in
`retained_positions`.

## Synthetic query sequences

Five queries, one per length regime. Each is derived from a clade B
sequence to make the aligner's job predictable (for `mafft_add` and
`hmmalign` both).

### `query_shorter_than_proc` — raw length 4

Raw: `ACDE`. Shorter than `L_proc=7`.

Expected aligned row: `AC-DE-----` (10 cols; residues at cols 0,1,3,4; gaps elsewhere).

| proc col | orig col | aligned char | raw idx |
|---|---|---|---|
| 0 | 0 | A | 0 |
| 1 | 1 | C | 1 |
| 2 | 3 | D | 2 |
| 3 | 4 | E | 3 |
| 4 | 6 | - | -1 |
| 5 | 7 | - | -1 |
| 6 | 9 | - | -1 |

- `raw_sequence` (post-fix) = `ACDE` (length 4 = `aligned.replace("-", "")`).
- `residue_by_processed_col` = `[0, 1, 2, 3, -1, -1, -1]`.

### `query_equal_to_proc` — raw length 7

Raw: `ACDEFGI`. Same as any `synth_clade_A_*_0`. Equal to `L_proc`.

Expected aligned row: `AC-DE-FG-I` (the clade-A shape, gaps at the three dropped cols).

| proc col | orig col | aligned char | raw idx |
|---|---|---|---|
| 0 | 0 | A | 0 |
| 1 | 1 | C | 1 |
| 2 | 3 | D | 2 |
| 3 | 4 | E | 3 |
| 4 | 6 | F | 4 |
| 5 | 7 | G | 5 |
| 6 | 9 | I | 6 |

- `raw_sequence` = `ACDEFGI` (length 7).
- `residue_by_processed_col` = `[0, 1, 2, 3, 4, 5, 6]`.

### `query_between_proc_and_orig` — raw length 8

Raw: `ACTDEFGI`. 8 residues = L_proc + 1; can either land all 8 in
match columns (with T at orig col 2 — the aligner's choice) or drop
the T as an insert.

**Expected aligned row for mafft --add --keeplength**: `ACTDE-FG-I`
(T matched at col 2, which is a dropped col; aligned has 8 non-gap
chars). Verified by running the aligner; if it differs, the test
should use the actual output and only assert the invariant.

| proc col | orig col | aligned char | raw idx |
|---|---|---|---|
| 0 | 0 | A | 0 |
| 1 | 1 | C | 1 |
| 2 | 3 | D | 3 |
| 3 | 4 | E | 4 |
| 4 | 6 | F | 5 |
| 5 | 7 | G | 6 |
| 6 | 9 | I | 7 |

- `raw_sequence` = `ACTDEFGI` (length 8 — T is retained in the
  aligned output even though its column gets dropped during
  processed-col indexing).
- `residue_by_processed_col` = `[0, 1, 3, 4, 5, 6, 7]`.
- Note: raw idx 2 ('T') is in raw_sequence but never appears in
  `residue_by_processed_col` (its orig col was dropped).

### `query_equal_to_orig` — raw length 10

Raw: `ACTDEVFGWI`. Identical to `synth_clade_B_0`.

Expected aligned row: `ACTDEVFGWI` (no gaps, no dropped residues).

| proc col | orig col | aligned char | raw idx |
|---|---|---|---|
| 0 | 0 | A | 0 |
| 1 | 1 | C | 1 |
| 2 | 3 | D | 3 |
| 3 | 4 | E | 4 |
| 4 | 6 | F | 6 |
| 5 | 7 | G | 7 |
| 6 | 9 | I | 9 |

- `raw_sequence` = `ACTDEVFGWI` (length 10).
- `residue_by_processed_col` = `[0, 1, 3, 4, 6, 7, 9]`.
- Raw indices 2 ('T'), 5 ('V'), 8 ('W') are in raw_sequence but
  drop out of `residue_by_processed_col`.

### `query_longer_than_orig` — raw length 12

Raw: `ACTDEVXYFGWI`. L_orig + 2 residues; the aligner must drop 2.

Expected aligned row: `ACTDEVFGWI` (the X and Y land as insert-state
columns, stripped before `residue_by_processed_col` is derived).

- **Under the `raw = input` bug**: `raw_sequence` would be
  `ACTDEVXYFGWI` (length 12), and `raw_sequence[5]` = 'V' but
  `residue_by_processed_col[3]` → raw idx 4 → 'E'. The invariant
  `raw_sequence == aligned_sequence.replace("-", "")` fails
  (length 12 ≠ 10).
- **Under the fix**: `raw_sequence` = `ACTDEVFGWI` (length 10);
  X and Y are gone. `residue_by_processed_col[3]` = 4 = 'E' in
  `raw_sequence`. Invariant holds.

| proc col | orig col | aligned char | raw idx |
|---|---|---|---|
| 0 | 0 | A | 0 |
| 1 | 1 | C | 1 |
| 2 | 3 | D | 3 |
| 3 | 4 | E | 4 |
| 4 | 6 | F | 6 |
| 5 | 7 | G | 7 |
| 6 | 9 | I | 9 |

`residue_by_processed_col` = `[0, 1, 3, 4, 6, 7, 9]` (same as the
equal-to-orig case, by design).

## Synthetic PDB fixtures

One PDB per unique raw sequence, written with `_write_minimal_pdb`
(already defined in `tests/test_structure.py`). Residue numbering
is **deliberately non-trivial** so the `raw_idx ↔ PDB resi` map is
distinct from the `raw_idx ↔ processed_col` map.

| seq id | raw | residue numbers | notes |
|---|---|---|---|
| synth_clade_A_0 | ACDEFGI | 10..16 | 7 residues, offset +10 |
| synth_clade_B_0 | ACTDEVFGWI | 50..59 | 10 residues, offset +50 |
| query_shorter_than_proc | ACDE | 201..204 | 4 residues, offset +201 |
| query_equal_to_proc | ACDEFGI | 300..306 | 7 residues, offset +300 |
| query_between_proc_and_orig | ACTDEFGI | 401..408 | 8 residues, offset +401 |
| query_equal_to_orig | ACTDEVFGWI | 501..510 | 10 residues, offset +501 |
| query_longer_than_orig | ACTDEVXYFGWI | 601..612 | 12 residues, offset +601 |

For each PDB, `project_groups_to_pdb(sequence_projection, pdb)` should
pull raw indices from `sequence_projection.ic_memberships` and map
them through `pdb.residue_ids`. So for synth_clade_B_0's PDB
(residues 50..59), raw idx 0 → PDB residue 50; raw idx 3 → PDB
residue 53; raw idx 9 → PDB residue 59.

## Test-scope outline (not yet implemented)

Once the fixture lands, the following tests can be written:

1. **Preprocessing shape**: run sca-preprocess on the synthetic FASTA
   and assert `retained_positions == [0, 1, 3, 4, 6, 7, 9]`.
2. **In-sample projection** for each clade A and clade B reference:
   assert `residue_by_processed_col` matches the tables above.
3. **Out-of-sample projection** for each of the five queries, under
   both `mafft_add` and `hmmalign`. Assertions:
   - The invariant `raw_sequence == aligned_sequence.replace("-", "")`.
   - `len(raw_sequence) == min(len(input), L_orig)` (the column-
     preservation bound).
   - `residue_by_processed_col` matches the table above.
4. **Structure projection** for each of the synthetic PDBs: run
   `project_pdb` with the matching preprocessing/scacore dirs, assert
   `ic_pdb_residues` resolves to the PDB numbering (not to raw
   residue indices).

## Files to create (in the follow-up session)

- `tests/_data/synthetic/synthetic_msa.fasta` — the 10 training
  sequences with gaps (aligned FASTA).
- `tests/_data/synthetic/queries.fasta` — the 5 query sequences,
  ungapped.
- `tests/_data/synthetic/expected.json` — the index tables above as a
  machine-readable JSON blob, keyed by sequence id and aligner name.
- `tests/_data/synthetic/README.md` — human-readable copy of the
  index tables + residue offsets. Links back to this plan.
- `tests/test_project_synthetic.py` — new test module consuming the
  fixture and asserting every mapping above.

The PDBs themselves are generated on the fly from the sequences +
`expected.json`'s residue offsets via `_write_minimal_pdb`; they do
not need to be shipped.

## Open questions before implementation

1. **Reference clade for the PDB residue tests**: clade A (raw length
   == L_proc, simpler) or clade B (raw length == L_orig, exercises
   the "residue in dropped col" case). My vote: **both** — one PDB per
   unique sequence is cheap.
2. **Aligner-specific expectations**: mafft and hmmalign may pick
   different column layouts for the "between-proc-and-orig"
   case. If they disagree on the predicted aligned row, split the
   `expected.json` into per-aligner blocks. I'd prefer to verify
   empirically once implemented and only assert the invariant if the
   column layout diverges.
3. **Stability of SCA on such a tiny MSA**: kstar will likely fall
   back to 1; some IC groups may be empty. That's fine for the
   invariant and mapping tests, but the test's own assertions must
   not depend on non-trivial IC content — only on `retained_positions`,
   `residue_by_processed_col`, and `raw_sequence`.

## Follow-up after the fixture lands

- Parametrize the mafft/hmmalign invariant tests over the five
  queries instead of the current single
  donor-with-insertion case.
- Parametrize the `sca-structure` in-sample roundtrip over the two
  unique training-sequence PDBs.
- If a future test ever needs a fixture with branches-per-column
  more complex than this, extend clade_A / clade_B rather than
  building yet another MSA.
