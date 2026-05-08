# 2026-05-07 — sca-project input flexibility: `--raw` and `--align_target`

## Summary

Two related sca-project ergonomics improvements:

1. **`--raw`**: accept a literal amino-acid sequence string at `-i` instead
   of a FASTA path. The string is uppercased and whitespace-stripped, then
   materialized inside the run outdir as `raw_input.fasta` and fed through
   the standard projection pipeline. Avoids hand-writing a one-record
   FASTA when projecting a single sequence interactively. The record's ID
   defaults to `raw_input` (override with `--seq_id`).

2. **`--align_target {original, processed}`**: choose which reference MSA
   the aligner uses for out-of-sample queries. Default `original` is
   today's behavior (align against `msa_orig.fasta-aln`, length `L_orig`).
   `processed` aligns against the post-preprocessing MSA (length
   `L_proc`, sliced from `msa_obj_loaded` by `retained_sequences` ×
   `retained_positions`). Processed mode yields a denser, higher-quality
   reference but more aggressively clips input residues that don't fall
   into a reference column. Three new schema fields surface the
   trade-off: `align_target`, `n_input_residues_dropped`,
   `input_coverage_fraction`. A per-record WARNING fires when coverage
   < 0.95, plus a roll-up summary at end of run.

Both features are gated behind opt-in flags. Default behavior is
unchanged for existing callers.

## Why

`--raw`: trivial DX win — projecting a single hand-typed sequence used
to require writing a FASTA to disk first, which made it inconvenient at
the CLI / interactive REPL.

`--align_target`: in some workflows, the unfiltered MSA's gappy
low-quality columns drag down out-of-sample alignment quality. The
processed MSA is the one the SCA model is actually defined on, and
aligning directly against it can produce cleaner residue-to-IC
mappings — especially when the user's query is closer in length /
domain coverage to the post-truncation MSA than to the original.

The user flagged a specific concern during planning: a 550-aa input vs
a 500-column processed MSA — what gets clipped? The answer turns out
to be: input residues that don't fall in any reference column are
silently dropped by both `mafft --add --keeplength` and `hmmalign +
match-only stripping`. This already happens under `--align_target
original` for any input longer than `L_orig` (rare); processed mode
makes it routine. The two new coverage fields give callers a
programmatic signal so they can detect and respond.

## Plan

Approved plan lives at
[.claude/plans/precious-skipping-hollerith.md](../../.claude/plans/precious-skipping-hollerith.md).

## Changes

### Source

- **[src/mysca/run_project.py](../../src/mysca/run_project.py)**
  - `--raw` argparse flag and post-parse mutual-exclusion with
    `--from_msa`. `_materialize_raw_input` helper writes
    `raw_input.fasta` to outdir.
  - `--align_target {original, processed}` argparse flag, threaded
    into `project_sequences`. Module docstring `COMMAND LINE
    ARGUMENTS:` and `OUTPUTS:` sections updated.

- **[src/mysca/project/projection.py](../../src/mysca/project/projection.py)**
  - Three new `SequenceProjection` fields (`align_target`,
    `n_input_residues_dropped`, `input_coverage_fraction`), threaded
    through `to_dict()` so `projection.json` carries them.
  - `_build_processed_reference_msa` helper: slices `msa_obj_loaded`
    by `retained_sequences × retained_positions` and (optionally)
    writes the resulting alignment to
    `<workdir>/processed_reference.fasta-aln` for transparency / debug.
  - `project_sequences` accepts `align_target: str = "original"`,
    validates it against `ALIGN_TARGET_CHOICES`, and branches:
    - active reference → `msa_obj_loaded` vs sliced view
    - in-sample row → as-loaded (L_orig) vs sliced (L_proc)
    - post-alignment indexing → `resi_by_orig[retained_positions]` vs
      `resi_by_orig` directly
    - length sanity check against L_orig vs L_proc.
  - Per-record + roll-up coverage WARNINGs at threshold 0.95.
  - Bug fix: `_aligned_to_xmsa` was indexing `aligned[pos]`, which
    only works when the aligned sequence is in original-MSA columns.
    Added an `aligned_in_processed_coords` kwarg so processed-mode
    callers index by `j` (the processed-column index) instead. Caught
    by the synthetic OOS smoke test.

### Docs

- **[docs/cli_reference.md](../../docs/cli_reference.md)**: new
  `--align_target` row in the optional-arguments table; `--raw`
  documented earlier; `projection.json` schema description carries the
  three new fields; `_align_workdir/processed_reference.fasta-aln`
  listed under outputs.

- **[README.md](../../README.md)**: paragraphs documenting `--raw`
  and `--align_target` (with the coverage caveat) under "Project a
  sequence". Outputs bullet list updated to include the three
  conditionally-materialized FASTAs.

### Tests

- **[tests/test_project.py](../../tests/test_project.py)** — 11 new
  `--raw` tests (argparse mutual exclusion, normalization, default
  seq_id, equivalence with `-i fasta`) plus 11 new `--align_target`
  tests covering: argparse default + validation, in-sample slice
  equivalence, length contracts, schema field presence, filtered-but-
  loaded record handling, processed-reference-FASTA materialization,
  in-sample coverage defaults, and the low-coverage WARNING.

- **[tests/test_project_synthetic.py](../../tests/test_project_synthetic.py)**
  — mafft- and hmmer-gated OOS smoke verifying that
  `len(aligned_sequence) == L_proc` under processed mode and the
  raw/aligned invariant still holds.

## Verification

Focused tests:
```bash
conda activate ./env
python -m pytest tests/test_project.py -k "raw or align_target or coverage" -v
python -m pytest tests/test_project_synthetic.py -k "align_target" -v
```

Full project + structure + pymol suite:
```bash
CUDA_VISIBLE_DEVICES= python -m pytest \
    tests/test_project.py tests/test_project_synthetic.py \
    tests/test_structure.py tests/test_entrypoint_pymol.py -q
```
→ 166 passed.

Full non-CUDA suite:
```bash
CUDA_VISIBLE_DEVICES= python -m pytest tests/ -q --ignore=tests/test_core.py
```
→ 1138 passed.

The end-to-end SH3 demo smoke from the plan was skipped per user
request; all unit and integration coverage is green.

## Decisions made under user clarification

The user picked four specific behaviors during planning:

- Flag name: `--align_target {original,processed}` (not `--ref_msa`,
  `--align_to`).
- Schema: persist `align_target` per-record in `projection.json` (not
  args-only).
- Filtered-in-sample handling: project from the loaded row sliced by
  `retained_positions` — no aligner invocation, no error.
- Processed-MSA materialization: lazy, inside the run's
  `_align_workdir/`, not a permanent `sca-preprocess` artifact.
- Input-longer-than-reference: warn + surface coverage in
  `projection.json`, no hard rejection.

The `--raw` validation was originally going to reject characters
outside `AA_STD20 ∪ {'-'}`, but the user revised mid-implementation:
no alphabet check now, just whitespace strip + uppercase + reject
empty / all-gap. The corresponding test was flipped from "rejection"
to "acceptance".

## Out of scope (deferred)

- Adding `--align_target` to `sca-structure` (passthrough). Trivial
  follow-up; left out to keep this PR focused.
- Persisting `msa_processed.fasta-aln` as a permanent
  `sca-preprocess` artifact.
- A `--coverage_warn_threshold` flag — today's plan hardcodes 0.95.

## Post-implementation activities (no code changes)

After the feature work, two follow-up checks happened:

1. **Python API section of the README re-verified** against the live
   code by `inspect.signature` / `__init__` parameter introspection.
   Every symbol the section names resolves: `PreprocessingResults` /
   `SCAResults` (`load`, `n_sequences` / `n_positions`,
   `sequence_weights`, `msa_binary3d`, `info`, `n_ic_positions`,
   `conservation`, `sca_matrix`, `project_sequences(xmsa)`,
   `to_dataframe(prep)`, `sequence_metadata`); the `mysca.project` /
   `mysca.structure` exports (`project_sequences`, `PDBStructure`,
   `project_pdb`, `SequencePdbMap`); `SequenceProjection` /
   `ProjectionResult` / `PdbProjection` attributes. Two minor
   freshness gaps were noted (the `result.to_dataframe()` column
   listing is incomplete, and the `SequenceProjection` blurb doesn't
   echo this session's three new fields) but neither is invalid;
   both deferred.

2. **Working-tree audit before committing.** Six files in the
   working tree at end of session are pre-existing in-progress
   changes from the user that predate this session and are orthogonal
   to `--raw` / `--align_target`:
   - `src/mysca/preprocess.py` (+21) — adds `seq_retained_fraction`
     to `preprocess_msa()` return dict.
   - `src/mysca/results.py` (+69) — corresponding `PreprocessingResults`
     field, plus other unrelated fields used by the run_sca tests below.
   - `src/mysca/run_sca.py` (+128) — separate run_sca additions.
   - `tests/test_entrypoint_scarun.py` (+135) — tests for run_sca
     additions.
   - `tests/test_preprocess.py` (+156) — tests for the
     `seq_retained_fraction` change.
   - `tests/test_results.py` (+21) — tests for the results changes.

   These should land in a separate commit to keep topics focused.
   `README.md` and `docs/cli_reference.md` carry interleaved edits
   from both topics; their `seq_retained_fraction` mentions belong
   logically with the pre-session work but committing them with the
   `--raw` / `--align_target` topic is acceptable since they're
   small.

## Pre-session anchor

```bash
git checkout f8b5f5c
```

(`f8b5f5c` = "Packaging cleanup, MPS fp64 fallback, install/doc fixes",
the commit that landed the earlier portion of this same multi-topic
session: pymol-required, MPS fp64 fallback, install hints, soft test
warnings, README polish + bullet conversion, sca-preprocess `--plot`
default-on, run_*.py docstring audit. The current commit is the
follow-up that adds `--raw` and `--align_target` on top.)
