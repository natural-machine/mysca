# Plan — `mysca.project` and `mysca.structure` subpackages

Planning doc drafted 2026-04-22, to be executed in a future session.
Not yet started. Commit this alongside the session note so the plan
survives context compaction.

## Goal

Extend `mysca` so that work done on a "training" MSA (preprocessing +
SCA + IC analysis) can be applied to **new** amino-acid sequences that
were **not** part of the training MSA. Some of those sequences will
have known 3D structures; many will not. Two subpackages, landing in
order:

1. **`mysca.project`** — out-of-sample **primary-sequence**
   projection onto an existing SCA result. Fundamental, smaller in
   scope, covers the "given a new sequence, which sectors light up?"
   use case.
2. **`mysca.structure`** — PDB / tertiary integration. Depends on
   `mysca.project` for the primary-sequence step. Enables "project a
   PDB structure onto our ICs" and "given a training-set sequence
   with a known PDB, show its sector residues on the structure."

Rationale for the split: the primary-sequence path is the building
block. A PDB is fundamentally "a primary sequence + 3D coordinates",
so the structure subpackage can reuse everything `project` does and
then layer the residue-index-to-PDB-residue concern on top. Shipping
`project` first also gives the user a usable feature even if
`structure` slips.

## Coordinate systems (extension of what's already in `results.py`)

The existing pipeline has three coordinate systems (see the session
note [session_2026-04-22_replay_cli_docs_mapping_info.md](session_2026-04-22_replay_cli_docs_mapping_info.md)):

1. **Original MSA column** `[0..L_orig)`.
2. **Processed MSA column** `[0..L_proc)`. Bridge: `retained_positions`.
3. **Raw sequence residue** (per training sequence). Bridge:
   `get_rawseq_indices_of_msa`.

`mysca.project` adds a fourth system:

4. **Out-of-sample sequence residue** (per *new* sequence). Bridge:
   an alignment-derived map `new_resi ↔ L_orig column` produced on
   demand. This composes with the existing `retained_positions` to
   map a new sequence's residues to processed-MSA columns and
   therefore to IC groups.

`mysca.structure` adds a fifth:

5. **PDB residue** (per chain). Bridge: the sequence → PDB map,
   either supplied (in-sample case, user hands us `seq_id → (pdb,
   chain)`) or derived on demand (SIFTS / UniProt-PDB lookup).

## `mysca.project` — design

### Public API (sketch)

```python
from mysca.project import (
    SequenceProjection,        # one projected sequence
    ProjectionResult,          # batch result, printable summary
    align_to_msa,              # dispatch entry for out-of-sample align
    ALIGNERS,                  # {"mafft_add": ..., "hmmalign": ...}
    project_sequences,         # top-level driver
)

# Typical call:
result = project_sequences(
    sequences=["ACDEFGHIKL…", "WQEIMNPRST…"],  # or read from a FASTA path
    sca_result_dir="path/to/scacore_outdir",
    preproc_result_dir="path/to/preprocessing_outdir",
    aligner="mafft_add",           # default
    cache_dir=".mysca_cache/",      # optional; caches derived HMMs / alignments
)
print(result.info())
```

`project_sequences` returns a `ProjectionResult` containing, per
input sequence:

- `aligned_to_original` — the input sequence string with gaps inserted
  so its length equals `L_orig`.
- `residue_idx_by_processed_col` — array length `L_proc`; each entry
  is the raw residue index in the input sequence at that processed
  MSA column, or `-1` if the input sequence has a gap there.
- `ic_memberships` — list length `n_components`; each entry is the
  set of input-sequence residue indices falling in IC group `i`.
- `ic_scores` — parallel to `ic_memberships`; projection of the
  binary/one-hot residue onto the normalized IC vector (`v_ica[col,
  i]`) summed over the group residues.

The result exposes an `info()` method (same pattern as
`PreprocessingResults.info()`) describing each field.

### Alignment dispatch

`ALIGNERS` is a dict of `name → callable(new_fasta, msa_obj_orig,
workdir) → aligned_fasta`. Default is `mafft_add`:

```python
def _mafft_add(new_fasta_path, msa_obj_orig, workdir):
    # 1. Write msa_obj_orig to workdir/ref.fasta
    # 2. Run: mafft --add <new.fasta> --keeplength ref.fasta > out.fasta
    # 3. Parse out.fasta and return only the newly-added rows aligned
    #    to the same length as ref (--keeplength guarantees this).
    ...
```

`--keeplength` is load-bearing: it prevents mafft from inserting new
columns to accommodate the new sequences, which would break
`retained_positions` indexing.

HMM path is registered but unimplemented in the first commit. The
signature is the same; the registration placeholder is there so
adding it later is a one-function change. Registration API:

```python
@register_aligner("hmmalign")
def _hmmalign(new_fasta_path, msa_obj_orig, workdir):
    ...
```

### In-sample shortcut

If the input sequence's ID is already in `msa_obj_orig`, skip
alignment entirely. The existing `get_rawseq_indices_of_msa` gives us
the column map; the rest of `project_sequences` is a pure index
lookup. This also means the same API serves both "analyze training-
set sequences" and "analyze new ones" — just with different speed.

### CLI

```bash
sca-project -i new_sequences.fasta \
    --scacore path/to/scacore_outdir \
    --preprocessing path/to/preprocessing_outdir \
    -o projection_out \
    [--aligner mafft_add] \
    [--cache-dir .mysca_cache/]
```

Writes `projection_out/`:

- `projection.json` — per-sequence summary (residue counts per IC,
  IDs, alignment source).
- `per_sequence/<seqid>_residues.tsv` — residue-level detail.
- `projection_args.json`, `projection.log`.

### Tests

- Round-trip: project each training sequence back onto its own SCA
  result; residue→IC memberships should exactly match what
  `statsectors_msa.npz` already has for that sequence.
- Non-trivial out-of-sample sequence: hand-construct a new sequence
  that differs from an MSA entry by 2 residues; verify those
  residues' IC memberships are consistent with the original.
- Aligner dispatch: verify `ALIGNERS["mafft_add"]` is called when
  default; verify an unknown alignment name raises.
- In-sample shortcut: project a sequence whose ID IS in the MSA;
  verify no alignment binary is invoked.

### Dependencies

- `mafft` on PATH (already required by `sca-prealign`). Make
  `sca-project` gated on it with the same `_resolve_bin` pattern.
- No new Python deps.

## `mysca.structure` — design (lands after `mysca.project`)

### Public API (sketch)

```python
from mysca.structure import (
    PDBStructure,          # load / parse / residue index
    SequencePdbMap,        # seq_id → (pdb_id, chain), with lookup
    project_pdb,           # wraps mysca.project.project_sequences
    project_groups_to_pdb, # groups + col_map → pdb_resi sets
)

struct = PDBStructure.from_file("1abc.pdb", chain="A")
# or
struct = PDBStructure.fetch("1abc", chain="A", cache_dir=".pdb_cache/")

proj = project_pdb(
    struct,
    sca_result_dir="path/to/scacore_outdir",
    preproc_result_dir="path/to/preprocessing_outdir",
)
pdb_sectors = proj.pdb_sectors  # dict[ic_idx → list[pdb_resi]]
```

### Sequence → PDB map

Two lookup sources, configurable:

- **User-supplied** (`SequencePdbMap.from_tsv("map.tsv")`) — two
  columns `seq_id\tpdb_id[:chain]`. Primary, explicit path.
- **SIFTS on demand**
  (`SequencePdbMap.from_sifts_for_uniprot_ids(seq_ids)`) — pulls
  EBI's SIFTS mapping, caches locally under `~/.mysca/sifts_cache/`.
  Explicit opt-in; never called unless the user asks.

### CLI

```bash
sca-structure -s <pdb_id_or_path> [--chain A] \
    --scacore path/to/scacore_outdir \
    --preprocessing path/to/preprocessing_outdir \
    -o structure_out \
    [--seq-map map.tsv] \
    [--pdb-lookup sifts]
```

### Relation to `sca-pymol`

Current `sca-pymol` consumes `statsectors_seq.npz` (raw-sequence
coordinates). Once `mysca.structure` lands, `sca-pymol` should
eventually accept a `structure_out/` directory and read
`pdb_sectors.json` directly — one coordinate system in/out. This
is a follow-up rewrite, not part of the initial landing.

### Tests

- Small single-chain PDB fixture under `tests/_data/pdbs/`.
- In-sample case: MSA sequence ID maps to a PDB via TSV; verify
  the PDB residue sector set equals what the existing
  `statsectors_seq.npz` has for that sequence.
- Out-of-sample case: PDB whose sequence is NOT in the MSA;
  verify the alignment-then-project path produces a non-empty
  `pdb_sectors` for at least one IC.
- SIFTS lookup: mocked HTTP response (do not hit the network in
  tests); verify parsing and caching behavior.

## Commit plan

First commit: `mysca.project` only.

1. `src/mysca/project/__init__.py`, `alignment.py`,
   `projection.py`, CLI (`run_project.py`) + `sca-project` entry.
2. Corresponding tests under `tests/test_project_*.py`.
3. Update [docs/cli_reference.md](../../docs/cli_reference.md)
   (follow the doc-update rule already in CLAUDE.md: help string,
   top docstring, cli_reference all in the same commit).
4. Update [CLAUDE.md](../../CLAUDE.md)'s "five CLIs" line to six.

Second commit (follow-up, separate session): `mysca.structure`.

## Open scoping questions answered this session

- **Name**: `mysca.project` + `mysca.structure`. The primary-sequence
  path lives under `project` since "structure" is overloaded; the
  user accepted this split explicitly in chat.
- **Default aligner**: `mafft --add`. HMMER/hmmalign registered as an
  option, not implemented in the first commit.
- **Sequence → PDB map**: user-supplied TSV (primary) **plus**
  on-demand SIFTS/UniProt lookup (opt-in).
- **First commit scope**: `mysca.project` only. `mysca.structure`
  lands in a follow-up.
