# mysca

An implementation of the Statistical Coupling Analysis (SCA) pipeline for identifying co-evolving groups of amino acid positions (sectors) in protein families from multiple sequence alignments.

Based on:

- Halabi et al., "Protein Sectors: Evolutionary Units of Three-Dimensional Structure," *Cell* 138, 774 (2009).
- Rivoire et al., "Evolution-Based Functional Decomposition of Proteins," *PLoS Comput Biol* 12, e1004817 (2016).

## Setup

Create a conda environment:

```bash
# Local project environment
conda env create -p ./env -f environment.yml
conda activate ./env
```

```bash
# Global environment
conda env create -n mysca-env -f environment.yml
conda activate mysca-env
```

Then install the package:

```bash
python -m pip install -e '.[dev]'
```

Verify the installation:

```bash
pytest tests
```

### Optional external binaries

Several CLIs shell out to external tools. Install what you need via bioconda:

```bash
conda install -c bioconda mafft     # sca-prealign --align mafft (default);
                                    # sca-project --aligner mafft_add (default)
conda install -c bioconda clustalo  # sca-prealign --align clustalo
conda install -c bioconda hmmer     # sca-project --aligner hmmalign
                                    #   (provides hmmbuild + hmmalign)
conda install -c bioconda mmseqs2   # sca-prealign --cluster mmseqs2
```

`environment.yml` lists these as commented-out entries — uncomment whichever you use. Every CLI checks for its binaries up-front and raises `FileNotFoundError` with a clear message if a required tool is missing. In-sample projection and PDB loading require no external binary.

### Optional Python extras

Three pip extras are declared in `pyproject.toml`. `[pymol]` and `[prealign]` are documented placeholders — they install nothing but validate the spelling. The actual binaries must be installed via conda (see "Optional external binaries" above for `prealign`; for PyMOL, `conda install -c conda-forge pymol-open-source`). Only `[mp4]` ships a real pip dependency.

```bash
pip install -e '.[mp4]'     # imageio-ffmpeg — sca-pymol --format mp4 / both
```

The `[mp4]` extra ships a bundled ffmpeg via `imageio-ffmpeg` and is only needed if you want MP4 (rather than GIF) animations out of `sca-pymol --animate`.

## Usage

mysca ships seven CLI tools. The core pipeline chains the first three; the others are opt-in for projection, visualization, and plot replay:

1. **`sca-prealign`** — (optional) cluster and align raw (unaligned) sequences.
2. **`sca-preprocess`** — filter and weight an aligned MSA.
3. **`sca-core`** — run SCA, identify significant components, and assign sectors.
4. **`sca-project`** — project primary amino-acid sequences (in- or out-of-sample) onto an existing SCA result.
5. **`sca-structure`** — lift `sca-project` onto a PDB structure; IC memberships are expressed in the PDB's own residue numbering.
6. **`sca-pymol`** — render sectors on a structure via PyMOL, with user-supplied protein-specific annotations loaded from a Python file.
7. **`sca-plots`** — regenerate diagnostic figures from any of the persisted output directories without rerunning the pipeline.

Full per-flag documentation (including outputs of every CLI): [docs/cli_reference.md](docs/cli_reference.md).

### Preparing raw sequences (optional)

If you start from unaligned sequences, `sca-prealign` will (optionally) cluster them to reduce redundancy and then align them into an MSA suitable for `sca-preprocess`.

```bash
sca-prealign -i <raw.fasta> -o <prealign-outdir>
# or with pre-clustering:
sca-prealign -i <raw.fasta> -o <prealign-outdir> \
    --cluster mmseqs2 --cluster_min_seq_id 0.9
# or with Clustal Omega instead of MAFFT:
sca-prealign -i <raw.fasta> -o <prealign-outdir> \
    --align clustalo --align_args guidetree_out=true output_order=tree-order
```

**Inputs:** raw (unaligned) FASTA file.
**Outputs (under `<prealign-outdir>`):** `aligned.fasta` (or `aligned.sto` when `--output_format stockholm`), `clustered.fasta` if `--cluster` is used, `filter_history.json` (per-stage sequence counts, replayable via `sca-plots --prealign`), `prealign_args.json`, `prealign.log`.

Aligned output feeds directly into `sca-preprocess -i` (with matching `--input_format` when Stockholm).

### Preprocessing

```bash
sca-preprocess \
    -i <input-msa.fasta> \
    -o <preprocessing-outdir> \
    --reference <reference-id> \
    --gap_truncation_thresh 0.4 \
    --sequence_gap_thresh 0.2 \
    --reference_similarity_thresh 0.2 \
    --sequence_similarity_thresh 0.8 \
    --position_gap_thresh 0.2
```

**Inputs:** an aligned MSA (FASTA or Stockholm; pass `--input_format stockholm` for the latter — never inferred from filename).
**Outputs (under `<preprocessing-outdir>`):** `preprocessing_results.npz` (filtered MSA + retained indices + sequence weights + pre-truncation gap frequencies), `msa_binary2d_sp.npz` (sparse one-hot MSA), `sym2int.json`, `msa_orig.fasta-aln` (the unfiltered original MSA), `filter_history.json`, `preprocessing_args.json`, `preprocessing.log`. With `--plot`, also `images/filter_history.png` + `images/filter_distributions.png`.

Key options:

- `--weight_method` — algorithm for sequence weighting (`sparse` default, or `gpu` for torch acceleration)
- `--plot` — emit `filter_history.png` + `filter_distributions.png` to `<outdir>/images/`

### SCA Core

```bash
sca-core \
    -i <preprocessing-outdir> \
    -o <core-outdir> \
    --regularization 0.03 \
    --seed 42
```

**Inputs:** an `sca-preprocess` output directory.
**Outputs (under `<core-outdir>`):** `scarun_results.npz` (`Dia`, `conservation`, `sca_matrix`, `phi_ia`, `fi0`, `fia`, `Cij_raw`; plus `Cijab_raw`/`fijab` with `--save_all`), `sca_eigendecomp.npz`, `ic_positions/ic_<i>_msaproc.npy` + `ic_<i>_msaorig.npy` + `ic_<i>_loadings.npy` (per-IC high-load positions in both MSA coord spaces, plus loadings), `ic_residues_per_seq.npz` and `ic_loadings_per_seq.npz` (per-target raw-residue indices and loadings, keyed `ic_<i>_<seqid>`), `sca_results/` (ICA + bootstrap byproducts, scalar text files), `scarun_args.json`, `scarun.log`, and `images/` (conservation, SCA matrix, spectrum, dendrogram, t-distributions, EV/IC scatter sweeps, sector subset, and `seq_proj_ic0v1.png` — sequences projected onto the first two ICs). Optionally `seq_projections.tsv` (`--save_dataframe`) and `sequence_metadata.tsv` (`--seq_metadata`).

Key options:

- `--n_boot` — number of bootstrap iterations (default 10; `0` reuses an existing bootstrap; `-1` skips bootstrapping entirely)
- `--kstar` — override the bootstrap-derived number of significant components
- `--n_components` — number of ICs to compute (integer or `all`; default `kstar`)
- `--pstar` — percentile threshold for sector assignment (default 95)
- `--assignment overlap|exclusive` — how a residue that clears multiple ICs' cutoffs is placed
- `--sectors_for` — which target sequences expand into the per-seq output files (default reference only; `all` or a text file of IDs)
- `--save_all` — include large intermediate matrices in the output
- `--save_dataframe` — also write `seq_projections.tsv` (per-sequence Uᵖ scores) for every retained sequence
- `--seq_metadata <tsv>` — optional per-sequence metadata TSV (`seq_id` column + arbitrary others); persisted alongside results and merged into `seq_projections.tsv`
- `--seq_proj_color_by <column>` — color the `seq_proj_ic0v1.png` plot by a metadata column (numeric → colorbar, categorical → legend)
- `--accelerator {none,gpu}` — flips per-step kernel defaults to GPU variants when set to `gpu`
- `--precision {fp64,fp32,fp16}` — GPU compute precision for `fijab` / eigvalsh-bootstrap kernels (default `fp64`; ignored on CPU)
- `--use_jax` — DEPRECATED alias for `--freq_method=jax`

### Project a sequence

Given an SCA result, project a new amino-acid sequence (in- or out-of-sample) and read off which residues fall into each IC group:

```bash
sca-project \
    -i <sequences.fasta> \
    --preprocessing <preprocessing-outdir> \
    --scacore <core-outdir> \
    -o <project-outdir> \
    [--aligner mafft_add|hmmalign]
```

Records whose IDs are already in the reference MSA short-circuit the alignment step; the rest are aligned onto the reference MSA columns via MAFFT (default) or HMMER.

**Inputs:** a FASTA of sequences to project, plus the upstream `sca-preprocess` and `sca-core` output directories.
**Outputs (under `<project-outdir>`):** `projection.json` (per-sequence: `seq_id`, `raw_sequence`, `aligned_sequence`, `residue_by_processed_col`, `ic_residues`, `ic_loadings`, `ic_processed_cols`, `in_sample`, `up_score` — the sequence's Uᵖ row of length `n_components`, or `null` when the source SCAResults lacks the eigendecomposition fields), `per_sequence/<seqid>_residues.tsv` (one row per IC residue), `projection_args.json`, `projection.log`. With `--save_dataframe`, also `seq_projections.tsv` (per-sequence Uᵖ scores in tabular form).

### Project a PDB structure

`sca-structure` composes over `sca-project` to land IC memberships in the structure's own residue numbering:

```bash
# Single PDB
sca-structure -s <protein.pdb> --chain A \
    --preprocessing <preprocessing-outdir> \
    --scacore <core-outdir> \
    -o <structure-outdir>

# Batch via a seq_id → pdb_path TSV
sca-structure --seq_map <seq_to_pdb.tsv> \
    --preprocessing <preprocessing-outdir> \
    --scacore <core-outdir> \
    -o <structure-outdir>

# Batch via UniProt → PDB resolution (SIFTS best_structures)
sca-structure --uniprot_ids P06241 P12931 \
    --pdb_dir <dir-of-pre-downloaded-pdbs> \
    --preprocessing <preprocessing-outdir> \
    --scacore <core-outdir> \
    -o <structure-outdir>
```

Exactly one of `-s/--structure`, `--seq_map`, or `--uniprot_ids` is required. `--uniprot_ids` resolves accessions via EBI's SIFTS service (responses cached under `--cache_dir`, default `./.sifts_cache`); the resolved PDBs must already exist in `--pdb_dir` (SIFTS does not download structures).

**Inputs:** one of (single PDB, seq_id → pdb_path TSV, UniProt ID list + `--pdb_dir`); plus the upstream `sca-preprocess` and `sca-core` directories.
**Outputs (under `<structure-outdir>`):** `structure_projection.json` (per-structure: `structure_id`, `chain_id`, full `sequence_projection` from `sca-project`, `ic_pdb_residues` keyed by IC index, `pdb_path`), `per_structure/<structure_id>_ic_residues.tsv` (one row per IC residue, including raw + PDB residue numbers), `structure_args.json`, `structure.log`.

The library-level `mysca.structure.SequencePdbMap.from_sifts_for_uniprot_ids([...], pdb_dir="./pdbs")` resolves UniProt IDs to best-available PDBs via EBI's SIFTS service (cached locally) for users who don't want to hand-maintain the TSV.

### PyMOL visualization

```bash
sca-pymol \
    --structure <structure-outdir> \
    [--structure_id <id>] \
    --groups 0 1 2 \
    [--multisector] \
    [--animate] [--mode {spin,reveal}] [--format {gif,mp4,both}] \
    [--features_py <my_features.py> --features show_cofactor,show_ligand] \
    -o <pymol-outdir>
```

**Inputs:** an `sca-structure` output directory (reads `structure_projection.json`).
**Outputs (under `<pymol-outdir>`):** `<structure_id>_group<N>.png` (one per IC group; or `<structure_id>_groups_<idxs>.png` under `--multisector`), `views/` (with `--views`), per-frame PNGs in `frames/<basename>_frames/` and a `<basename>.gif` (or `.mp4`) per render under `--animate`, `pymol.log`. See the [CLI reference](docs/cli_reference.md#sca-pymol) for animation modes (`spin` / `reveal`) and ray-tracing knobs.

Requires the optional `pymol-open-source` dependency (`conda install -c conda-forge pymol-open-source`); MP4 output additionally requires `imageio-ffmpeg` (`pip install -e '.[mp4]'`).

Protein-specific annotations (cofactors, ligands, iron-sulfur clusters, etc.) are user-supplied as a Python file with callables of signature `fn(struct, cmd, *, color=None, context=None)`. A worked example lives at [`demo/pymol_features/narg_1q16.py`](demo/pymol_features/narg_1q16.py).

### Replay plots

Regenerate diagnostic figures from any persisted output directory without rerunning the pipeline:

```bash
sca-plots --prealign <prealign-outdir>
sca-plots --preprocessing <preprocessing-outdir>
sca-plots --scacore <core-outdir> --preprocessing <preprocessing-outdir>
```

**Inputs:** any combination of `--prealign`, `--preprocessing`, `--scacore` directories from prior runs. At least one must be passed.
**Outputs:** plots written into each stage's own `images/` subdirectory by default, or all into `--imgdir DIR` when given.

Each flag is opt-in. When `--scacore` and `--preprocessing` are both given, all plot variants (including the positional conservation plot) are produced.

## Python API

The `PreprocessingResults` and `SCAResults` classes provide programmatic access to saved outputs:

```python
from mysca.results import PreprocessingResults, SCAResults

# Load preprocessing results
prep = PreprocessingResults.load("path/to/preprocessing")
print(prep.n_sequences, prep.n_positions)
print(prep.sequence_weights)

# Load SCA results
sca = SCAResults.load("path/to/scacore")
print(sca.n_ic_positions)
print(sca.conservation)   # positional conservation (Di)
print(sca.sca_matrix)     # corrected covariance matrix (Cij_corr)
print(sca.info())         # printable field-by-field summary

# Sequence-space projection (Rivoire et al. 2016 Eqs. 14–15): Uᵖ
# coordinates for any one-hot sequence tensor (M, L_proc, D) — both
# in-sample (prep.msa_binary3d) and out-of-sample.
up = sca.project_sequences(prep.msa_binary3d)   # (M, n_components)

# Tabular view: seq_id, aligned_sequence, up_0, ..., up_{k-1}, plus
# any columns merged in from sca.sequence_metadata.
df = sca.to_dataframe(prep)
```

All output files use standard formats (`.npz`, `.npy`, `.json`, `.tsv`) and can be read without mysca installed.

Projection and structure APIs:

```python
from mysca.project import project_sequences
from mysca.structure import PDBStructure, project_pdb, SequencePdbMap

# Project a new primary sequence onto an existing SCA result. Each
# SequenceProjection now carries an `up_score` (length n_components):
# the sequence's Uᵖ row in IC sequence-space.
result = project_sequences(
    "new_sequences.fasta",
    sca_result_dir="path/to/scacore",
    preproc_result_dir="path/to/preprocessing",
    aligner="mafft_add",
)
for proj in result.projections:
    print(proj.seq_id, proj.ic_residues, proj.up_score)

# Stacked Uᵖ matrix and a tabular view across all projected sequences.
result.up_scores         # (M, n_components) np.ndarray
result.to_dataframe()    # seq_id / aligned_sequence / raw_sequence /
                         # in_sample / up_0 .. up_{k-1}

# Load a PDB and project it
pdb = PDBStructure.from_file("1SHF.pdb", chain="A")
proj = project_pdb(
    pdb,
    sca_result_dir="path/to/scacore",
    preproc_result_dir="path/to/preprocessing",
)
print(proj.ic_pdb_residues)  # per-IC lists of PDB residue numbers
```

## Demo

The `demo/SH3/` directory contains a working example using the SH3 protein domain family. It exercises the whole pipeline end-to-end on the Pfam PF00018 seed alignment:

```bash
cd demo
./run_demo_SH3.sh
```

The demo covers both entry points (preformed MSA and raw FASTA) and walks through preprocessing, SCA core, primary-sequence projection (with both `mafft_add` and `hmmalign` backends), PDB-level projection (against 1SHF chain A), a plot-replay step, and a PyMOL rendering step. See `demo/SH3/scripts/` for the individual steps (`step0_*` through `step7_*`). `step7_pymol.sh` skips automatically when `pymol-open-source` isn't installed; install it via `conda install -c conda-forge pymol-open-source` to exercise the full pipeline. `demo/pymol_features/narg_1q16.py` is a reference example of a user features file — it isn't invoked by the default SH3 demo since the 1SHF structure has no NarG cofactors.

## References

[1] N. Halabi, O. Rivoire, S. Leibler, and R. Ranganathan, "Protein Sectors: Evolutionary Units of Three-Dimensional Structure," *Cell* 138, 774 (2009).

[2] O. Rivoire, K. A. Reynolds, and R. Ranganathan, "Evolution-Based Functional Decomposition of Proteins," *PLoS Comput Biol* 12, e1004817 (2016).
