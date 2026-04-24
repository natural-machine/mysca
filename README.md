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
conda install -c bioconda mafft    # sca-prealign; sca-project --aligner mafft_add (default)
conda install -c bioconda hmmer    # sca-project --aligner hmmalign
conda install -c bioconda mmseqs2  # sca-prealign --cluster mmseqs2
```

`environment.yml` lists these as commented-out entries — uncomment whichever you use. Every CLI checks for its binaries up-front and raises `FileNotFoundError` with a clear message if a required tool is missing.

### PyMOL

To visualize sectors on 3D protein structures, install PyMOL separately:

```bash
conda install conda-forge::pymol-open-source
```

## Usage

mysca ships seven CLI tools. The core pipeline chains the first three; the others are opt-in for projection, visualization, and plot replay:

1. **`sca-prealign`** — (optional) cluster and align raw (unaligned) sequences.
2. **`sca-preprocess`** — filter and weight an aligned MSA.
3. **`sca-core`** — run SCA, identify significant components, and assign sectors.
4. **`sca-project`** — project primary amino-acid sequences (in- or out-of-sample) onto an existing SCA result.
5. **`sca-structure`** — lift `sca-project` onto a PDB structure; IC memberships are expressed in the PDB's own residue numbering.
6. **`sca-pymol`** — render sectors on a structure via PyMOL, with user-supplied protein-specific annotations loaded from a Python file.
7. **`sca-plots`** — regenerate diagnostic figures from any of the persisted output directories without rerunning the pipeline.

Full per-flag documentation: [docs/cli_reference.md](docs/cli_reference.md).

### Preparing raw sequences (optional)

If you start from unaligned sequences, `sca-prealign` will (optionally) cluster them to reduce redundancy and then align them into an MSA suitable for `sca-preprocess`.

```bash
sca-prealign -i <raw.fasta> -o <prealign-outdir>
# or with pre-clustering:
sca-prealign -i <raw.fasta> -o <prealign-outdir> \
    --cluster mmseqs2 --cluster_min_seq_id 0.9
```

Aligned output lives at `<prealign-outdir>/aligned.fasta` (or `aligned.sto` if `--output_format stockholm`) and feeds directly into `sca-preprocess -i` (with matching `--input_format` when Stockholm).

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

Key options:

- `--n_boot` — number of bootstrap iterations (default 10)
- `--kstar` — override the bootstrap-derived number of significant components
- `--n_components` — number of ICs to compute (integer or `all`; default `kstar`)
- `--pstar` — percentile threshold for sector assignment (default 95)
- `--assignment overlap|exclusive` — how a residue that clears multiple ICs' cutoffs is placed
- `--sectors_for` — which sequences get per-sequence sector mappings (default reference only; `all` or a text file of IDs)
- `--save_all` — include large intermediate matrices in the output
- `--use_jax` — use JAX for accelerated computation

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

Records whose IDs are already in the reference MSA short-circuit the alignment step; the rest are aligned onto the reference MSA columns via mafft (default) or HMMER.

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
```

The library-level `mysca.structure.SequencePdbMap.from_sifts_for_uniprot_ids([...], pdb_dir="./pdbs")` resolves UniProt IDs to best-available PDBs via EBI's SIFTS service (cached locally) for users who don't want to hand-maintain the TSV.

### PyMOL visualization

```bash
sca-pymol \
    --structure <structure-outdir> \
    [--structure_id <id>] \
    --groups 0 1 2 \
    [--multisector] \
    [--animate] \
    [--features_py <my_features.py> --features show_cofactor,show_ligand] \
    -o <pymol-outdir>
```

Protein-specific annotations (cofactors, ligands, iron-sulfur clusters, etc.) are user-supplied as a Python file with callables of signature `fn(struct, cmd, *, color=None, context=None)`. A worked example lives at [`demo/pymol_features/narg_1q16.py`](demo/pymol_features/narg_1q16.py).

### Replay plots

Regenerate diagnostic figures from any persisted output directory without rerunning the pipeline:

```bash
sca-plots --prealign <prealign-outdir>
sca-plots --preprocessing <preprocessing-outdir>
sca-plots --scacore <core-outdir> --preprocessing <preprocessing-outdir>
```

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
print(sca.n_sectors)
print(sca.conservation)   # positional conservation (Di)
print(sca.sca_matrix)     # corrected covariance matrix (Cij_corr)
print(sca.info())         # printable field-by-field summary
```

All output files use standard formats (`.npz`, `.npy`, `.json`) and can be read without mysca installed.

Projection and structure APIs:

```python
from mysca.project import project_sequences
from mysca.structure import PDBStructure, project_pdb, SequencePdbMap

# Project a new primary sequence onto an existing SCA result
result = project_sequences(
    "new_sequences.fasta",
    sca_result_dir="path/to/scacore",
    preproc_result_dir="path/to/preprocessing",
    aligner="mafft_add",
)
for proj in result.projections:
    print(proj.seq_id, proj.ic_memberships)

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

The demo covers both entry points (preformed MSA and raw FASTA) and walks through preprocessing, SCA core, primary-sequence projection (with both `mafft_add` and `hmmalign` backends), PDB-level projection (against 1SHF chain A), and a plot-replay step. See `demo/SH3/scripts/` for the individual steps (`step0_*` through `step6_*`). `demo/pymol_features/narg_1q16.py` is a reference example of a user features file for `sca-pymol`; it is not exercised by the automated demo because PyMOL is an optional dependency.

## References

[1] N. Halabi, O. Rivoire, S. Leibler, and R. Ranganathan, "Protein Sectors: Evolutionary Units of Three-Dimensional Structure," *Cell* 138, 774 (2009).

[2] O. Rivoire, K. A. Reynolds, and R. Ranganathan, "Evolution-Based Functional Decomposition of Proteins," *PLoS Comput Biol* 12, e1004817 (2016).
