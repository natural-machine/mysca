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

### PyMOL

To visualize sectors on 3D protein structures, install PyMOL separately:

```bash
conda install conda-forge::pymol-open-source
```

## Usage

mysca provides CLI tools that can be chained. The typical workflow is:

1. **`sca-prealign`** — (optional) cluster and align raw (unaligned) sequences
2. **`sca-preprocess`** — filter and weight an aligned MSA
3. **`sca-core`** — run SCA, identify significant components, and assign sectors
4. **`sca-pymol`** — (optional) visualize sectors on a protein structure

For the full list of options for each command, see the [CLI Reference](docs/cli_reference.md).

### Preparing raw sequences (optional)

If you start from unaligned sequences, `sca-prealign` will (optionally) cluster
them to reduce redundancy and then align them into an MSA suitable for
`sca-preprocess`.

```bash
sca-prealign -i <raw.fasta> -o <prealign-outdir>
# or with pre-clustering:
sca-prealign -i <raw.fasta> -o <prealign-outdir> \
    --cluster mmseqs2 --cluster_min_seq_id 0.9
```

The aligned output lives at `<prealign-outdir>/aligned.fasta` (or
`aligned.sto` if `--output_format stockholm`) and can be fed directly to
`sca-preprocess -i` (with matching `--input_format` when Stockholm).

**External tools.** `sca-prealign` shells out to `mafft` (always) and `mmseqs`
(only when `--cluster mmseqs2`). These binaries must be on `PATH` — the
program checks up front and raises `FileNotFoundError` immediately if any
required tool is missing. Install them however you like, e.g.:

```bash
conda install -c bioconda mafft mmseqs2
```

If you need to point at a specific binary (e.g. a non-default install), use
`--align_bin /path/to/mafft` and `--cluster_bin /path/to/mmseqs`.

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

- `--weight_method` — algorithm for sequence weighting (`v3`, `v4`, `v5`, `gpu`; default `v5`)
- `--plot` — generate diagnostic plots

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
- `--kstar` — override the number of significant components
- `--pstar` — percentile threshold for sector assignment (default 95)
- `--sectors_for` — which sequences get per-sequence sector mappings (default: reference only; use `all` for every sequence, or a filepath listing sequence IDs)
- `--save_all` — include large intermediate matrices in the output
- `--use_jax` — use JAX for accelerated computation

### PyMOL Visualization

```bash
sca-pymol \
    -s <scaffold> \
    --pdb_dir <pdb-directory> \
    --modes <core-outdir>/statsectors_seq.npz \
    -r <reference> \
    --groups 0 1 2 \
    --animate
```

Key options:

- `--multisector` — plot all sectors on the same structure
- `--features` — path to a JSON file with annotations
- `--views` — save rotated views

## Python API

The `PreprocessingResults` and `SCAResults` classes provide programmatic access to saved outputs:

```python
from mysca import PreprocessingResults, SCAResults

# Load preprocessing results
prep = PreprocessingResults.load("path/to/preprocessing")
print(prep.n_sequences, prep.n_positions)
print(prep.sequence_weights)

# Load SCA results
sca = SCAResults.load("path/to/scacore")
print(sca.n_sectors)
print(sca.conservation)   # positional conservation (Di)
print(sca.sca_matrix)     # corrected covariance matrix (Cij_corr)
```

All output files use standard formats (`.npz`, `.npy`, `.json`) and can be read without mysca installed.

## Demo

The `demo/SH3/` directory contains a working example using the SH3 protein domain family. To run it:

```bash
cd demo
./run_demo_SH3.sh
```

This runs preprocessing and SCA core analysis on the included SH3 MSA. See `demo/SH3/scripts/` for the individual steps.

## References

[1] N. Halabi, O. Rivoire, S. Leibler, and R. Ranganathan, "Protein Sectors: Evolutionary Units of Three-Dimensional Structure," *Cell* 138, 774 (2009).

[2] O. Rivoire, K. A. Reynolds, and R. Ranganathan, "Evolution-Based Functional Decomposition of Proteins," *PLoS Comput Biol* 12, e1004817 (2016).
