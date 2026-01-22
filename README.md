# mysca

## Description

<!-- TODO: Add description -->

## Setup


Create either a global or local conda environment as follows:

```bash
# Local project environment
mamba env create -p ./env -f environment.yml
conda activate env
```

```bash
# Global project environment (replace mysca-env if desired)
mamba env create -n mysca-env -f environment.yml
conda activate env
```

Next, install the project source code.
<!-- TODO: Add instructions for direct pip install via github -->
Clone the repository, and then from the project directory, activate the environment and run:

```bash
conda activate <env-name>
python -m pip install -e '.[dev]'
```

Verify things have installed successfully by running:

```bash
pytest tests
```

If plotting results with `pymol` is desired, this must be installed separately.
We use the open source project, installable via conda:

```bash
conda install conda-forge::pymol-open-source
```

## Usage

There are three entrypoints to this project: `sca-preprocess`, `sca-core`, and `sca-pymol`.
See the demo directory for example usage.

### Preprocessing

```bash
sca-preprocess \
    -i <input-msa> \
    -o <preprocessing-outdir> \
    --gap_truncation_thresh 0.4 \
    --sequence_gap_thresh 0.2 \
    --reference <reference-id> \
    --reference_similarity_thresh 0.2 \
    --sequence_similarity_thresh 0.8 \
    --position_gap_thresh 0.2
```

### SCA Core

```bash
sca-core \
    -i <preprocessing-outdir> \
    -o <core-outdir> \
    --regularization 0.03 \
    --seed 42
```

### Pymol plots

```bash
sca-pymol \
    -s <scaffold> \
    -r <reference> \
    --pdb_dir <pdb-directory> \
    --modes <core-outdir>/statsectors_seq.npz \
    --outdir <pymol-outdir> \
    --features <features-fpath> \
    --groups "-1" \
    --animate 
```

## References

[1] N. Halabi, O. Rivoire, S. Leibler, and R. Ranganathan, Protein Sectors: Evolutionary Units of Three-Dimensional Structure, Cell 138, 774 (2009).

[2] O. Rivoire, K. A. Reynolds, and R. Ranganathan, Evolution-Based Functional Decomposition of Proteins, PLoS Comput Biol 12, e1004817 (2016).
