# CLI Reference

mysca provides three command-line tools for running Statistical Coupling Analysis.

---

## sca-preprocess

Load a multiple sequence alignment (MSA), filter sequences and positions, and compute sequence weights.

### Usage

```bash
sca-preprocess -i <input-msa> -o <output-dir> [options]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `-i, --msa_fpath` | Filepath of input MSA in FASTA format |
| `-o, --outdir` | Output directory |

### SCA Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--gap_truncation_thresh` | 0.4 | Remove columns with gap frequency above this threshold (tau) |
| `--sequence_gap_thresh` | 0.2 | Remove sequences with gap frequency above this threshold (gamma_seq) |
| `--reference` | None | Reference sequence ID in the MSA |
| `--reference_similarity_thresh` | 0.2 | Minimum similarity to the reference sequence (Delta) |
| `--sequence_similarity_thresh` | 0.8 | Clustering threshold for sequence weighting (delta) |
| `--position_gap_thresh` | 0.2 | Remove columns with weighted gap frequency above this threshold (gamma_pos) |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-v, --verbosity` | 1 | Verbosity level |
| `--pbar` | off | Enable progress bar |
| `--plot` | off | Generate plots |
| `--syms` | "default" | Symbol set ("default" or custom) |
| `--gapsym` | "-" | Gap symbol |
| `--weight_method` | "v5" | Method for computing sequence weights. Choices: `v3`, `v4`, `v5`, `gpu`. See [weight methods](weight_methods.md) for details |
| `--block_size` | 512 | Block size for weight computations |

### Weight Methods

See [weight_methods.md](weight_methods.md) for full details, including method internals and known issues.

| Method | Description |
|--------|-------------|
| `v3` | Direct integer comparison (slow) |
| `v4` | Sparse matrix, row-by-row (medium) |
| `v5` | Sparse matrix, vectorized (default, recommended) |
| `gpu` | PyTorch GPU-accelerated computation |

### Output

Writes to the specified output directory:
- `preprocessing_results.npz` — filtered MSA, retained indices, sequence weights
- `preprocessing_args.json` — arguments used
- `sym2int.json` — symbol-to-integer mapping
- `msa_binary2d_sp.npz` — sparse binary MSA representation
- `msa_orig.fasta-aln` — original MSA

---

## sca-core

Run SCA on preprocessed data: compute the SCA matrix, perform eigendecomposition, bootstrap for significance, run ICA, and assign sectors.

### Usage

```bash
sca-core -i <preprocessing-dir> -o <output-dir> [options]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `-i, --indir` | Path to preprocessed data (output of `sca-preprocess`) |
| `-o, --outdir` | Output directory |

### SCA Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--regularization` | 0.03 | Regularization parameter (lambda) |
| `-nb, --n_boot` | 10 | Number of bootstrap iterations for eigenvalue significance |
| `-k, --kstar` | 0 | Override the number of significant components (0 = use bootstrap estimate) |
| `-p, --pstar` | 95 | Percentile threshold for IC group assignment |
| `--weak_assignment` | [] | Weak assignment list (variadic integers) |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--seed` | None | Random seed for reproducibility |
| `--sectors_for` | None | Which sequences get per-sequence sector mappings. `None`: reference only. `"all"`: every retained sequence. Or a path to a text file with one sequence ID per line |
| `--save_all` | off | Save all results including large intermediate matrices (Cijab_raw, fijab) |
| `--use_jax` | off | Use JAX for accelerated computations |
| `--nodendro` | off | Skip dendrogram plots |
| `--load_data` | "" | Path to a previous SCA output directory to load precomputed data |
| `--sector_cmap` | "default" | Sector colormap. Choices: `none`, `default` |
| `-v, --verbosity` | 1 | Verbosity level |
| `--pbar` | off | Enable progress bar |

### Output

Writes to the specified output directory:
- `scarun_results.npz` — core SCA results (conservation, SCA matrix, eigendecomposition)
- `scarun_args.json` — arguments used
- `sca_eigendecomp.npz` — eigenvalues and eigenvectors
- `statsectors_msa.npz` — per-sequence sector mappings (MSA coordinates)
- `statsectors_seq.npz` — per-sequence sector mappings (raw sequence coordinates)
- `sca_results/` — detailed results (eigenvalues, eigenvectors, ICA components, sector assignments, t-distribution info)
- `groups/` — per-group position lists
- `images/` — plots (conservation, SCA matrix, spectrum, dendrogram, etc.)

---

## sca-pymol

Visualize sectors on 3D protein structures using PyMOL.

Requires the optional `pymol-open-source` dependency (see [installation](#pymol)).

### Usage

```bash
sca-pymol -s <scaffold> --pdb_dir <pdb-dir> --modes <modes-file> [options]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `-s, --scaffold` | Scaffold protein identifier |
| `--pdb_dir` | Directory containing PDB files |
| `--modes` | Path to modes file (`.npz`, typically `statsectors_seq.npz`) |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-r, --reference` | None | Reference scaffold to align to |
| `-o, --outdir` | None | Output directory |
| `--groups` | -1 | Group indices (0-indexed) to plot. `-1` plots all groups |
| `--features` | None | Path to JSON file with annotations to include |
| `--multisector` | off | Plot sectors simultaneously on the same protein |
| `--views` | off | Save rotated views of the structure |
| `--animate` | off | Generate animation (GIF) |
| `--nframes` | None | Number of frames for animation |
| `--duration` | None | Duration in seconds for animation |
| `--show_molybdenum` | off | Show molybdenum atoms |
| `-v, --verbosity` | 1 | Verbosity level |
