# CLI Reference

mysca provides command-line tools for running Statistical Coupling Analysis.

---

## sca-prealign

Optionally cluster and then align raw (unaligned) FASTA sequences, producing an aligned MSA suitable for `sca-preprocess`.

### Usage

```bash
sca-prealign -i <raw.fasta> -o <output-dir> [options]
```

The aligned output is written to `<output-dir>/aligned.fasta`.

### Required Arguments

| Argument | Description |
|----------|-------------|
| `-i, --input_fpath` | Filepath of the input (raw) FASTA |
| `-o, --outdir` | Output directory |

### Clustering

| Argument | Default | Description |
|----------|---------|-------------|
| `--cluster` | `none` | Clustering method. Choices: `none`, `mmseqs2` |
| `--cluster_min_seq_id` | 0.9 | Minimum sequence identity for clustering |
| `--cluster_coverage` | 0.8 | Minimum coverage for clustering |
| `--cluster_cov_mode` | 1 | mmseqs2 coverage mode |
| `--cluster_threads` | 1 | Threads for the clustering tool |
| `--cluster_bin` | (from PATH) | Explicit path to the clustering binary |

### Alignment

| Argument | Default | Description |
|----------|---------|-------------|
| `--align` | `mafft` | Alignment method. Choices: `mafft` |
| `--align_threads` | 1 | Threads for the alignment tool |
| `--align_bin` | (from PATH) | Explicit path to the alignment binary |
| `--align_extra` | [] | Extra arguments passed through to the aligner |
| `--output_format` | `fasta` | Format of the aligned output. Choices: `fasta`, `stockholm`. The output filename is `aligned.fasta` or `aligned.sto` accordingly |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-v, --verbosity` | 1 | Verbosity level |
| `--pbar` | off | Enable progress bar |
| `--plot` | off | Write a per-stage sequence-count diagnostic plot to `outdir/images/prealign_filter_history.png` |

### Output

Writes to the specified output directory:
- `aligned.fasta` or `aligned.sto` — aligned MSA (depending on `--output_format`)
- `clustered.fasta` — clustered FASTA (only when `--cluster mmseqs2`)
- `filter_history.json` — per-stage sequence counts (initial / cluster / align); always persisted so `sca-plots` can replay the diagnostic plot later
- `prealign_args.json` — arguments used
- `prealign.log` — run log
- `images/prealign_filter_history.png` — only when `--plot` is passed (replay it later with `sca-plots --prealign`)

### External Binaries

`mafft` (always) and `mmseqs` (when `--cluster mmseqs2`) must be resolvable on `PATH` — the CLI checks up front and raises `FileNotFoundError` immediately if a required tool is missing. Install e.g. via `conda install -c bioconda mafft mmseqs2`, or pass explicit paths with `--align_bin` / `--cluster_bin`.

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
| `-i, --msa_fpath` | Filepath of input MSA |
| `-o, --outdir` | Output directory |

### SCA Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--gap_truncation_thresh` | 0.4 | Remove columns with gap frequency above this threshold (τ). Applied before any sequence-level filtering |
| `--sequence_gap_thresh` | 0.2 | Remove sequences with gap frequency above this threshold (γ_seq) |
| `--reference` | None | Reference sequence ID in the MSA. When set, sequences that diverge too far from the reference are filtered and the reference's raw-residue coordinates are used in downstream logs |
| `--reference_similarity_thresh` | 0.2 | Minimum similarity to the reference sequence (Δ). Requires `--reference`; sequences whose fractional identity to the reference falls below this value are dropped |
| `--sequence_similarity_thresh` | 0.8 | Clustering threshold for sequence weighting (δ). Sequences within this pairwise similarity contribute down-weighted to the SCA statistics |
| `--position_gap_thresh` | 0.2 | Remove columns with *weighted* gap frequency above this threshold (γ_pos). Applied after sequence weighting |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-v, --verbosity` | 1 | Verbosity level |
| `--pbar` | off | Enable progress bar |
| `--plot` | off | Emit `filter_history.png` and `filter_distributions.png` to `outdir/images/` |
| `--input_format` | `fasta` | Format of the input MSA file. Choices: `fasta`, `stockholm`. Never inferred from the filename |
| `--syms` | `default` | Symbol alphabet. `default` → standard 20 amino acids; `none` → disable excluded-symbol filtering and auto-detect; any other string is treated as an explicit character set |
| `--gapsym` | `-` | Gap symbol in the input MSA |
| `--gap_value` | 0 | Integer assigned to the gap symbol in the `SymMap`. Default 0 (gap first). Pass `len(aa_syms)` (e.g. 20) to place the gap at the end (legacy behavior) |
| `--weight_method` | `sparse` | Sequence-weight computation backend. `sparse` uses a CPU sparse-CSR implementation; `gpu` dispatches to torch (CUDA/MPS/XPU), falling back to `sparse` if no accelerator is detected. See [weight methods](weight_methods.md) for full details |
| `--block_size` | 512 | Block size for relevant weight computations |

### Output

Writes to the specified output directory:

- `preprocessing_results.npz` — filtered MSA, retained indices, sequence weights, pre-truncation gap frequencies
- `preprocessing_args.json` — arguments used
- `sym2int.json` — symbol-to-integer mapping
- `msa_binary2d_sp.npz` — sparse one-hot MSA
- `msa_orig.fasta-aln` — original MSA (written before any filtering)
- `filter_history.json` — per-stage filter diagnostics (counts + threshold + stat distribution); always persisted so `sca-plots` can replay plots later
- `preprocessing.log` — run log
- `images/` — only when `--plot` is passed (`filter_history.png`, `filter_distributions.png`)

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
| `--regularization` | 0.03 | SCA regularization parameter (λ) |
| `--background` | None | Optional JSON file mapping each amino-acid symbol to a background frequency. When omitted, the built-in default (`DEFAULT_BACKGROUND_FREQ`) is used |
| `-nb, --n_boot` | 10 | Number of bootstrap iterations used to determine the eigenvalue significance cutoff. `0` loads existing bootstrap output if available; `-1` skips bootstrapping and treats all components as significant |
| `-k, --kstar` | 0 | Override the bootstrap-derived number of significant components. `0` (default) uses the bootstrap estimate |
| `--n_components` | None | Number of ICs to compute. Positive integer or `all` (meaning `L`, the number of retained positions). Default: `kstar`. Values below `kstar` are clamped up |
| `-p, --pstar` | 95 | Percentile defining the t-distribution cutoff that nominates positions for IC groups |
| `--assignment` | `overlap` | How to assign a position that clears the cutoff on multiple ICs. `overlap`: keep it in every qualifying IC (default). `exclusive`: assign only to the IC where its projection is maximal. `--weak_assignment` applies only under `exclusive` |
| `--weak_assignment` | [] | IC indices to exclude from the `exclusive`-assignment tie-break (variadic integers). Ignored under `overlap` |
| `--n_logged_comps` | 10 | Number of top ICs to summarize in the log after assignment (significance marker, eigenvalue, and MSA positions in processed / unprocessed / reference coordinates). `0` disables the summary |
| `--sectors_for` | None | Which sequences get per-sequence sector mappings. `None`: reference only. `all`: every retained sequence. Otherwise: path to a text file with one sequence ID per line |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--seed` | None | Random seed for reproducibility. `None` or non-positive auto-generates one |
| `--save_all` | off | Save large intermediate matrices (`Cijab_raw`, `fijab`) into `scarun_results.npz` |
| `--use_jax` | off | Use JAX in the core SCA computations |
| `--nodendro` | off | Skip dendrogram and sequence-similarity plots |
| `--load_data` | "" | Path to a previous SCA output directory to load precomputed data (skips recomputation) |
| `--sector_cmap` | `default` | Sector colormap for the SCA-matrix sector-subset plot. Choices: `none`, `default` |
| `-v, --verbosity` | 1 | Verbosity level |
| `--pbar` | off | Enable progress bar |

### Output

Writes to the specified output directory:

- `scarun_results.npz` — core SCA results (`Dia`, `conservation`, `sca_matrix`, `phi_ia`, `fi0`, `fia`; optionally `Cijab_raw`, `fijab` with `--save_all`)
- `sca_eigendecomp.npz` — full + significant eigenvalues/eigenvectors
- `scarun_args.json` — arguments used
- `statsectors_msa.npz` / `statsectors_seq.npz` — per-sequence sector mappings in processed-MSA and raw-sequence coordinates; only the top-`kstar` IC groups are expanded per sequence
- `sca_results/` — `v_ica_normalized.npy`, `w_ica.npy`, `t_dists_info.json`, `evals_shuff.npy`, `sca_matrix_sector_subset.npy`, scalar text files (`kstar.txt`, `n_components.txt`, etc.), `msa_sectors/sector_*_msapos.npy` + `sector_*_scores.npy`
- `groups/` — `group_{i}_msapos.npy` (sector positions in processed-MSA coordinates), one per IC
- `scarun.log` — run log
- `images/` — plots (conservation, SCA matrix, spectrum vs null, dendrogram, t-distributions, EV/IC scatter sweeps)

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

---

## sca-plots

Regenerate diagnostic plots from persisted results, without rerunning the pipeline. Each stage is opt-in via its own flag; at least one must be given.

### Usage

```bash
sca-plots [--prealign DIR] [--preprocessing DIR] [--scacore DIR] [--imgdir DIR] [-v N]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--prealign` | None | Prealign output directory (contains `filter_history.json`). Regenerates `plot_prealign_filter_history` |
| `--preprocessing` | None | Preprocessing output directory (contains `preprocessing_results.npz`, `filter_history.json`, `msa_binary2d_sp.npz`). Regenerates `plot_filter_history`, `plot_filter_distributions`, `plot_sequence_similarity` |
| `--scacore` | None | SCA core output directory (contains `scarun_results.npz`, `sca_eigendecomp.npz`, `sca_results/`). Regenerates `plot_dendrogram`, `plot_t_distributions`, and the EV/IC 2D/3D scatter sweeps |
| `--imgdir` | None | Output directory for all plots. When omitted, plots go into each stage's own `images/` subdirectory |
| `-v, --verbosity` | 1 | Verbosity level |

### Notes

The inline matplotlib figures currently in `run_sca.py::make_plots` (conservation, SCA-matrix imshow, spectrum vs null, sector-subset) are not replayed by this CLI — they will be picked up automatically once those plots are refactored into `mysca.pl`.
