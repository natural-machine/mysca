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
| `--preprocessing` | None | Preprocessing output directory (contains `preprocessing_results.npz`, `filter_history.json`, `msa_binary2d_sp.npz`). Regenerates `plot_filter_history`, `plot_filter_distributions`, `plot_sequence_similarity`. When passed alongside `--scacore`, also enables the positional conservation plots (they need `retained_positions` + the original MSA length) |
| `--scacore` | None | SCA core output directory. Regenerates `plot_conservation`, `plot_sca_matrix`, `plot_sca_spectrum`, `plot_sca_spectrum_vs_null`, `plot_dendrogram`, `plot_t_distributions`, `plot_data_2d`/`3d` (EV + IC sweeps), `plot_sca_matrix_sector_subset`. With `--preprocessing` also given, adds `plot_conservation_top` and `plot_conservation_positional` |
| `--imgdir` | None | Output directory for all plots. When omitted, plots go into each stage's own `images/` subdirectory |
| `-v, --verbosity` | 1 | Verbosity level |

### Notes

`plot_covariance_matrix` is not replayed: the raw (pre-weighting) covariance matrix `Cij_raw` is computed in-memory during `sca-core` and not persisted to disk. Rerun `sca-core` to produce that figure.

---

## sca-project

Project primary amino-acid sequences (in- or out-of-sample) onto an existing SCA result. For each input sequence, map its raw residues onto the IC groups from the source SCA run.

### Usage

```bash
sca-project -i <sequences.fasta> \
    --preprocessing <preprocess-dir> \
    --scacore <scacore-dir> \
    -o <output-dir> [options]
```

Records whose ID is already present in the reference MSA (under `--preprocessing`) are resolved in-sample (no external alignment). Other records are aligned onto the reference via the chosen aligner.

### Required Arguments

| Argument | Description |
|----------|-------------|
| `-i, --input_fpath` | Path to an input FASTA of sequences to project |
| `--preprocessing` | `sca-preprocess` output directory (must include `msa_orig.fasta-aln`) |
| `--scacore` | `sca-core` output directory (must include `sca_results/msa_sectors/sector_*_msapos.npy` and `sca_results/v_ica_normalized.npy`) |
| `-o, --outdir` | Output directory |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--aligner` | `mafft_add` | Out-of-sample alignment method. `mafft_add` uses `mafft --add --keeplength`. `hmmalign` builds a profile HMM (`hmmbuild --hand --amino`) with every reference column as a match state, then aligns new sequences (`hmmalign --outformat afa`) and keeps only match columns. In-sample records bypass alignment entirely |
| `--align_bin` | None | Explicit path to the alignment binary (default: resolve from PATH). For `--aligner hmmalign` this is `hmmalign`; `hmmbuild` is resolved from PATH |
| `--align_threads` | 1 | Threads for the alignment tool (unused by `hmmalign`) |
| `-v, --verbosity` | 1 | Verbosity level |

### Output

Writes to the specified output directory:

- `projection.json` — top-level result: per-sequence dicts containing `seq_id`, `raw_sequence`, `aligned_sequence`, `residue_by_processed_col` (length `L_proc`), `ic_memberships` (per-IC raw residue indices), `ic_loadings`, `ic_processed_cols`, `in_sample`
- `per_sequence/<seqid>_residues.tsv` — one row per (IC, residue) for readable inspection
- `projection_args.json` — arguments used
- `projection.log` — run log

### External Binaries

`mafft` (for the default `mafft_add` aligner) must be resolvable via PATH or via `--align_bin`. For `--aligner hmmalign`, both `hmmbuild` and `hmmalign` must be on PATH (install via `conda install -c bioconda hmmer`). In-sample projection does not invoke any external binary.

---

## sca-structure

Project PDB structure(s) onto an existing SCA result. Wraps `sca-project`: the PDB's primary sequence is run through the standard project pipeline, then IC-group memberships are translated from raw residue indices into PDB residue numbers.

### Usage

```bash
sca-structure -s <pdb_path> [--chain A] [--seq_id <id>] \
    --preprocessing <preprocess-dir> \
    --scacore <scacore-dir> \
    -o <output-dir> [options]

# Or iterate over a sequence-to-PDB map:
sca-structure --seq_map <tsv> \
    --preprocessing <preprocess-dir> \
    --scacore <scacore-dir> \
    -o <output-dir> [options]
```

Exactly one of `-s/--structure` or `--seq_map` is required.

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--preprocessing` | `sca-preprocess` output directory |
| `--scacore` | `sca-core` output directory |
| `-o, --outdir` | Output directory |
| `-s, --structure` OR `--seq_map` | Either a single PDB path, or a TSV mapping MSA sequence IDs to PDB paths (format below) |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--chain` | first chain | Chain ID within `-s/--structure` |
| `--seq_id` | None | Header used when projecting `-s/--structure`'s sequence. When it matches an ID in the reference MSA, the project step takes the in-sample short-circuit. Ignored when `--seq_map` is used (the TSV keys are used instead) |
| `--aligner` | `mafft_add` | Out-of-sample alignment method (inherits from `sca-project`) |
| `--align_bin` | None | Explicit path to the alignment binary |
| `--align_threads` | 1 | Threads for the alignment tool |
| `-v, --verbosity` | 1 | Verbosity level |

### `--seq_map` TSV format

Two or three tab-separated columns per row:

```text
seq_id<TAB>pdb_path[<TAB>chain]
```

Lines starting with `#` and blank lines are ignored. Relative `pdb_path` entries resolve relative to the TSV's directory.

### Output

Writes to the specified output directory:

- `structure_projection.json` — list of per-structure dicts. Each includes `structure_id`, `chain_id`, the full raw-residue-coordinate `sequence_projection` (as per `sca-project`), and `ic_pdb_residues` (per-IC list of PDB residue numbers)
- `per_structure/<structure_id>_ic_residues.tsv` — one row per (IC, residue) including both raw residue index and PDB residue number
- `structure_args.json` — arguments used
- `structure.log` — run log

### Lookup extensions (planned)

`SequencePdbMap.from_sifts_for_uniprot_ids()` is registered in `mysca.structure.mapping` but currently raises `NotImplementedError`. SIFTS on-demand lookup will land in a follow-up without changing the public CLI surface.
