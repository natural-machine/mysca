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

- `scarun_results.npz` — core SCA results (`Dia`, `conservation`, `sca_matrix`, `phi_ia`, `fi0`, `fia`, `Cij_raw`; optionally `Cijab_raw`, `fijab` with `--save_all`)
- `sca_eigendecomp.npz` — full + significant eigenvalues/eigenvectors
- `scarun_args.json` — arguments used
- `statsectors_msa.npz` / `statsectors_seq.npz` — per-sequence sector mappings in processed-MSA and raw-sequence coordinates; only the top-`kstar` IC groups are expanded per sequence
- `sca_results/` — `v_ica_normalized.npy`, `w_ica.npy`, `t_dists_info.json`, `evals_shuff.npy`, `sca_matrix_sector_subset.npy`, scalar text files (`kstar.txt`, `n_components.txt`, etc.), `msa_sectors/sector_*_msapos.npy` + `sector_*_scores.npy`
- `groups/` — `group_{i}_msapos.npy` (sector positions in processed-MSA coordinates), one per IC
- `scarun.log` — run log
- `images/` — plots (conservation, SCA matrix, spectrum vs null, dendrogram, t-distributions, EV/IC scatter sweeps)

---

## sca-pymol

Render SCA sectors on 3D protein structures using PyMOL. Consumes `sca-structure` output directly — the per-structure `ic_pdb_residues` list carries authoritative PDB residue numbers (via `PDBStructure.residue_ids`), and `pdb_path` tells `sca-pymol` which file to load. Protein-specific annotations (cofactors, iron-sulfur clusters, ligands, etc.) are supplied by a user Python file via `--features_py`.

Requires the optional `pymol-open-source` dependency (`conda install -c conda-forge pymol-open-source`).

### Usage

```bash
sca-pymol --structure <structure-out-dir> \
    [--structure_id ID] \
    [--groups G [G ...]] \
    [--multisector] \
    [-r REF_STRUCTURE_ID] \
    [--features_py PATH] [--features NAME[,NAME...]] \
    [--views] [--animate] [--nframes N] [--duration SEC] \
    [--spin_axis {x,y,z}] [--spin_degrees N] \
    [--ray {none,first,all}] [--dpi N] \
    -o <outdir> [-v N]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--structure` | `sca-structure` output directory containing `structure_projection.json` |
| `-o, --outdir` | Output directory for rendered images |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--structure_id` | None | Specific `structure_id` to render when the json has more than one entry. Default: render every entry |
| `-r, --reference` | None | Another `structure_id` from the same json to align the target against via `cmd.align` |
| `--groups` | all | IC group indices (0-based) to render. Default: every group present in the projection |
| `--multisector` | off | Render all selected groups on a single frame per structure instead of one frame per group |
| `--features_py` | None | Path to a user Python file supplying protein-specific annotation functions |
| `--features` | None | Comma-separated names of callables in `--features_py` to invoke per render pass. Requires `--features_py` |
| `--views` | off | Save four rotated side views per frame under `outdir/views/` |
| `--animate` | off | Save a rotating GIF per rendered frame under `outdir/` — one per IC group in the default mode; one covering all selected groups under `--multisector` |
| `--nframes` | 24 | Animation frame count (only used with `--animate`) |
| `--duration` | 2.4 | Animation duration in seconds (only used with `--animate`) |
| `--spin_axis` | `y` | Rotation axis for `--animate` (choices: `x`, `y`, `z`) |
| `--spin_degrees` | 360 | Total rotation in degrees over `--nframes`. Set to e.g. 180 for a half-spin, 90 for a quarter-turn |
| `--ray` | `all` | Ray-tracing policy for animation frames: `all` (every frame, best quality, slowest), `first` (only frame 0), `none` (disabled, fastest) |
| `--dpi` | 300 | DPI for all rendered PNGs (stills, views, and animation frames) |
| `-v, --verbosity` | 1 | Verbosity level |

### Output

Writes to the specified output directory:

- `<structure_id>_group<N>.png` — one PNG per (structure, IC group) pair under the default rendering mode
- `<structure_id>_groups_<idxs>.png` — one PNG per structure under `--multisector`
- `views/` — four rotated views per frame when `--views` is passed
- `frames/<basename>_frames/` — raw per-frame PNGs from animation, where `<basename>` is `<structure_id>_group<N>` by default or `<structure_id>_groups_<idxs>` under `--multisector`
- `<basename>.gif` — rotating animation when `--animate` is passed (one per IC group by default; one covering all groups under `--multisector`)
- `pymol.log` — run log

### Features plugin

Protein-specific annotations are supplied by a user Python file. Each
function must match this signature:

```python
def feature_fn(struct, cmd, *, color=None, context=None) -> None:
    ...
```

- `struct` — PyMOL object name (always `"struct"` in the current implementation).
- `cmd` — PyMOL's `cmd` module, injected so the file does not need `from pymol import cmd`.
- `color` — optional per-feature color (currently always `None`; plumbed for a future flag).
- `context` — dict with `projection`, `scaffold`, `group_idx`, `outdir`. Read `projection["chain_id"]`, `projection["ic_pdb_residues"]`, `projection["pdb_path"]`, etc. as needed.

Ship-ready example: [`demo/pymol_features/narg_1q16.py`](../demo/pymol_features/narg_1q16.py) ports the previously-hardcoded molybdenum / [4Fe-4S] / MGD cofactor selections for 1Q16 NarG:

```python
def show_molybdenum(struct, cmd, *, color=None, context=None):
    cmd.select("mo", f"{struct}/F/A/6MO`1302/MO")
    cmd.show("everything", "mo")
    if isinstance(color, str):
        cmd.color(color, "mo")
```

Invoke via:

```bash
sca-pymol --structure out/structure --structure_id NarG_1Q16 \
    --features_py demo/pymol_features/narg_1q16.py \
    --features show_molybdenum,show_sf4_cluster,show_mgd \
    -o out/pymol
```

Loader errors surface at CLI startup (before any rendering): missing file → `FileNotFoundError`, missing attribute → `ValueError`, non-callable attribute → `TypeError`.

### Animation

`--animate` writes a rotating GIF (default 24 frames over 2.4 s, Y-axis spin) for each rendered frame — one per IC group in the default mode, or a single combined rotation under `--multisector`. Frames are also written to `outdir/frames/<basename>_frames/` for inspection or re-encoding.

```bash
# One GIF covering the top two ICs lit up together.
sca-pymol --structure out/structure \
    --groups 0 1 --multisector --animate \
    -o out/pymol_anim

# Per-group GIFs (one per IC), longer and smoother.
sca-pymol --structure out/structure \
    --groups 0 1 2 --animate --nframes 36 --duration 3.6 \
    -o out/pymol_anim

# Fast preview: X-axis half-swing, no ray-tracing, lower dpi.
sca-pymol --structure out/structure \
    --groups 0 --animate \
    --spin_axis x --spin_degrees 180 --ray none --dpi 150 \
    -o out/pymol_preview
```

`--ray all` ray-traces every frame (today's default — best quality, slowest). `--ray first` only rays frame 0 (viewport for the rest — mixed look but ~10× faster). `--ray none` disables ray-tracing entirely for previews.

Requires `imageio` + `Pillow` (both ship as deps of `pymol-open-source` on conda-forge; on a minimal env install via `pip install imageio pillow`).

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

`plot_covariance_matrix` depends on `Cij_raw`, which is persisted in `scarun_results.npz` as of this version of sca-core. Directories produced by older `sca-core` runs will skip `covariance_matrix.png` and log a note; rerun `sca-core` to refresh them.

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
# Single PDB file:
sca-structure -s <pdb_path> [--chain A] [--seq_id <id>] \
    --preprocessing <preprocess-dir> \
    --scacore <scacore-dir> \
    -o <output-dir> [options]

# Batch via a sequence-to-PDB TSV:
sca-structure --seq_map <tsv> \
    --preprocessing <preprocess-dir> \
    --scacore <scacore-dir> \
    -o <output-dir> [options]

# Batch via UniProt → PDB resolution (SIFTS best_structures):
sca-structure --uniprot_ids P06241 P12931 \
    --pdb_dir <dir-of-pre-downloaded-pdbs> \
    [--cache_dir <dir>] \
    --preprocessing <preprocess-dir> \
    --scacore <scacore-dir> \
    -o <output-dir> [options]
```

Exactly one of `-s/--structure`, `--seq_map`, or `--uniprot_ids` is required.

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--preprocessing` | `sca-preprocess` output directory |
| `--scacore` | `sca-core` output directory |
| `-o, --outdir` | Output directory |
| One of: `-s/--structure`, `--seq_map`, `--uniprot_ids` | Single PDB, TSV map, or UniProt accession list (SIFTS-resolved). `--uniprot_ids` additionally requires `--pdb_dir` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--chain` | first chain | Chain ID within `-s/--structure` |
| `--pdb_dir` | None | Directory of pre-downloaded PDB files. Required with `--uniprot_ids`; SIFTS resolves accessions but does not fetch structures |
| `--cache_dir` | `./.sifts_cache` | Local directory to cache SIFTS JSON responses. Only consulted in `--uniprot_ids` mode |
| `--seq_id` | None | Header used when projecting `-s/--structure`'s sequence. When it matches an ID in the reference MSA, the project step takes the in-sample short-circuit. Ignored in `--seq_map` / `--uniprot_ids` modes (seq IDs come from the map / UniProt list itself) |
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

### SIFTS lookup

`--uniprot_ids` resolves each UniProt accession to its top-ranked PDB structure via EBI PDBe's [`mappings/best_structures`](https://www.ebi.ac.uk/pdbe/api/doc/sifts.html) endpoint. Responses are cached under `--cache_dir` (default `./.sifts_cache/`) so repeat runs don't re-hit the network. SIFTS returns lowercase PDB IDs; the lookup tries both `{id}.pdb` and `{ID}.pdb` inside `--pdb_dir`, so either RCSB-style or lowercase filenames work.

The PDB files must already exist in `--pdb_dir`; SIFTS only resolves IDs, it does not fetch structures.

The same mechanism is available at the library level for programmatic users:

```python
from mysca.structure import SequencePdbMap
seq_map = SequencePdbMap.from_sifts_for_uniprot_ids(
    ["P00742", "P09211", "P02768"],
    pdb_dir="./pdbs",
    cache_dir="./.sifts_cache",  # optional; default ./.sifts_cache
)
# seq_map["P00742"].pdb_path → "./pdbs/1c4v.pdb" (whichever case exists)
```

Missing files raise `FileNotFoundError` under `strict=True` (the default) or are logged and skipped under `strict=False`.
