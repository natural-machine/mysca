# mysca Codebase Overview

Implementation of Statistical Coupling Analysis (SCA) following:
**"Evolution-Based Functional Decomposition of Proteins"** — Rivoire et al., PLoS Comput Biol 12, e1004817 (2016).

---

## Project Structure

```
mysca/
├── src/mysca/          # Main source code
│   ├── __main__.py     # CLI router (3 entry points)
│   ├── core.py         # SCA algorithm
│   ├── preprocess.py   # MSA preprocessing
│   ├── run_preprocessing.py  # Preprocessing CLI pipeline
│   ├── run_sca.py      # SCA core CLI pipeline
│   ├── run_pymol.py    # PyMOL visualization CLI
│   ├── run_full_pipeline.py  # End-to-end orchestrator
│   ├── io.py           # File I/O (FASTA, PDB)
│   ├── mappings.py     # SymMap class (AA <-> int)
│   ├── helpers.py      # Utility functions
│   ├── constants.py    # AA list, background freqs, colors
│   ├── structures.py   # PDB structure utilities
│   ├── tools.py        # MSA conversion, filtering
│   └── pl/
│       └── plotting.py # Matplotlib visualizations
├── tests/              # Unit and integration tests
├── demo/SH3/           # Working example (SH3 domain)
├── notebooks/          # Jupyter notebooks (empty)
├── pyproject.toml      # Config, dependencies, entry points
├── setup.py
├── environment.yml     # Conda environment
└── README.md
```

---

## CLI Entry Points

Defined in `pyproject.toml` as console scripts, routed through `__main__.py`:

### 1. `sca-preprocess`
**Source:** `run_preprocessing.py`
**Purpose:** Load MSA, filter sequences/positions, compute sequence weights.

Key parameters:
- `-i` input MSA (FASTA)
- `-o` output directory
- `--gap_truncation_thresh` (tau, default 0.4) — remove columns with gap freq > tau
- `--sequence_gap_thresh` (gamma_seq, default 0.2) — remove rows with gap freq > gamma_seq
- `--reference` — optional reference sequence ID
- `--reference_similarity_thresh` (Delta, default 0.2) — min similarity to reference
- `--sequence_similarity_thresh` (delta, default 0.8) — clustering threshold for weights
- `--position_gap_thresh` (gamma_pos, default 0.2) — remove weighted-gap columns
- `--weight_method` — algorithm for weights (v3, v4, v5, v6, torch, gpu)

**Output:** `preprocessing_results.npz` containing msa, retained indices, weights, etc.

### 2. `sca-core`
**Source:** `run_sca.py`
**Purpose:** Run SCA analysis on preprocessed data.

Key parameters:
- `-i` preprocessing output directory
- `-o` output directory
- `--regularization` (lambda, default 0.03)
- `--n_boot` (default 10) — bootstrap iterations
- `--seed` — random seed
- `--kstar` — override number of significant components
- `--pstar` (default 95) — percentile for IC cutoff

**Output:** `scarun_results.npz`, eigendecomposition, ICA results, sector definitions, plots.

### 3. `sca-pymol`
**Source:** `run_pymol.py`
**Purpose:** Visualize sectors on 3D protein structures in PyMOL.

---

## Core Modules

### `preprocess.py` — MSA Preprocessing

Main function: `preprocess_msa()`

Pipeline steps:
1. Remove columns with gap frequency > tau
2. Remove sequences with gap frequency > gamma_seq
3. Filter by similarity to reference sequence (optional)
4. Compute sequence weights (round 1)
5. Remove columns with weighted gap frequency > gamma_pos
6. Recompute sequence weights (round 2)

**Sequence weighting:** `w[i] = 1 / |{j : similarity(i,j) >= delta}|`

Multiple implementations of `compute_weights()`:
| Version | Method                        | Notes          |
|---------|-------------------------------|----------------|
| v3      | Direct integer comparison     | Slow           |
| v4      | Sparse matrix, row-by-row     | Medium         |
| v5      | Sparse matrix, vectorized     | **Default**    |
| v6      | JAX-accelerated sparse ops    | GPU-capable    |
| torch   | PyTorch GPU computation       | Fastest (GPU)  |

Helper: `get_onehotmsa_sparse()` — converts integer MSA to sparse one-hot format.

### `core.py` — SCA Algorithm

Main function: `run_sca()`

Steps:
1. **Conservation** — positional amino acid frequencies and relative entropy:
   - `fi0`: gap frequency per position
   - `fia`: AA frequency per position (with regularization lambda)
   - `Di = sum_a [fia * log(fia/qa) + (1-fia) * log((1-fia)/(1-qa))]` (KL divergence)
   - `Dia`: per-AA conservation

2. **Pairwise covariance** — `fijab[i,j,a,b]` frequencies:
   - v1: standard numpy implementation
   - v2: JAX-accelerated implementation
   - Uses `iterblockpairs()` for memory-efficient block computation

3. **Corrected covariance matrix:**
   - `Cijab_raw = fijab - fia (x) fia`
   - `phi_ia = log((fia * (1-qa)) / ((1-fia) * qa))`
   - `Cijab_corr = phi_ia (x) phi_ia * Cijab_raw`
   - `Cij_corr = sqrt(sum_{a,b} Cijab_corr^2)` (Frobenius-like norm)

4. **Eigendecomposition** of Cij_corr (scipy.linalg.eigh)

5. **Bootstrap** — shuffle MSA columns, recompute eigenvalues, determine significance cutoff

6. **ICA** (Infomax algorithm on significant eigenvectors):
   - `W <- W + rho * (I + (1 - 2*sigmoid(y)) * y^T) * W`
   - Output: `V_ica = evecs @ W.T` (normalized independent components)

7. **Sector calling** — fit t-distributions to ICs, assign positions exceeding pstar percentile

### `io.py` — Input/Output

- `load_msa(fpath)` — load FASTA via Bio.SeqIO, returns (msa_matrix, seq_ids, SymMap)
- `load_pdb_structure(fpath)` — load PDB via Bio.PDB.PDBParser
- `get_residue_sequence_from_pdb_structure()` — extract AA sequence from structure
- `msa_from_aligned_seqs()` — create MSA matrix from aligned sequence strings

### `mappings.py` — Symbol Mapping

`SymMap` class:
- `aa_list`: 20 standard amino acids
- `gapsym`: gap character (default "-")
- `sym2int` / `aa2int`: dictionaries mapping symbols to integers
- `gapint`: integer index for gap
- `exclude_syms`: symbols to exclude (e.g. "X", "B", "Z")

`DEFAULT_MAP`: pre-built SymMap instance

### `helpers.py` — Utilities

Key functions:
- `get_top_k_conserved_retained_positions()` — find most conserved positions
- `map_msa_positions_to_sequence()` — map aligned positions to raw sequence coords
- `get_rawseq_indices_of_msa()` — raw sequence indices for MSA positions
- `get_conserved_rawseq_positions()` — map conserved positions to raw sequences
- `get_rawseq_positions_in_groups()` / `get_rawseq_scores_in_groups()` — group residues by sector
- `iterblocks(N, blocksize)` — memory-efficient block iterator
- `iterblockpairs(N, blocksize)` — pairwise block iterator

### `constants.py`

- `AA_STD20`: "ACDEFGHIKLMNPQRSTVWY"
- `DEFAULT_BACKGROUND_FREQ`: amino acid background frequencies (array of 20)
- `SECTOR_COLORS`: color palette for sector visualization

### `pl/plotting.py` — Visualization

- `plot_data_2d()` — scatter plot in 2D component space
- `plot_data_3d()` — 3D scatter plot
- `plot_dendrogram()` — hierarchical clustering
- `plot_sequence_similarity()` — pairwise identity heatmap
- `plot_t_distributions()` — histogram + t-distribution fits

---

## Data Flow

```
FASTA MSA file
     |
     v
[sca-preprocess]
  load_msa() -> integer matrix (N_seq x N_pos)
  preprocess_msa():
    filter columns (gap threshold)
    filter rows (gap threshold)
    filter by reference similarity
    compute_weights() -> sequence weights
    filter columns (weighted gap threshold)
    recompute_weights()
     |
     v
preprocessing_results.npz
  (msa, retained_sequences, retained_positions, sequence_weights, ...)
     |
     v
[sca-core]
  run_sca():
    compute conservation (fi0, fia, Di, Dia)
    compute pairwise covariance (fijab)
    compute corrected covariance (Cij_corr)
  eigendecomposition -> eigenvalues, eigenvectors
  bootstrap -> significance cutoff -> kstar
  run_ica() -> independent components (V_ica)
  t-distribution fitting -> sector assignments
     |
     v
scarun_results.npz + statsectors_msa.npz + statsectors_seq.npz + plots
     |
     v
[sca-pymol]  (optional)
  Visualize sectors on 3D protein structure
```

---

## Key Data Shapes

| Variable    | Shape                      | Description                          |
|-------------|----------------------------|--------------------------------------|
| msa         | (N_seq, N_pos)             | Integer-encoded MSA                  |
| xmsa        | (N_seq, N_pos, N_aa)       | One-hot encoded MSA                  |
| ws          | (N_seq,)                   | Sequence weights                     |
| fia         | (N_pos, N_aa)              | Position-AA frequencies              |
| fijab       | (N_pos, N_pos, N_aa, N_aa) | Pairwise position-AA frequencies     |
| Dia         | (N_pos, N_aa)              | Per-AA conservation                  |
| Di          | (N_pos,)                   | Total conservation per position      |
| Cij_corr    | (N_pos, N_pos)             | SCA correlation matrix               |
| evecs_sca   | (N_pos, K)                 | Eigenvectors                         |
| V_ica       | (N_pos, K)                 | Independent components               |

---

## Dependencies

**Core:** numpy, scipy, scikit-learn, pandas, matplotlib, seaborn, biopython, tqdm
**Optional accelerators:** jax, torch (GPU support)
**Optional visualization:** pymol-open-source
**Notebook support:** ipykernel, ipywidgets, imageio

---

## Test Suite

Located in `tests/`, with test data in `tests/_data/`:

| File                              | Tests                                    |
|-----------------------------------|------------------------------------------|
| test_core.py                      | fi0, fia, Dia, fijab calculations        |
| test_preprocess.py                | Preprocessing pipeline                   |
| test_io.py                        | load_msa, MSA I/O                        |
| test_helpers.py                   | Helper functions                         |
| test_structures.py                | PDB structure handling                   |
| test_entrypoint_preprocessing.py  | CLI preprocessing integration            |
| test_entrypoint_scarun.py         | CLI SCA core integration                 |
| test_sym_maps.py                  | Symbol mapping                           |

Test MSAs: msa05.faa, msa06.faa, msa07.faa (small test alignments)
Precomputed expected results for validation.

---

## Demo

`demo/SH3/` — working example with SH3 domain protein family:
- Input: `data/msas/SH3_demo_MSA_1.afa`
- Scripts: `step1_preprocessing.sh`, `step2_scacore.sh`
- Output: `out/preprocessing/`, `out/scacore/`
