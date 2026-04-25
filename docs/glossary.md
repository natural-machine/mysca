# Glossary

This document establishes the vocabulary used across the mysca documentation.

## Concepts and objects

| Concept | Object | Coord space |
|---|---|---|
| **Unaligned sequence set** | $M_{\text{orig}}$ sequences of variable length | — |
| **Unprocessed MSA** | $M_{\text{orig}} \times L_{\text{orig}}$ aligned MSA | original-MSA cols |
| **Processed MSA** | $M_{\text{proc}} \times L_{\text{proc}}$ filtered MSA | processed-MSA cols |
| **Retained positions** | set of $L_{\text{proc}}$ indices in $\{0,\ldots,L_{\text{orig}}-1\}$ | processed-MSA col → original-MSA col |
| **Retained sequences** | set of $M_{\text{proc}}$ indices in $\{0,\ldots,M_{\text{orig}}-1\}$ | processed-MSA row → original-MSA row |
| **SCA correlation matrix** | $L_{\text{proc}} \times L_{\text{proc}}$ conservation-weighted covariance matrix | processed-MSA col × processed-MSA col |
| **SCA eigenvector / eigenmode** | One column of the eigenvector basis (length $L_{\text{proc}}$) | processed-MSA cols, real-valued loadings |
| **Independent component** | One ICA-rotated mode (length $L_{\text{proc}}$) | processed-MSA cols, real-valued loadings |
| **(High-load) IC positions** | Subset of $\{0, \dots, L_{\text{proc}} - 1\}$; positions clearing a threshold on the IC's loadings | processed-MSA cols (canonical); also persisted in original-MSA cols |
| **Target sequence / structure** | Variable length-$L$ sequence $(r_0, \dots, r_{L-1})$; optionally with associated PDB structure | residue indices |
| **Per-target IC residues** | Subset of $\{0, \dots, L - 1\}$ — the residues of a specific target corresponding to an IC | residue indices |

## Coordinate-space discipline

Every "position" or "index" lives in one of three spaces:

| Space | Indexes |
|---|---|
| **Original-MSA col** | columns of the unprocessed MSA (length $L_{\text{orig}}$) |
| **Processed-MSA col** | columns of the filtered MSA (length $L_{\text{proc}}$); SCA runs here |
| **Target residues** | residues of a specific target — raw-seq index, or PDB residue ID |

Mapping between these spaces can be a source of confusion. "Retained positions" refers to those positions of the original MSA that are retained in the processing steps.

- `original_col = retained_positions[processed_col]`
- For an out-of-sample target, an aligner produces a map from original MSA column to raw-residue index (or `-1` if the target has a gap at that column), giving a target_residue for each processed-MSA col.
- For a PDB target, the structure loader records each residue's PDB residue ID (chain + number), giving a pdb_residue for each residue of the target's input sequence.

The target → processed-MSA chain is always two steps: a column-preserving aligner (e.g. `mafft --add --keeplength` or `hmmalign --outformat afa` with insert columns stripped) first produces a length-$L_{\text{orig}}$ alignment, and `retained_positions` then filters down to $L_{\text{proc}}$.

## Terms used in this document

- **Component / IC** — a mode of the SCA correlation structure: an eigenvector or, after the ICA rotation, an independent component. We use **IC** when the rotation has been applied, and **component** as an umbrella term.
- **Significant IC** — an IC whose eigenvalue / weight clears the bootstrap null distribution. The number of significant ICs is `kstar`.
- **(High-load) IC positions** — the subset of processed-MSA cols on which an IC's loading clears a per-IC defined cutoff (e.g. a given percentile of an empirical $t$-distribution).
- **Per-target IC residues** — the subset of a target's residues corresponding to an IC. The same statistical structure as high-load IC positions, but mapped to a specific sequence or PDB structure (with target gaps dropped).
- **Sector** — biological term of art for a co-evolving residue group acting as a functional/evolutionary unit.

