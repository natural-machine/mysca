# Sequence Weight Methods

## Overview

Sequence weights correct for phylogenetic bias in the input MSA. Closely related sequences are down-weighted so that clusters of near-identical sequences don't dominate the SCA statistics. For each sequence, the weight is `1 / |neighbors|`, where a neighbor is any sequence with identity >= the `--sequence_similarity_thresh` (delta, default 0.8).

Weights are computed during `sca-preprocess` and saved with the preprocessing results. The `sca-core` step loads and uses these weights without recomputation.

## Available Methods

| Version | Status | CLI | Tested | Description |
|---------|--------|-----|--------|-------------|
| v1 | Disabled | No | No | Buggy, raises error |
| v2 | Disabled | No | No | Buggy, raises error |
| v3 | Active | Yes | Yes | Direct element-wise comparison in NumPy |
| v4 | Active | Yes | Yes | Sparse one-hot matrix, blockwise dot product |
| v5 | Active | Yes (default) | Yes | Optimized sparse CSR operations |
| v6 | Active | No | Yes | v5 + JAX JIT compilation |
| gpu | Active | Yes | Yes | PyTorch with automatic device detection |

## Selection via `--accelerator`

`sca-preprocess` exposes a global `--accelerator {none, gpu}` flag (default `none`). When unset, `--weight_method` defaults to `sparse`. When `--accelerator gpu`, `--weight_method` auto-defaults to `gpu`. An explicit `--weight_method` always wins over `--accelerator`. The same flag and resolution logic exist on `sca-core` for the frequency-tensor kernel (`--freq_method`).

## Method Details

### v3 — Direct comparison

Compares sequences element-wise using dense integer arrays. Gaps are excluded from the similarity count. Simple and correct, but slow for large MSAs due to explicit pairwise iteration.

### v4 — Sparse blockwise

Converts the integer MSA to a sparse one-hot binary matrix, then computes pairwise similarity via dot products in blocks (controlled by `--block_size`). Faster than v3 for moderately sized MSAs.

### v5 — Optimized sparse (default)

Builds on v4 with direct CSR data structure operations for counting row similarities, avoiding materializing full dense blocks. This is the default and recommended method for most use cases.

### v6 — JAX JIT

Extends v5 by JIT-compiling the row-counting step with JAX. Available in the codebase and passes unit tests, but is not currently exposed via the CLI.

### gpu — PyTorch

Uses PyTorch tensors with automatic device detection (CUDA, MPS, XPU). Falls back to v5 if no GPU is available. Best for very large MSAs where GPU memory is sufficient.

## Recommendations

- **Default (v5)** works well for most protein family MSAs.
- **gpu** is recommended for large MSAs (thousands of sequences with hundreds of positions) when a GPU is available.
- **v3** is the simplest implementation and useful as a reference, but significantly slower.

## Size Limits

The sparse methods (v4, v5, v6) size their integer types to the MSA, so no alignment size produces wrapped values. When an MSA is too large for a default type, a wider one is used and a WARNING is logged naming the limit that was exceeded. The sizes that matter are those of the MSA after the filtering steps that precede weighting.

| Quantity | Default type | Exceeded when | Fallback |
|----------|--------------|---------------|----------|
| Pairwise match counts | `int16` | more than 32,767 positions | `int32` |
| Sparse matrix indices | `int32` | sequences x positions above 2,147,483,647 (e.g. 3 million sequences of 1,000 positions) | `int64` |

The wider types take more memory: 4 instead of 2 bytes per stored match count, and 8 instead of 4 bytes per index.

All methods accumulate neighbor counts in 32-bit integers. An MSA with more than 2,147,483,647 sequences is rejected with an error rather than miscounted.

## Known Issues (Resolved)

**uint8 overflow in sparse dot product** (fixed): The sparse one-hot matrix used by v4, v5, and v6 was originally created with `dtype=np.uint8`, which overflows at 255. For MSAs with more than 255 positions, pairwise similarity counts wrapped around silently, producing incorrect weights. Fixed by using `np.int16` (max 32,767); see the int16 entry below for wider alignments. Methods v3 and gpu were unaffected as they use different computation paths.

**uint16 overflow in sparse one-hot indices** (fixed): The row and column indices of the one-hot matrix were built as `np.uint16`. Rows wrapped for MSAs with more than 65,536 sequences, leaving later sequences with a weight of 1.0, and columns wrapped for MSAs with more than 3,121 positions, adding spurious matches between unrelated positions. Both silently produced incorrect weights. Fixed by building the CSR arrays directly with `int32` indices. Methods v3 and gpu were unaffected.

**int16 overflow in sparse dot product** (fixed): With `np.int16` data, pairwise match counts wrapped to negative values for MSAs with more than 32,767 positions, so similar sequences stopped counting as neighbors and weights were too high. Fixed by choosing the data type from the number of positions (see Size Limits). Methods v3 and gpu were unaffected.
