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

## Known Issues (Resolved)

**uint8 overflow in sparse dot product** (fixed): The sparse one-hot matrix used by v4, v5, and v6 was originally created with `dtype=np.uint8`, which overflows at 255. For MSAs with more than 255 positions, pairwise similarity counts wrapped around silently, producing incorrect weights. Fixed by using `np.int16` (max 32,767), which is sufficient for any protein MSA. Methods v3 and gpu were unaffected as they use different computation paths.
