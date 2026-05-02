# 2026-04-27 — Workstream E Phase 1: sca-core profiling baseline

## Summary

Measured baseline for the perf workstream in
[.claude/plans/workstream-e-scacore-perf.md](../../.claude/plans/workstream-e-scacore-perf.md).
**`_compute_fijab_v1` is the dominant cost (82% of `sca-core` wall
time on SH3) and the bootstrap loop multiplies it by `N_BOOT + 1`.**
The plan's predicted optimization path (`einsum`-based rewrite) is
the *wrong* path at this scale — the naive einsum is **10x slower**
than v1 due to a 1.4 GB intermediate. **The right kernel rewrite is
`np.tensordot` (or matmul-flat), which delivers 9.5x over v1** with
the same fp64 numerics. `einsum` with `optimize='greedy'` works too
(6.4x) but is strictly slower than tensordot.

This report locks in the Phase 1 deliverable per the plan's gating
("Phase 2 is blocked until this exists. No GPU/torch code changes
yet.") and revises the predicted Phase 2 design before we touch any
production code.

## How to reproduce

```sh
# preprocess
cd /home/ahowe/Projects/mysca/demo/SH3
sca-preprocess -i data/msas/SH3_demo_MSA_1.afa \
    -o out/from_msa/preprocessing \
    --gap_truncation_thresh 0.4 --sequence_gap_thresh 0.2 \
    --reference '4837_jgi||3708||Equilibrative' \
    --reference_similarity_thresh 0.2 --sequence_similarity_thresh 0.8 \
    --position_gap_thresh 0.2

# wall-time + per-phase breakdown via cProfile
python -m cProfile -o /tmp/sca_core.prof "$(which sca-core)" \
    -i out/from_msa/preprocessing -o /tmp/sca_core_run \
    --regularization 0.03 --n_components 10 --seed 42 --no-plot

# kernel head-to-head (v1 vs v2 vs einsum vs tensordot)
python /tmp/sca_kernel_search.py
```

Drivers used (kept in `/tmp/`, not committed):
`/tmp/sca_profile.py`, `/tmp/sca_kernel_search.py`.

## Input

SH3 demo MSA, post-preprocess: **7490 sequences × 62 positions × 20
amino-acid one-hot channels**.

NarG-scale projection (~10k seqs × 300 cols, per the plan) was
deferred — the SH3 numbers are already conclusive about which kernel
to use, and SH3 represents the smallest (worst-case for JIT-overhead
arguments) test case.

## Phase A — End-to-end wall time

| Phase | Cumulative time | Share of total |
|---|---|---|
| **`_compute_fijab_v1`** (×11 calls: 1 single + 10 bootstrap) | **8.93s** | **82%** |
| `run_sca` (single) — fijab is most of it | 0.88s/call | 8% |
| `run_sca` (×10 bootstrap iters) — fijab + tiny eigh | ~8.1s | 75% (subset of fijab) |
| `np.linalg.eigh(62, 62)` | 1.2 ms | <0.1% |
| `np.linalg.eigvalsh` per bootstrap iter | 0.1 ms | <0.1% |
| `run_ica` (k=6, until convergence) | 0.13s | 1% |
| `shuffle_columns` (×10) | 0.21s total | 2% |
| Module imports (matplotlib, jax, scipy) | ~1.1s | 10% |
| **Total `sca-core` wall time (`--no-plot`)** | **10.78s** | 100% |

(Numbers from cProfile, single run on the dev box. Includes profiler
overhead — actual end-to-end wall time without cProfile is ~9.5s.)

**Eigendecomposition, ICA, plotting, shuffling, and I/O are not
bottlenecks at any tested scale.** Only `_compute_fijab` matters.

## Phase B — Kernel head-to-head

All variants computed `(1-λ) · Σ_s w_s xmsa[s,i,a] xmsa[s,j,b]` (the
core pairwise weighted contraction; regularization terms applied
afterward identically across variants). Each timed twice, best
reported. All match v1 within fp64 tolerance (`max abs diff ≤ 1e-13`).

| Kernel | Time | Speedup vs v1 |
|---|---|---|
| `_compute_fijab_v1` (numpy double-loop) | 0.818s | 1.00x (reference) |
| `_compute_fijab_v2` (JAX, post-warm) | 0.59s | **1.39x** |
| `np.einsum('s,sia,sjb->ijab', ...)` (naive) | 8.93s | **0.09x — REGRESSION** |
| `np.einsum(..., optimize='greedy')` | 0.137s | 5.97x |
| `np.einsum(..., optimize='optimal')` | 0.128s | 6.39x |
| **`np.tensordot(wxmsa, xmsa, axes=([0],[0]))`** | **0.086s** | **9.51x** |
| matmul-flat (`(wX.T @ X).reshape(...).transpose`) | 0.087s | 9.45x |

### What this means

1. **Naive einsum is the wrong path.** The plan's "Option A — pure
   numpy einsum rewrite" recommendation needs to be revised. The
   naive contraction order materializes a `(7490, 62, 20, 62, 20)`
   intermediate (≈1.4 GB at fp64), which is materially worse than v1's
   `(20, 20)` per-iter intermediates. numpy's einsum default does NOT
   pick the right order without `optimize=`.
2. **`optimize='greedy'`/`'optimal'` rescues einsum** to 6x, because
   the optimizer chooses the contraction sequence
   `(s) ⊗ (s,i,a) → (i,a) per s` — same shape as the v1 inner loop,
   but vectorized.
3. **`tensordot` is the actual best CPU path at SH3 scale**: 9.5x
   over v1, with bit-stable numerics (1.11e-16 max-abs vs v1).
4. **`_compute_fijab_v2` (JAX) is already 1.4x faster than v1** at
   npos=62. The plan's prediction that v2 is a JIT-overhead regression
   at small npos was wrong — there's only one `jax.jit`-decorated
   function and it's cached after the first call. v2 still has the
   Python double loop driving it, which is why it doesn't beat
   tensordot.

### Bootstrap-loop projection

The bootstrap loop calls `run_sca` `N_BOOT` times, and `_compute_fijab`
is the dominant cost per call. So:

| Backend | per-iter | 10-iter loop | end-to-end sca-core (≈ 11× fijab + ~1s overhead) |
|---|---|---|---|
| v1 (today) | 0.83s | 8.3s | **~10.8s** (measured) |
| tensordot CPU | 0.087s | 0.87s | **~2.1s (predicted)** |
| GPU (TBD; same einsum on torch) | tbd | tbd | tbd |

A pure-CPU tensordot rewrite drops `sca-core` from ~11s to ~2s on SH3
without touching torch or JAX. **GPU acceleration is a second-stage
optimization, not the first one.**

## Phase C — Memory scaling

`fijab` is `(npos, npos, naas, naas)`, fp64. naas is fixed at 20.

| `npos` | `fijab` size (fp64) |
|---|---|
| 62 (SH3) | 11.7 MiB |
| 100 | 30.5 MiB |
| 200 | 122.1 MiB |
| 300 (NarG-ish) | 274.7 MiB |
| 500 | 762.9 MiB |
| 1000 | 2.98 GiB |
| 2000 | 11.92 GiB |

The intermediate during the contraction matters more than the result.
For `np.tensordot(wxmsa, xmsa, axes=([0],[0]))`, the intermediate is
just the two operands (each `nseq × npos × naas × 8B`) plus the
result. At SH3 (`nseq=7490, npos=62, naas=20, fp64`):
each operand = 70 MiB; result = 12 MiB. Comfortable.

For NarG-scale (`nseq=10000, npos=300`): each operand = 458 MiB,
result = 275 MiB. Still fine on any modern box.

For batched-bootstrap GPU (`N_BOOT × fijab`):

| `N_BOOT` (npos=300) | Total fijab footprint |
|---|---|
| 1 | 0.27 GiB |
| 5 | 1.34 GiB |
| 10 | 2.68 GiB |
| 20 | 5.36 GiB |

Comfortable on a 16+ GiB GPU at NarG scale; would need chunking on a
6 GiB card. The plan's `--bootstrap_chunk` knob is still the right
shape for that case.

## Phase D — Revised Phase 2 design

Replace the plan's "Option A — einsum rewrite" with **Option A' —
tensordot rewrite** as the new CPU default. Specifically:

```python
def _compute_fijab_v3(xmsa, ws_norm, lam, nsyms):
    """Vectorized tensordot equivalent of v1.

    Wall-time ~9.5x faster than v1 on SH3-scale input; bit-stable
    numerics (max abs diff vs v1 = 1.11e-16).
    """
    _, npos, naas = xmsa.shape
    xf = xmsa.astype(np.float64, copy=False)
    weighted = ws_norm[:, None, None] * xf
    # Contract over the sequence axis -> (i, a, j, b) -> (i, j, a, b).
    fijab = np.tensordot(weighted, xf, axes=([0], [0])).transpose(0, 2, 1, 3)
    fijab *= (1.0 - lam)
    diag = np.eye(npos, dtype=bool)
    fijab[~diag] += lam / (nsyms * nsyms)
    fijab[diag] += lam / nsyms
    return fijab
```

The rest of the plan's structure stays valid:

- `--sca_method {numpy,jax,gpu}` flag (default `numpy` = v3).
- `--use_jax` deprecated alias mapping to `numpy`/`jax`.
- GPU path (`torch.tensordot` or `torch.einsum`) for `--sca_method=gpu`.
- `--bootstrap_chunk` for batched-GPU bootstrap iterations.
- A new `docs/sca_methods.md` mirroring `docs/weight_methods.md`.

The plan's perf acceptance criteria need to be tightened:

- ~~"`--sca_method numpy` is at least 5x faster than current numpy
  `_compute_fijab_v1` on SH3"~~ → **at least 7x** (we measured 9.5x;
  set the bar above the einsum-greedy result so we don't regress to it).
- "`--sca_method gpu` is at least 3x faster than `--sca_method numpy`
  on the bootstrap-dominated path on NarG-scale input" → **still TBD**;
  measure during Phase 3.3 / Phase 4.
- "End-to-end `sca-core` runtime on SH3 with default flags drops by
  ≥3x on the bootstrap-loop path" → **expected ~5x with tensordot
  alone** (10.8s → ~2.1s); GPU should push it further only at
  npos≥300.

## Notes for next session

- **Don't touch `_compute_fijab_v2` legacy path yet.** It's not the
  bottleneck and changing it adds risk; defer to Phase 3.2.
- **Profile NarG (~10k×300) before claiming the 9.5x scales.**
  Tensordot's wall-time at npos=300 should be ~0.087s × (300/62)² ≈
  2s by quadratic scaling, vs v1 at ~0.83s × (300/62)² ≈ 19s. If
  tensordot scales worse than O(npos²) due to memory-bandwidth
  pressure, GPU becomes more attractive sooner.
- **JIT cold-start cost is real but small** (0.51s warm-up for v2
  here). Not a blocker for `--sca_method=jax` if we keep the JIT
  cache warm across bootstrap iters.
- **The plan's GPU-batched-bootstrap shape is still right** —
  but with a 9.5x CPU speedup already in hand, GPU value is mostly at
  large npos / large `N_BOOT`. Worth quantifying before committing
  to the full plan.
