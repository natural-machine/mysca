# 2026-04-21 — Entrypoint test fixes after fijab regularization bug fix

## Summary

Diagnosed and fixed 54 failing tests in `tests/test_entrypoint_scarun.py`. Root cause: commit `84023cd` corrected the `fijab` diagonal-block regularization (`lam/nsyms²` → `lam/nsyms`), but the expected test data for the entrypoint tests was generated before that fix. Updated the expected `fijab` and `Cijab_raw` arrays for all three MSAs; user then refactored the inline arrays into external `.txt` data files. Final state: all 352 tests pass in the main suite. A separate (still-open) question about `evals_sca` ordering for msa08 was investigated — it's an eigenvalue-degeneracy problem, not a correctness problem.

## Pre-session state

Session started at commit `30df2fd updated test data to pass entrypoint tests`. Despite the commit title, **54 tests were still failing** when the session began — the commit had externalized only some of the fijab arrays but left the inline expected values that still used the old `0.0012 = lam/nsyms²` regularization term.

To reach pre-session state:

```sh
git checkout 30df2fd
```

## What was wrong

For each i==j block of `fijab` (5 blocks per MSA, over 5×5 positions):
- **Off-diagonal symbol entries** (a ≠ b within i==j) were `lam / nsyms² = 0.0012`; should be `lam / nsyms = 0.006`.
- **Diagonal symbol entries** (a == b within i==j) were `(1-lam)·fia_unreg + lam/nsyms²`; should be `(1-lam)·fia_unreg + lam/nsyms`, i.e., add `0.0048`.
- Off-diagonal i≠j blocks were unchanged.
- `Cijab_raw = fijab - fia·fiaᵀ` gets a uniform `+0.0048` shift on i==j blocks (since `fia` was already correct).

All three MSAs have `naas=4, nsyms=5`, so the delta is the same `0.0048` everywhere.

## Changes made

1. **Arithmetic update of diagonal-block rows** in [tests/test_entrypoint_scarun.py](tests/test_entrypoint_scarun.py) for:
   - msa06 (argstring6a) `fijab` rows 0, 6, 12, 18, 24
   - msa06 `Cijab_raw` rows 0, 6, 12, 18, 24
   - msa07 (argstring7a) `fijab` rows 0, 6, 12, 18, 24
   - msa08 (argstring8a) `fijab` rows 0, 6, 12, 18, 24
   - msa08 `Cijab_raw` rows 0, 6, 12, 18, 24

   After these edits, `fijab` and `Cijab_raw` checks passed for all three MSAs. msa07 fully passed (its eigendecomp fixture was empty `{}`).

2. **Refactor to external data files** — user replaced the inline arrays with `np.genfromtxt(...)` reads from new `tests/_data/test_msa0{6,7,8}_{fijab,Cijab_raw,Cijab_sca}.txt` files. These files are now the source of truth for the expected test tensors, and should be regenerated from `run_sca` output whenever the algorithm changes.

## Post-session state

After the session's combined edits + the external-file refactor: full test suite passes (352 passed).

## Lingering issue — msa08 `evals_sca` ordering

The user later enabled the msa08 eigendecomp check and hit this:

```
Expected: [0.46532286  0.11290216  0.00371416  0.01944467  0.02699881]
Got:      [0.46532286  0.11290216  0.02699881  0.01944467  0.00371416]
```

**This is not a bug in the code.** Same set of 5 eigenvalues — only the tail three are in different order. Those three values all live in `[0.004, 0.027]` and are near-degenerate, which means:
- Their sort order is sensitive to floating-point noise.
- Their corresponding eigenvectors can rotate arbitrarily within the degenerate subspace and still be correct answers.

`Cij_corr` for msa08 has rows 0 and 1 structurally very similar (cols 3, 4 are literally equal: `[0.31974211, 0.17166678]`), so the matrix is near-rank-deficient. This is a property of the synthetic test MSA, not the SCA algorithm.

### Options discussed but not implemented

1. Reorder expected `evals_sca` to descending and permute `evecs_sca` columns accordingly. Least invasive but fragile — any future numerical noise can re-shuffle again.
2. Compare as sets (`np.allclose(np.sort(v), np.sort(v_exp))`) for `evals_sca` and compare eigenvector *spans/projectors* (`V·Vᵀ`) instead of column-by-column. Robust to rotation within degenerate subspaces.
3. Compare only the well-isolated top-k eigenpairs; skip the degenerate tail.
4. Leave the msa08 eigendecomp check commented out (as it was); rely on `test_core.py` for numerical correctness and have the entrypoint tests verify I/O plumbing only.

No option was chosen — user hasn't said how they want to handle it.

## Note on Cij_corr derivation (for future reference)

`sca_matrix` as saved by the entrypoint is `Cij_corr`, computed as:

```python
phi_ia = np.log((fia * (1 - qa)) / ((1 - fia) * qa))
Cijab_corr = phi_ia[:,None,:,None] * phi_ia[None,:,None,:] * Cijab_raw
Cij_corr = np.sqrt(np.sum(Cijab_corr**2, axis=(-1,-2)))  # Frobenius over (a,b)
```

`Cij_corr` is symmetric by construction (max asymmetry was `1.4e-17` — floating-point noise), so `np.linalg.eigh` is the right tool. During the session I initially tried to reproduce the entrypoint's `sca_matrix` by calling `run_sca` directly, but used the wrong preprocessing parameters (hardcoded rather than loaded from `argstring8a.txt`); that gave a different filtered MSA and an unrelated matrix. Lesson: reproduce via the actual entrypoint (`run_preprocessing.main` then `run_sca.main`), not by re-wiring `run_sca` with guessed args.

## Related commits

- `84023cd` — fixed fijab regularization bug (`lam/nsyms²` → `lam/nsyms` on diagonal). Predates this session.
- `30df2fd` — session-start commit; refactored some test data to external files but left broken inline fijab values.
- No new commits made during this session (edits are in working tree / were subsumed by user-side edits).
