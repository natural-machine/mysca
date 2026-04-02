# pySCA Frequency Regularization Inconsistency

## Summary

pySCA's pipeline uses two different background frequency distributions for
regularization within the same SCA computation. This appears to be accidental
rather than intentional — the two functions (`posWeights` and `scaMat`) were
developed independently with different defaults, and `scaMat` does not forward
its `freq0` when calling `posWeights`.

## Details

### `scaMat` (SCA matrix computation)

Default: `freq0 = np.ones(20) / 21` (uniform across 20 amino acids).

This uniform background is used to regularize both the single-site frequencies
(`freq1`) and pairwise frequencies (`freq2`) that form the covariance term
`freq2 - outer(freq1, freq1)`.

### `posWeights` (conservation and position weights)

Default: `freq0 = [0.073, 0.025, 0.050, ...]` (UniProt amino acid frequencies).

This is used to regularize the `freq1` from which Dia, Di, and the position
weights Wia are derived.

### Where the inconsistency occurs

In `scaMat` (scaTools.py, line ~1987):

```python
freq1, freq2, freq0 = freq(alg, ..., freq0=np.ones(20)/21)   # uniform
Wpos = posWeights(alg, seqw, lbda)[0]                         # UniProt (freq0 not passed)
tildeC = np.outer(Wpos, Wpos) * (freq2 - np.outer(freq1, freq1))
```

`posWeights` is called without a `freq0` argument, so it falls back to its own
default (UniProt). This means `Wpos` is derived from frequencies regularized
with UniProt, while the covariance it multiplies uses frequencies regularized
with uniform 1/21.

## How mysca handles this

In `core.py`, we maintain two sets of regularized single-site frequencies:

- `fia`: regularized with `lam / nsyms` (uniform, matching `scaMat`). Used for
  the covariance `fijab - outer(fia, fia)`.
- `fia_pw`: regularized with `lam * qa` (UniProt, matching `posWeights`). Used
  for Dia, Di, and `phi_ia` (the position weights).

This replicates pySCA's behaviour so that results are comparable.
