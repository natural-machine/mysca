"""Gap-position invariance tests.

Verify that the one-hot helpers and sparse encoders in preprocess.py produce
outputs that are consistent across arbitrary placements of the gap symbol in
``SymMap``. The dense helper ``onehot_without_gap`` should yield an output
whose columns match ``aa_list`` order regardless of ``gapint``, and the
sparse encoders should agree between gap positions up to a permutation of
per-position column blocks.
"""

import numpy as np
import pytest

from mysca.mappings import SymMap
from mysca.preprocess import (
    onehot_without_gap,
    get_onehotmsa_sparse,
    get_onehotmsa_sparse_nogap,
)


AA_SYMS = "ABCDE"
GAPSYM = "-"
NUM_AAS = len(AA_SYMS)


def _encode_msa_strings(seqs, mapping):
    return np.array(
        [[mapping[c] for c in s] for s in seqs], dtype=np.int_
    )


MSA_STRINGS = [
    "A-CDE",
    "BCDE-",
    "--CDA",
    "ABCDE",
]


@pytest.mark.parametrize("gap_value", [0, 1, 2, NUM_AAS])
def test_onehot_without_gap_matches_aa_list_order(gap_value):
    """``onehot_without_gap`` column i should indicate aa_list[i]."""
    mapping = SymMap(AA_SYMS, GAPSYM, gap_value=gap_value)
    msa = _encode_msa_strings(MSA_STRINGS, mapping)
    xmsa = onehot_without_gap(msa, len(mapping), mapping.gapint)

    assert xmsa.shape == (len(MSA_STRINGS), len(MSA_STRINGS[0]), NUM_AAS)

    for seq_idx, seq in enumerate(MSA_STRINGS):
        for pos_idx, sym in enumerate(seq):
            row = xmsa[seq_idx, pos_idx]
            if sym == GAPSYM:
                assert not row.any(), (
                    f"gap at ({seq_idx},{pos_idx}) should have all-zero row"
                )
            else:
                aa_idx = mapping.aa_list.index(sym)
                assert row[aa_idx], (
                    f"aa {sym} at ({seq_idx},{pos_idx}) should set column "
                    f"{aa_idx}; got {row}"
                )
                assert row.sum() == 1


@pytest.mark.parametrize("gap_value", [0, 1, 2, NUM_AAS])
def test_sparse_nogap_matches_dense(gap_value):
    mapping = SymMap(AA_SYMS, GAPSYM, gap_value=gap_value)
    msa = _encode_msa_strings(MSA_STRINGS, mapping)

    dense = onehot_without_gap(msa, len(mapping), mapping.gapint).astype(int)
    sparse = get_onehotmsa_sparse_nogap(msa, NUM_AAS, mapping.gapint)

    # Sparse format is (Nseq, Npos * num_aa); reshape and compare.
    nseq, npos, naa = dense.shape
    assert sparse.shape == (nseq, npos * naa)
    assert np.array_equal(sparse.toarray(), dense.reshape(nseq, npos * naa))


@pytest.mark.parametrize("gap_value", [0, 1, 2, NUM_AAS])
def test_sparse_with_gap_preserves_similarity(gap_value):
    """The sparse-with-gap encoding is a permutation of columns relative to
    the gap-at-end layout, so pairwise sequence similarity (via dot product)
    must be invariant to gap position."""
    reference = SymMap(AA_SYMS, GAPSYM, gap_value=NUM_AAS)
    msa_ref = _encode_msa_strings(MSA_STRINGS, reference)
    sim_ref = (
        get_onehotmsa_sparse(msa_ref, NUM_AAS, reference.gapint)
        @ get_onehotmsa_sparse(msa_ref, NUM_AAS, reference.gapint).T
    ).toarray()

    mapping = SymMap(AA_SYMS, GAPSYM, gap_value=gap_value)
    msa = _encode_msa_strings(MSA_STRINGS, mapping)
    sim = (
        get_onehotmsa_sparse(msa, NUM_AAS, mapping.gapint)
        @ get_onehotmsa_sparse(msa, NUM_AAS, mapping.gapint).T
    ).toarray()

    assert np.array_equal(sim_ref, sim)


@pytest.mark.parametrize(
    "bad_gap",
    [-1, NUM_AAS + 1],
)
def test_sparse_rejects_out_of_range_gap(bad_gap):
    msa = np.zeros((2, 3), dtype=np.int_)
    with pytest.raises(ValueError):
        get_onehotmsa_sparse(msa, NUM_AAS, bad_gap)
    with pytest.raises(ValueError):
        get_onehotmsa_sparse_nogap(msa, NUM_AAS, bad_gap)
