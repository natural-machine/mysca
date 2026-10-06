"""Test that sparse one-hot dot products do not overflow for long alignments.

Regression test for a bug where get_onehotmsa_sparse used np.uint8 data,
causing the sparse dot product (used in weight computations `_v4`,
`sparse`, and `_v6`) to silently wrap around for MSAs with more than
255 positions.

Example: two identical 300-position sequences should have a pairwise match
count of 300, but uint8 overflow gives 300 % 256 = 44.

Also covers a second bug of the same family, where the one-hot row and
column indices were built as np.uint16. Rows wrapped beyond 65,536
sequences and columns wrapped beyond 65,536 (position, symbol) pairs,
i.e. from 3,122 positions with the gap symbol and 3,278 without.
"""

import numpy as np
import pytest

from mysca.preprocess import (
    compute_weights,
    get_onehotmsa_sparse,
    get_onehotmsa_sparse_nogap,
)


NUM_AA = 20
GAP = NUM_AA  # gap is encoded as num_aa


class TestSparseOneHotOverflow:

    @pytest.mark.parametrize("npos", [100, 256, 300, 500])
    def test_dot_product_no_overflow(self, npos):
        """Sparse one-hot dot product gives correct counts for long MSAs."""
        # Two identical sequences: all amino acid 0
        msa = np.zeros((2, npos), dtype=int)
        sp = get_onehotmsa_sparse(msa, NUM_AA, GAP)

        counts = (sp @ sp.T).toarray()

        # Self-similarity should equal npos (every position matches)
        assert counts[0, 0] == npos, (
            f"Self dot product was {counts[0, 0]}, expected {npos} "
            f"(overflow if {counts[0, 0]} == {npos % 256})"
        )
        # Cross-similarity should also equal npos (identical sequences)
        assert counts[0, 1] == npos

    def test_dot_product_with_gaps(self):
        """Gaps are included in one-hot as their own symbol, contributing to counts."""
        npos = 300
        msa = np.zeros((2, npos), dtype=int)
        # Insert gaps at 50 positions
        msa[:, :50] = GAP
        sp = get_onehotmsa_sparse(msa, NUM_AA, GAP)

        counts = (sp @ sp.T).toarray()
        expected = npos  # all positions match, including gaps

        assert counts[0, 0] == expected
        assert counts[0, 1] == expected

    def test_dot_product_partial_match(self):
        """Two sequences that differ at some positions."""
        npos = 300
        rng = np.random.default_rng(42)
        seq_a = rng.integers(0, NUM_AA, size=npos)
        seq_b = seq_a.copy()
        # Make 100 positions differ
        differ_idxs = rng.choice(npos, size=100, replace=False)
        seq_b[differ_idxs] = (seq_b[differ_idxs] + 1) % NUM_AA

        msa = np.stack([seq_a, seq_b])
        sp = get_onehotmsa_sparse(msa, NUM_AA, GAP)
        counts = (sp @ sp.T).toarray()

        n_match = np.sum(seq_a == seq_b)
        assert counts[0, 1] == n_match
        # Self-similarity is always npos
        assert counts[0, 0] == npos
        assert counts[1, 1] == npos

    @pytest.mark.parametrize("version", ["_v4", "sparse"])
    def test_weights_correct_for_long_alignment(self, version):
        """Sequence weights are correct for MSAs longer than 255 positions.

        Constructs an MSA where all sequences are identical (so each has
        nseqs neighbors at threshold 1.0), giving weight = 1/nseqs.
        With uint8 overflow, the similarity counts would be wrong and the
        weights would not match.
        """
        npos = 300
        nseqs = 5
        # All identical sequences
        rng = np.random.default_rng(7)
        seq = rng.integers(0, NUM_AA, size=npos)
        msa = np.tile(seq, (nseqs, 1))

        ws = compute_weights(
            version=version,
            msa=msa,
            seqsim_thresh=1.0,
            gap=GAP,
            num_aas=NUM_AA,
            use_pbar=False,
            block_size=512,
        )

        # All identical -> every sequence has nseqs neighbors -> w = 1/nseqs
        expected = np.full(nseqs, 1.0 / nseqs)
        np.testing.assert_allclose(ws, expected, err_msg=(
            f"Weights wrong for {version} with npos={npos}. "
            f"Likely uint8 overflow in sparse dot product."
        ))


def _dense_onehot(msa, num_symbols):
    onehot = np.zeros(msa.shape + (num_symbols,), dtype=np.int16)
    np.put_along_axis(onehot, msa[:, :, None], 1, axis=2)
    return onehot


class TestSparseOneHotIndexOverflow:

    @pytest.mark.parametrize("nseqs, npos", [(65540, 3), (4, 3300)])
    def test_onehot_matches_dense(self, nseqs, npos):
        """Every sequence and position keeps its own row and columns."""
        rng = np.random.default_rng(0)
        msa = rng.integers(0, NUM_AA + 1, size=(nseqs, npos))
        expected = _dense_onehot(msa, NUM_AA + 1).reshape(nseqs, -1)

        sp = get_onehotmsa_sparse(msa, NUM_AA, GAP)

        assert sp.shape == expected.shape
        assert np.array_equal(sp.toarray(), expected)

    @pytest.mark.parametrize("nseqs, npos", [(65540, 3), (4, 3300)])
    def test_onehot_nogap_matches_dense(self, nseqs, npos):
        rng = np.random.default_rng(0)
        msa = rng.integers(0, NUM_AA + 1, size=(nseqs, npos))
        expected = np.delete(
            _dense_onehot(msa, NUM_AA + 1), GAP, axis=2
        ).reshape(nseqs, -1)

        sp = get_onehotmsa_sparse_nogap(msa, NUM_AA, GAP)

        assert sp.shape == expected.shape
        assert np.array_equal(sp.toarray(), expected)

    @pytest.mark.parametrize("bad_value", [-1, NUM_AA + 1])
    def test_rejects_out_of_range_symbols(self, bad_value):
        """Out-of-range entries would index another position's columns."""
        msa = np.zeros((2, 3), dtype=int)
        msa[0, 1] = bad_value
        with pytest.raises(ValueError):
            get_onehotmsa_sparse(msa, NUM_AA, GAP)
        with pytest.raises(ValueError):
            get_onehotmsa_sparse_nogap(msa, NUM_AA, GAP)

    def test_weights_correct_for_many_sequences(self):
        """Sequences beyond row 65,536 are still counted as neighbors.

        With a single position and threshold 1.0, each sequence's neighbors
        are exactly the sequences carrying the same symbol.
        """
        nseqs = 65540
        rng = np.random.default_rng(1)
        msa = rng.integers(0, NUM_AA, size=(nseqs, 1))

        ws = compute_weights(
            version="sparse",
            msa=msa,
            seqsim_thresh=1.0,
            gap=GAP,
            num_aas=NUM_AA,
            use_pbar=False,
            block_size=512,
        )

        expected = 1.0 / np.bincount(msa[:, 0])[msa[:, 0]]
        np.testing.assert_allclose(ws, expected)

    def test_weights_correct_for_wide_alignment(self):
        """Weights match brute force when pairs sit near the threshold.

        Column wraparound adds spurious matches between unrelated positions,
        which pushes near-threshold pairs over the similarity cutoff.
        """
        npos, thresh = 6000, 0.8
        rng = np.random.default_rng(2)
        centers = rng.integers(0, NUM_AA, size=(6, npos))
        msa = np.repeat(centers, 10, axis=0)
        for seq in msa:
            idxs = rng.choice(npos, size=rng.integers(480, 780), replace=False)
            seq[idxs] = rng.integers(0, NUM_AA, size=idxs.size)

        ws = compute_weights(
            version="sparse",
            msa=msa,
            seqsim_thresh=thresh,
            gap=GAP,
            num_aas=NUM_AA,
            use_pbar=False,
            block_size=512,
        )

        n_match = (msa[:, None, :] == msa[None, :, :]).sum(axis=2)
        expected = 1.0 / (n_match >= thresh * npos).sum(axis=1)
        np.testing.assert_allclose(ws, expected)
