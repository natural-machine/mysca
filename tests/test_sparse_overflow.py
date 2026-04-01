"""Test that sparse one-hot dot products do not overflow for long alignments.

Regression test for a bug where get_onehotmsa_sparse used np.uint8 data,
causing the sparse dot product (used in weight computation v4/v5/v6) to
silently wrap around for MSAs with more than 255 positions.

Example: two identical 300-position sequences should have a pairwise match
count of 300, but uint8 overflow gives 300 % 256 = 44.
"""

import numpy as np
import pytest

from mysca.preprocess import get_onehotmsa_sparse, compute_weights


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

    @pytest.mark.parametrize("version", ["v4", "v5"])
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
