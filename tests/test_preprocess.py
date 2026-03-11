"""Preprocessing tests

"""

import pytest
from contextlib import nullcontext as does_not_raise
from tests.conftest import DATDIR, TMPDIR, remove_dir

import numpy as np

from mysca.io import load_msa
from mysca.mappings import SymMap
from mysca.preprocess import preprocess_msa


#####################
##  Configuration  ##
#####################

SYMMAP1 = SymMap("ACDEF", '-')
SYMMAP1_EXC_X = SymMap("ACDEF", '-', "X")

SYMMAP2 = SymMap("ABCDEFGH", '-')

TEST_MSA1 = f"{DATDIR}/msas/msa01.faa"
TEST_MSA2 = f"{DATDIR}/msas/msa02.faa"
TEST_MSA3 = f"{DATDIR}/msas/msa03.faa"
TEST_MSA4 = f"{DATDIR}/msas/msa04.faa"
TEST_MSA5 = f"{DATDIR}/msas/msa05.faa"
TEST_MSA6 = f"{DATDIR}/msas/msa06.faa"
TEST_MSA7 = f"{DATDIR}/msas/msa07.faa"

        
###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize(
        "fa_fpath, symmap, " \
        "gap_truncation_thresh, sequence_gap_thresh, " \
        "reference_id, reference_similarity_thresh, " \
        "sequence_similarity_thresh, position_gap_thresh, " \
        "retained_sequences_exp, retained_positions_exp, " \
        "weights_exp, seqids_exp", [
    # Test MSA: msa01.faa
    [# Keep all positions and sequences, regardless of gaps
        TEST_MSA1, SYMMAP1,
        1.0, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.arange(5),
        np.arange(10),
        None, None,
    ],
    [# Keep positions with fewer than 50% gaps. Keep all sequences.
        TEST_MSA1, SYMMAP1,
        0.5, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.arange(5),
        np.arange(10),
        None, None,
    ],
    [# Keep positions with fewer than 40% gaps. Keep all sequences.
        TEST_MSA1, SYMMAP1,
        0.4, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.arange(5),
        np.arange(10),
        None, None,
    ],

    # Test MSA: msa02.faa
    [# Keep all positions and sequences, regardless of gaps
        TEST_MSA2, SYMMAP1_EXC_X,
        1.0, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.arange(2),
        np.arange(10),
        None, None,
    ],
    [# Keep positions with fewer than 50% gaps. Keep all sequences.
        TEST_MSA2, SYMMAP1_EXC_X,
        0.5, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.arange(2),
        np.arange(10),
        None, None,
    ],
    [# Keep positions with fewer than 40% gaps. Keep all sequences.
        TEST_MSA2, SYMMAP1_EXC_X,
        0.4, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.arange(2),
        np.arange(10),
        None, None,
    ],

    # Test MSA: msa03.faa
    [# Keep all positions and sequences, regardless of gaps
        TEST_MSA3, SYMMAP1,
        1.0, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        None, None,
    ],
    [# Keep positions with fewer than 50% gaps. Keep all sequences.
        TEST_MSA3, SYMMAP1,
        0.5, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 1, 2, 4, 5, 6, 7, 8, 9]),  # remove position 3
        None, None,
    ],
    [# Keep positions with fewer than 40% gaps. Keep all sequences.
        TEST_MSA3, SYMMAP1,
        0.4, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 2, 6, 8]),  # remove position 1, 3, 4, 5, 7, 9
        None, None,
    ],
    [# Keep positions with fewer than 40% gaps. Keep sequences with fewer than 50% gaps.
        TEST_MSA3, SYMMAP1,
        0.4, 0.5,
        None, 1.0, 
        1.0, 1.0, 
        np.array([1, 2, 3, 4]),  # remove sequence 0
        np.array([0, 2, 6, 8]),  # remove position 1, 3, 4, 5, 7, 9
        None, None,
    ],
    [# Keep positions with fewer than 40% gaps. Keep sequences with fewer than 20% gaps.
        TEST_MSA3, SYMMAP1,
        0.4, 0.2,
        None, 1.0, 
        1.0, 1.0, 
        np.array([3, 4]),  # remove sequence 0, 1, 2
        np.array([0, 2, 6, 8]),  # remove position 1, 3, 4, 5, 7, 9
        None, None,
    ],

    # Test MSA: msa04.faa
    [
        TEST_MSA4, SYMMAP2,
        0.4, 0.2, 
        "msa04_sequence1", 0.499, 
        1.0, 0.2, 
        np.arange(20),  # remove sequences 20, 21, 22
        np.concatenate(  # remove positions 10, 16, 17
            [np.arange(10), np.arange(11, 16), np.arange(18, 22)]
        ), 
        np.array([
            0.25, 0.1, 0.25, 0.25, 0.25, 0.1, 0.2, 0.2, 0.1, 
            0.1, 1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.1
        ]),
        [f"msa04_sequence{i}" for i in range(20)],
    ],

    # Test MSA: msa05.faa
    [
        TEST_MSA5, SYMMAP2,
        0.4, 0.2, 
        "msa05_sequence1", 0.499, 
        1.0, 0.2, 
        np.array([0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20]),
        np.array([0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,18,19,20,21]), 
        np.array([
            0.25, 0.1, 0.25, 0.25, 0.25, 0.1, 0.2, 0.2, 0.1, 
            0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.1, 1
        ]),
        [f"msa05_sequence{i}" for i in 
         [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20]
        ],
    ],

    # Test MSA: msa06.faa
    [
        TEST_MSA6, SYMMAP2,
        0.4, 0.2, 
        "msa06_sequence0", 0.2, 
        0.8, 0.2, 
        np.arange(20),
        np.array([0,1,2,3,4]), 
        np.array([
            0.2, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 
            0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.333333333, 0.333333333, 0.333333333
        ]),
        [f"msa06_sequence{i}" for i in np.arange(20)],
    ],

    # Test MSA: msa07.faa  # msa07b in excel sheet
    [
        TEST_MSA7, SYMMAP2,
        0.4, 0.6, 
        "msa07_sequence0", 0.2, 
        0.4, 0.2, 
        np.arange(20),
        np.array([0,1,2]), 
        np.array([
            0.125,0.125,0.125,0.125,0.25,0.25,0.25,0.25,0.5,
            1/3, 1/3,1, 1/3,1,0.25, 1/3,
            0.125,0.125,0.125,0.125
        ]),
        [f"msa07_sequence{i}" for i in np.arange(20)],
    ],

])
@pytest.mark.parametrize("weight_computation_version", ["v3", "v4", "v5"])
@pytest.mark.parametrize("block_size", [1, 2, 20])
def test_preprocessing(
    fa_fpath, symmap, 
    gap_truncation_thresh,
    sequence_gap_thresh,
    sequence_similarity_thresh,
    reference_id,
    reference_similarity_thresh, 
    position_gap_thresh,
    retained_sequences_exp,
    retained_positions_exp,
    weights_exp,
    seqids_exp,
    weight_computation_version,
    block_size,
):
    
    msa_obj, msa_orig, msa_ids_orig, _ = load_msa(
        fa_fpath, format="fasta", mapping=symmap,
    )
    msa_obj_length = len(msa_obj)
    msa_orig_shape = msa_orig.shape
    msa_ids_orig_length = len(msa_ids_orig)

    msa, preprocessing_results = preprocess_msa(
        msa_orig, msa_ids_orig, 
        mapping=symmap, 
        gap_truncation_thresh=gap_truncation_thresh,
        sequence_gap_thresh=sequence_gap_thresh,
        reference_id=reference_id,
        reference_similarity_thresh=reference_similarity_thresh,
        sequence_similarity_thresh=sequence_similarity_thresh,
        position_gap_thresh=position_gap_thresh,
        verbosity=2,
        weight_computation_version=weight_computation_version,
        block_size=block_size,
    )

    retained_sequences = preprocessing_results["retained_sequences"]
    retained_positions = preprocessing_results["retained_positions"]
    seqids = preprocessing_results["retained_sequence_ids"]
    weights = preprocessing_results["sequence_weights"]

    errors = []
    # Check retained sequences
    if len(retained_sequences_exp) != len(retained_sequences) or \
            np.any(retained_sequences_exp != retained_sequences):
        msg = "Mismatch in retained sequences. "
        msg += f"Expected {retained_sequences_exp}. Got {retained_sequences}"
        errors.append(msg)
    # Check retained positions
    if len(retained_positions_exp) != len(retained_positions) or \
            np.any(retained_positions_exp != retained_positions):
        msg = "Mismatch in retained positions. "
        msg += f"Expected {retained_positions_exp}. Got {retained_positions}"
        errors.append(msg)
    # Check weights
    if weights_exp is not None:
        if not np.allclose(weights_exp, weights):
            msg = "Mismatch in weights. "
            msg += f"Expected {weights_exp}.\nGot {weights}"
            errors.append(msg)
    # Check sequence IDs
    if seqids_exp is not None:
        if len(seqids) != len(seqids_exp) or \
                any([x != y for x, y in zip(seqids_exp, seqids)]):
            msg = "Mismatch in seqids. "
            msg += f"Expected {seqids_exp}.\nGot {seqids}"
            errors.append(msg)
    # Check no change to original MSA objects
    if len(msa_obj) != msa_obj_length:
        msg = "msa_obj changed shape unexpectedly. "
        msg += f"Expected {msa_obj_length}. Got {len(msa_obj)}"
        errors.append(msg)
    if msa_orig.shape != msa_orig_shape:
        msg = "msa_orig changed shape unexpectedly. "
        msg += f"Expected {msa_orig_shape}. Got {msa_orig.shape}"
        errors.append(msg)
    if len(msa_ids_orig) != msa_ids_orig_length:
        msg = "msa_ids_orig changed shape unexpectedly. "
        msg += f"Expected {msa_ids_orig_length}. Got {len(msa_ids_orig)}"
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
