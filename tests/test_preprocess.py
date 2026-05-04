"""Preprocessing tests

"""

import pytest
from contextlib import nullcontext as does_not_raise
from tests.conftest import DATDIR, TMPDIR, remove_dir

import numpy as np

from mysca.io import load_msa
from mysca.mappings import SymMap, NONCANONICAL
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
        np.array([0,1,2,3,4]), 
        np.array([
            0.058823529, 0.058823529, 0.058823529, 0.0625, 0.2, 0.071428571, 
            0.071428571, 0.076923077, 0.125, 0.0625, 0.125, 0.071428571, 
            0.066666667, 0.142857143, 0.071428571, 0.066666667, 
            0.125, 0.125, 0.125, 0.1
        ]),
        [f"msa07_sequence{i}" for i in np.arange(20)],
    ],

])
@pytest.mark.parametrize("weight_computation_version", ["_v3", "_v4", "sparse", "_v6", "gpu"])
@pytest.mark.parametrize("block_size", [1, 2, 20])
@pytest.mark.parametrize("gap_value", [0, "mid", "end"])
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
    gap_value,
):
    # Re-build the SymMap at the requested gap position. Retained sequences,
    # positions, weights, and seqids are invariant under gap repositioning
    # because they depend only on ``msa == GAP`` comparisons, not on axis
    # ordering; the same expectations apply. "end" preserves coverage of
    # the pre-default-change layout (gapint == len(aa_list)).
    if gap_value == "mid":
        gap_value = len(symmap.aa_list) // 2
    elif gap_value == "end":
        gap_value = len(symmap.aa_list)
    exclude = NONCANONICAL if symmap._exclude_noncanonical else symmap.exclude_syms
    symmap = SymMap(
        "".join(symmap.aa_list),
        symmap.gapsym,
        exclude,
        gap_value=gap_value,
    )

    msa_obj, msa_loaded, msa_ids_loaded, _, _, _ = load_msa(
        fa_fpath, format="fasta", mapping=symmap,
    )
    msa_obj_length = len(msa_obj)
    msa_loaded_shape_pre = msa_loaded.shape
    msa_ids_loaded_length_pre = len(msa_ids_loaded)

    msa, preprocessing_results = preprocess_msa(
        msa_loaded, msa_ids_loaded, 
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
    if msa_loaded.shape != msa_loaded_shape_pre:
        msg = "msa_loaded changed shape unexpectedly. "
        msg += f"Expected {msa_loaded_shape_pre}. Got {msa_loaded.shape}"
        errors.append(msg)
    if len(msa_ids_loaded) != msa_ids_loaded_length_pre:
        msg = "msa_ids_loaded changed shape unexpectedly. "
        msg += f"Expected {msa_ids_loaded_length_pre}. Got {len(msa_ids_loaded)}"
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


def test_filter_history_records_excluded_symbols_stage():
    """When n_excluded_pre_load > 0, preprocess_msa prepends an
    'excluded_symbols' stage immediately after 'initial' and the
    'initial' bar reflects the pre-exclusion count."""
    from mysca.preprocess import preprocess_msa
    from mysca.mappings import SymMap

    sym_map = SymMap("ACDEFGHIKLMNPQRSTVWY", "-")
    rng = np.random.default_rng(0)
    msa = rng.integers(0, 20, size=(8, 10), dtype=np.int_)
    seqids = [f"seq_{i}" for i in range(8)]

    _, results = preprocess_msa(
        msa, seqids, mapping=sym_map,
        gap_truncation_thresh=0.5,
        sequence_gap_thresh=0.5,
        sequence_similarity_thresh=0.99,
        position_gap_thresh=0.5,
        weight_computation_version="sparse",
        n_excluded_pre_load=3,
    )

    fh = results["filter_history"]
    assert fh[0]["stage"] == "initial"
    assert fh[0]["n_sequences"] == 8 + 3, (
        "initial bar must show pre-exclusion count when "
        "n_excluded_pre_load > 0"
    )
    assert fh[1]["stage"] == "excluded_symbols"
    assert fh[1]["n_sequences"] == 8
    assert fh[1]["n_filtered"] == 3
    assert fh[1]["axis"] == "sequences"
    assert fh[1]["stat_values"] is None  # no distribution by design


def test_strip_trailing_stops_unit():
    """The trailing-stop-codon helper handles all the corner cases:
    pure trailing, gaps between * and the C-terminus, multiple trailing
    stops, internal *, and clean sequences."""
    from mysca.io import _strip_trailing_stops
    cases = [
        # (in, expected_clean, n_replaced, has_internal)
        ("ACDE*", "ACDE-", 1, False),
        ("ACDE-*--", "ACDE----", 1, False),
        ("AC*DE", "AC*DE", 0, True),
        ("AC*DE-*--", "AC*DE----", 1, True),
        ("ACDE", "ACDE", 0, False),
        ("ACDE**", "ACDE--", 2, False),
        ("ACDE**--", "ACDE----", 2, False),
        ("", "", 0, False),
    ]
    for inp, exp_clean, exp_n, exp_internal in cases:
        cleaned, n, internal = _strip_trailing_stops(inp)
        assert cleaned == exp_clean, f"input={inp!r}: clean={cleaned!r}"
        assert n == exp_n, f"input={inp!r}: n={n}"
        assert internal == exp_internal, f"input={inp!r}: internal={internal}"


def test_load_msa_strips_trailing_stop_and_drops_internal_stop(tmp_path):
    """End-to-end: a FASTA mixing trailing-* and internal-* records
    yields the right counts and post-load contents from load_msa."""
    from mysca.io import load_msa
    from mysca.mappings import SymMap

    fa = tmp_path / "stops.faa"
    # AlignIO requires equal-length records; pad with gaps. Six-column
    # alignment exercising: clean, trailing-* at end, trailing-* with
    # gap after, and internal-*.
    fa.write_text(
        ">clean\nACDE--\n"
        ">trailing_at_end\nACDE-*\n"
        ">trailing_with_gap_after\nACDE*-\n"
        ">internal\nAC*DE-\n"
    )
    sym_map = SymMap("ACDE", "-")
    msa_obj, _, msa_ids, _, n_excluded, n_internal_stop = load_msa(
        str(fa), format="fasta", mapping=sym_map,
    )
    # 1 internal-stop sequence dropped; 0 excluded-symbol drops because
    # trailing * was replaced by - before the alphabet check.
    assert n_internal_stop == 1
    assert n_excluded == 0
    assert "internal" not in msa_ids
    assert msa_ids == ["clean", "trailing_at_end", "trailing_with_gap_after"]
    # Trailing * was replaced by - in the surviving sequences.
    seqs = {rec.id: str(rec.seq) for rec in msa_obj}
    assert seqs["trailing_at_end"] == "ACDE--"
    assert seqs["trailing_with_gap_after"] == "ACDE--"


def test_symmap_rejects_star():
    """Stop codons must not be members of the alphabet."""
    from mysca.mappings import SymMap
    with pytest.raises(ValueError, match="stop codon"):
        SymMap("ACDE*", "-")


def test_filter_history_records_internal_stop_codon_stage():
    """When n_internal_stop_pre_load > 0, preprocess_msa inserts an
    'internal_stop_codon' stage between 'initial' and any
    'excluded_symbols' stage."""
    from mysca.preprocess import preprocess_msa
    from mysca.mappings import SymMap

    sym_map = SymMap("ACDEFGHIKLMNPQRSTVWY", "-")
    rng = np.random.default_rng(2)
    msa = rng.integers(0, 20, size=(5, 8), dtype=np.int_)
    seqids = [f"seq_{i}" for i in range(5)]

    _, results = preprocess_msa(
        msa, seqids, mapping=sym_map,
        gap_truncation_thresh=0.5,
        sequence_gap_thresh=0.5,
        sequence_similarity_thresh=0.99,
        position_gap_thresh=0.5,
        weight_computation_version="sparse",
        n_excluded_pre_load=2,
        n_internal_stop_pre_load=4,
    )
    fh = results["filter_history"]
    stages = [s["stage"] for s in fh]
    # initial first, then internal_stop_codon, then excluded_symbols.
    assert stages[:3] == [
        "initial", "internal_stop_codon", "excluded_symbols",
    ]
    assert fh[0]["n_sequences"] == 5 + 2 + 4
    assert fh[1]["n_sequences"] == 5 + 2
    assert fh[1]["n_filtered"] == 4
    assert fh[2]["n_sequences"] == 5
    assert fh[2]["n_filtered"] == 2


def test_filter_history_omits_excluded_symbols_stage_when_none_dropped():
    """No excluded-symbols stage is recorded when n_excluded_pre_load=0
    (default), preserving the legacy filter_history shape."""
    from mysca.preprocess import preprocess_msa
    from mysca.mappings import SymMap

    sym_map = SymMap("ACDEFGHIKLMNPQRSTVWY", "-")
    rng = np.random.default_rng(1)
    msa = rng.integers(0, 20, size=(6, 10), dtype=np.int_)
    seqids = [f"seq_{i}" for i in range(6)]

    _, results = preprocess_msa(
        msa, seqids, mapping=sym_map,
        gap_truncation_thresh=0.5,
        sequence_gap_thresh=0.5,
        sequence_similarity_thresh=0.99,
        position_gap_thresh=0.5,
        weight_computation_version="sparse",
    )
    stages = [s["stage"] for s in results["filter_history"]]
    assert "excluded_symbols" not in stages
    assert results["filter_history"][0]["n_sequences"] == 6


# --------------------------------------------------------------------------- #
# Defensive guards: missing reference ID, fully-emptied MSA after filtering.  #
# --------------------------------------------------------------------------- #


def test_missing_reference_id_raises_clear_error():
    """B1: an unknown --reference must raise ValueError with the offending ID,
    not an opaque IndexError."""
    msa_obj, msa_loaded, msa_ids_loaded, _, _, _ = load_msa(
        TEST_MSA4, format="fasta", mapping=SYMMAP2,
    )
    with pytest.raises(ValueError, match="not_a_real_id"):
        preprocess_msa(
            msa_loaded, msa_ids_loaded,
            mapping=SYMMAP2,
            gap_truncation_thresh=1.0,
            sequence_gap_thresh=1.0,
            reference_id="not_a_real_id",
            reference_similarity_thresh=0.0,
            sequence_similarity_thresh=1.0,
            position_gap_thresh=1.0,
        )


def test_empty_msa_after_position_gap_filter_raises():
    """B5: gap_truncation_thresh=0 drops every position; preprocessing must
    raise rather than silently producing an empty MSA."""
    msa_obj, msa_loaded, msa_ids_loaded, _, _, _ = load_msa(
        TEST_MSA4, format="fasta", mapping=SYMMAP2,
    )
    with pytest.raises(ValueError, match="position gap"):
        preprocess_msa(
            msa_loaded, msa_ids_loaded,
            mapping=SYMMAP2,
            gap_truncation_thresh=0.0,
            sequence_gap_thresh=1.0,
            reference_id=None,
            reference_similarity_thresh=0.0,
            sequence_similarity_thresh=1.0,
            position_gap_thresh=1.0,
        )


def test_empty_msa_after_sequence_gap_filter_raises():
    """B5: sequence_gap_thresh=0 drops every sequence; preprocessing must
    raise rather than silently producing an empty MSA."""
    msa_obj, msa_loaded, msa_ids_loaded, _, _, _ = load_msa(
        TEST_MSA4, format="fasta", mapping=SYMMAP2,
    )
    with pytest.raises(ValueError, match="sequence gap"):
        preprocess_msa(
            msa_loaded, msa_ids_loaded,
            mapping=SYMMAP2,
            gap_truncation_thresh=1.0,
            sequence_gap_thresh=0.0,
            reference_id=None,
            reference_similarity_thresh=0.0,
            sequence_similarity_thresh=1.0,
            position_gap_thresh=1.0,
        )
