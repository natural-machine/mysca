"""Tests for the --assignment overlap|exclusive flag.

The assignment step decides, for each IC, which positions from the t-dist
cutoff candidate set actually end up in that IC's group. Two strategies:

- `overlap` (default): the group is the raw candidate set; a position that
  clears the threshold on multiple ICs appears in each group.
- `exclusive`: among positions that qualify for more than one IC, keep
  each only in the IC where its IC-projection is maximal.
"""

import os
import numpy as np
import pytest

from tests.conftest import DATDIR, TMPDIR, remove_dir

from mysca.run_sca import assign_positions_to_groups
from mysca.run_preprocessing import parse_args as prep_parse_args
from mysca.run_preprocessing import main as prep_main
from mysca.run_sca import parse_args, main


################################################################################
# Unit-level tests (no pipeline run)
################################################################################

@pytest.fixture
def synthetic_top_idxs_and_v():
    """Three ICs, four positions, with a specific overlap pattern.

    position 0 qualifies for ICs 0 and 1
    position 1 qualifies for IC 1 only
    position 2 qualifies for ICs 0 and 2
    position 3 qualifies for ICs 2 only

    v_ica is chosen so that:
      - position 0's projection is largest on IC 0 (so exclusive → IC 0)
      - position 2's projection is largest on IC 2 (so exclusive → IC 2)
    """
    top_idxs = [
        np.array([0, 2]),   # IC 0 candidates
        np.array([0, 1]),   # IC 1 candidates
        np.array([2, 3]),   # IC 2 candidates
    ]
    v_ica = np.array([
        # col0  col1  col2
        [0.9,  0.1,  0.0],   # row 0 (position 0): IC 0 is max
        [0.0,  0.8,  0.0],   # row 1 (position 1): only IC 1
        [0.2,  0.0,  0.9],   # row 2 (position 2): IC 2 is max
        [0.0,  0.0,  0.7],   # row 3 (position 3): only IC 2
    ])
    return top_idxs, v_ica


def test_overlap_preserves_all_candidates(synthetic_top_idxs_and_v):
    top_idxs, v_ica = synthetic_top_idxs_and_v
    groups, scores = assign_positions_to_groups(
        top_idxs, v_ica, method="overlap",
    )
    # Each IC gets its full candidate set.
    assert [list(g) for g in groups] == [[0, 2], [0, 1], [2, 3]]
    # Scores are the corresponding v_ica[idx, i] values.
    np.testing.assert_allclose(scores[0], [0.9, 0.2])
    np.testing.assert_allclose(scores[1], [0.1, 0.8])
    np.testing.assert_allclose(scores[2], [0.9, 0.7])


def test_exclusive_picks_max_projection(synthetic_top_idxs_and_v):
    top_idxs, v_ica = synthetic_top_idxs_and_v
    groups, scores = assign_positions_to_groups(
        top_idxs, v_ica, method="exclusive",
    )
    # position 0 → IC 0; position 2 → IC 2; positions 1, 3 unique to
    # their respective ICs.
    assert [list(g) for g in groups] == [[0], [1], [2, 3]]


def test_overlap_is_superset_of_exclusive(synthetic_top_idxs_and_v):
    """For any input, overlap groups must be a superset of exclusive ones."""
    top_idxs, v_ica = synthetic_top_idxs_and_v
    overlap_groups, _ = assign_positions_to_groups(
        top_idxs, v_ica, method="overlap",
    )
    exclusive_groups, _ = assign_positions_to_groups(
        top_idxs, v_ica, method="exclusive",
    )
    for g_over, g_excl in zip(overlap_groups, exclusive_groups):
        assert set(g_excl).issubset(set(g_over))


def test_exclusive_weak_assignment_lowers_bar_for_other_ics():
    """weak_assignment excludes the listed ICs from the max-projection
    comparison, so a position that would otherwise lose to a "weak" IC can
    still enter non-weak groups."""
    top_idxs = [
        np.array([0]),    # IC 0 candidate
        np.array([0]),    # IC 1 candidate (listed as weak)
    ]
    v_ica = np.array([
        [0.3, 0.9],       # IC 1's projection is higher
    ])
    # Plain exclusive: IC 0 loses the tie-break (0.3 < 0.9); IC 1 wins.
    groups_plain, _ = assign_positions_to_groups(
        top_idxs, v_ica, method="exclusive",
    )
    assert list(groups_plain[0]) == []
    assert list(groups_plain[1]) == [0]
    # With IC 1 weak, IC 0 no longer has to beat IC 1's projection, so it
    # also claims the position. IC 1 still keeps it (weak_assignment only
    # affects the *comparison*, not the claimant).
    groups_weak, _ = assign_positions_to_groups(
        top_idxs, v_ica, method="exclusive", weak_assignment=(1,),
    )
    assert list(groups_weak[0]) == [0]
    assert list(groups_weak[1]) == [0]


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        assign_positions_to_groups([np.array([0])], np.zeros((1, 1)),
                                   method="bogus")


################################################################################
# End-to-end entrypoint test
################################################################################

PREP_ARGSTRING = (
    f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring5a.txt"
)
SCA_ARGSTRING = (
    f"{DATDIR}/entrypoint_tests/sca_run/argstrings/argstring5a.txt"
)


def _read_arglist(fpath):
    with open(fpath) as f:
        return f.readline().split(" ")


@pytest.fixture
def prepped_indir():
    prep_args = prep_parse_args(_read_arglist(PREP_ARGSTRING))
    prep_args.msa_fpath = f"{DATDIR}/{prep_args.msa_fpath}"
    prep_args.outdir = f"{TMPDIR}/preprocessing_out_assignment"
    prep_main(prep_args)
    yield prep_args.outdir
    remove_dir(prep_args.outdir)


def _run_sca(indir, outdir, assignment):
    args = parse_args(_read_arglist(SCA_ARGSTRING))
    args.indir = indir
    args.outdir = outdir
    args.seed = 42
    args.n_boot = 4
    args.assignment = assignment
    # Force kstar >= 2 so that IC overlap is at least possible.
    args.kstar = 3
    if isinstance(args.background, str) and args.background.endswith(".json"):
        args.background = f"{DATDIR}/{args.background}"
    main(args)
    return args


def _load_groups(outdir):
    ic_pos_dir = os.path.join(outdir, "ic_positions")
    groups = []
    i = 0
    while True:
        gpath = os.path.join(ic_pos_dir, f"ic_{i}_msaproc.npy")
        if not os.path.isfile(gpath):
            break
        groups.append(np.load(gpath))
        i += 1
    return groups


def test_entrypoint_overlap_default_is_superset_of_exclusive(prepped_indir):
    """Running the entrypoint with --assignment overlap vs exclusive: every
    position in an exclusive group must also appear in the corresponding
    overlap group."""
    over_dir = f"{TMPDIR}/scarun_assign_overlap"
    excl_dir = f"{TMPDIR}/scarun_assign_exclusive"
    _run_sca(prepped_indir, over_dir, "overlap")
    _run_sca(prepped_indir, excl_dir, "exclusive")
    groups_over = _load_groups(over_dir)
    groups_excl = _load_groups(excl_dir)
    assert len(groups_over) == len(groups_excl), (
        "Group count should match across methods (same n_components)."
    )
    for g_over, g_excl in zip(groups_over, groups_excl):
        assert set(map(int, g_excl)).issubset(set(map(int, g_over)))
    remove_dir(over_dir)
    remove_dir(excl_dir)
