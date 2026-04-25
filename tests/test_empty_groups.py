"""Tests for the empty-group edge case.

If the t-distribution cutoff exceeds every position's IC projection, an IC
can end up with zero assigned positions. The pipeline used to crash on the
downstream ``np.concatenate(groups, axis=0)`` when *every* IC was empty. The
fixes in run_sca and results.py route through ``_safe_concat_int`` so the
pipeline completes and produces a 0x0 sector subset (plus per-IC warnings).
"""

import os
import numpy as np
import pytest

from tests.conftest import DATDIR, TMPDIR, remove_dir

from mysca.run_sca import (
    _safe_concat_int,
    assign_positions_to_groups,
    fit_t_distributions,
)
from mysca.run_preprocessing import parse_args as prep_parse_args
from mysca.run_preprocessing import main as prep_main
from mysca.run_sca import parse_args, main


################################################################################
# Unit tests for the _safe_concat_int guard and the cutoff producing empties
################################################################################

def test_safe_concat_int_all_empty_returns_empty_int_array():
    out = _safe_concat_int([np.array([], dtype=int), np.array([], dtype=int)])
    assert out.shape == (0,)
    assert out.dtype == np.dtype(int) or out.dtype.kind == "i"


def test_safe_concat_int_mixed_empty_and_nonempty():
    out = _safe_concat_int([np.array([], dtype=int), np.array([1, 2])])
    np.testing.assert_array_equal(out, [1, 2])


def test_safe_concat_int_empty_list():
    out = _safe_concat_int([])
    assert out.shape == (0,)


def test_fit_t_distributions_at_100th_percentile_yields_empty_idxs():
    """ppf(1.0, ...) is +inf, so no position clears the cutoff."""
    v = np.random.default_rng(0).standard_normal((10, 3))
    _, top_idxs = fit_t_distributions(v, p=100)
    for idxs in top_idxs:
        assert len(idxs) == 0


def test_assign_positions_to_groups_with_all_empty_candidates():
    top_idxs = [np.array([], dtype=int) for _ in range(3)]
    v = np.zeros((5, 3))
    for method in ("overlap", "exclusive"):
        groups, scores = assign_positions_to_groups(
            top_idxs, v, method=method,
        )
        assert len(groups) == 3
        for g, s in zip(groups, scores):
            assert len(g) == 0
            assert len(s) == 0


################################################################################
# End-to-end test: pstar=100 → all IC groups empty, pipeline still completes
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
    prep_args.outdir = f"{TMPDIR}/preprocessing_out_empty_groups"
    prep_main(prep_args)
    yield prep_args.outdir
    remove_dir(prep_args.outdir)


def test_entrypoint_survives_all_empty_groups(prepped_indir):
    outdir = f"{TMPDIR}/scarun_all_empty"
    args = parse_args(_read_arglist(SCA_ARGSTRING))
    args.indir = prepped_indir
    args.outdir = outdir
    args.seed = 42
    args.n_boot = 4
    args.pstar = 100  # guarantees t-dist cutoff = +inf → no position qualifies
    if isinstance(args.background, str) and args.background.endswith(".json"):
        args.background = f"{DATDIR}/{args.background}"

    main(args)  # must not raise

    # Load saved IC positions from disk; each should be empty.
    ic_pos_dir = os.path.join(outdir, "ic_positions")
    assert os.path.isdir(ic_pos_dir)
    i = 0
    any_ic_seen = False
    while True:
        gpath = os.path.join(ic_pos_dir, f"ic_{i}_msaproc.npy")
        if not os.path.isfile(gpath):
            break
        any_ic_seen = True
        g = np.load(gpath)
        assert len(g) == 0, f"IC {i} unexpectedly non-empty at pstar=100"
        i += 1
    assert any_ic_seen, "Expected at least one IC position file on disk"

    # Sector-subset SCA matrix should be 0x0.
    subset_path = os.path.join(
        outdir, "sca_results", "sca_matrix_sector_subset.npy"
    )
    assert os.path.isfile(subset_path)
    subset = np.load(subset_path)
    assert subset.shape == (0, 0)

    remove_dir(outdir)
