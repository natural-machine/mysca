"""Tests for --n_components: decoupling IC count from kstar.

The sca-run pipeline determines kstar (the number of significant eigenvalues
identified by bootstrap or set via --kstar), and by default runs ICA on
exactly kstar eigenvectors. `--n_components` lets the caller request that
ICA run on more eigenvectors (up to all L) so that less-significant
components are still captured. The value is clamped so that
n_components >= kstar and <= L.
"""

import os
import pytest
import numpy as np

from tests.conftest import DATDIR, TMPDIR, remove_dir

from mysca.run_preprocessing import parse_args as prep_parse_args
from mysca.run_preprocessing import main as prep_main
from mysca.run_sca import parse_args, main


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
    """Run preprocessing once and yield the output dir for sca-run to load."""
    prep_args = prep_parse_args(_read_arglist(PREP_ARGSTRING))
    prep_args.msa_fpath = f"{DATDIR}/{prep_args.msa_fpath}"
    prep_args.outdir = f"{TMPDIR}/preprocessing_out_n_components"
    prep_main(prep_args)
    yield prep_args.outdir
    remove_dir(prep_args.outdir)


def _run_sca(indir, outdir, n_components=None, kstar=None):
    argstring = _read_arglist(SCA_ARGSTRING)
    args = parse_args(argstring)
    args.indir = indir
    args.outdir = outdir
    args.seed = 42
    args.n_boot = 4
    if n_components is not None:
        args.n_components = n_components
    if kstar is not None:
        args.kstar = kstar
    background = args.background
    if isinstance(background, str) and background.endswith(".json"):
        args.background = f"{DATDIR}/{background}"
    main(args)
    return args


def _load_v_ica(outdir):
    return np.load(os.path.join(outdir, "sca_results", "v_ica_normalized.npy"))


def _load_kstar(outdir):
    return int(
        np.loadtxt(os.path.join(outdir, "sca_results", "kstar.txt"))
    )


def _load_n_components(outdir):
    return int(
        np.loadtxt(os.path.join(outdir, "sca_results", "n_components.txt"))
    )


def _load_L(outdir):
    evecs = np.load(os.path.join(outdir, "sca_results", "all_evecs_sca.npy"))
    return evecs.shape[1]


def test_default_n_components_equals_kstar(prepped_indir):
    """No --n_components → n_components == kstar (baseline behavior)."""
    outdir = f"{TMPDIR}/scarun_nc_default"
    _run_sca(prepped_indir, outdir)
    kstar = _load_kstar(outdir)
    n_components = _load_n_components(outdir)
    v_ica = _load_v_ica(outdir)
    assert n_components == kstar
    assert v_ica.shape[1] == kstar
    remove_dir(outdir)


def test_n_components_all_runs_ica_on_all_eigenvectors(prepped_indir):
    """--n_components all → ICA runs on all L eigenvectors."""
    outdir = f"{TMPDIR}/scarun_nc_all"
    _run_sca(prepped_indir, outdir, n_components="all")
    L = _load_L(outdir)
    n_components = _load_n_components(outdir)
    v_ica = _load_v_ica(outdir)
    assert n_components == L
    assert v_ica.shape[1] == L
    remove_dir(outdir)


def test_n_components_int_above_kstar(prepped_indir):
    """An explicit integer above kstar is respected (and <= L)."""
    # First run with defaults so we can learn kstar for this MSA/seed.
    probe_dir = f"{TMPDIR}/scarun_nc_probe"
    _run_sca(probe_dir_indir := prepped_indir, probe_dir)
    kstar = _load_kstar(probe_dir)
    L = _load_L(probe_dir)
    remove_dir(probe_dir)

    target = min(kstar + 2, L)
    if target == kstar:
        pytest.skip(f"kstar ({kstar}) already at L ({L}); nothing to test.")

    outdir = f"{TMPDIR}/scarun_nc_explicit"
    _run_sca(prepped_indir, outdir, n_components=target)
    assert _load_n_components(outdir) == target
    assert _load_v_ica(outdir).shape[1] == target
    remove_dir(outdir)


def test_n_components_clamped_up_to_kstar(prepped_indir):
    """Passing n_components < kstar is clamped up to kstar with a warning."""
    # Force kstar high via --kstar so we can request a lower n_components.
    outdir = f"{TMPDIR}/scarun_nc_clamp"
    _run_sca(prepped_indir, outdir, kstar=4, n_components=1)
    kstar = _load_kstar(outdir)
    n_components = _load_n_components(outdir)
    assert kstar == 4
    assert n_components == kstar
    assert _load_v_ica(outdir).shape[1] == kstar
    remove_dir(outdir)


def test_n_components_clamped_down_to_L(prepped_indir):
    """Passing n_components > L is clamped down to L."""
    outdir = f"{TMPDIR}/scarun_nc_above_L"
    _run_sca(prepped_indir, outdir, n_components=10_000)
    L = _load_L(outdir)
    n_components = _load_n_components(outdir)
    assert n_components == L
    assert _load_v_ica(outdir).shape[1] == L
    remove_dir(outdir)


@pytest.mark.parametrize("bad", ["nope", "0", "-1"])
def test_n_components_type_rejects_bad_strings(bad):
    """argparse type converter rejects non-positive ints and unknown strings."""
    from argparse import ArgumentTypeError
    from mysca.run_sca import _n_components_type
    with pytest.raises(ArgumentTypeError):
        _n_components_type(bad)


def test_n_components_type_accepts_all_and_positive_int():
    from mysca.run_sca import _n_components_type
    assert _n_components_type("all") == "all"
    assert _n_components_type("All") == "all"
    assert _n_components_type("3") == 3
