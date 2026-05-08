"""SCA core entrypoint tests

"""

import pytest
import os
import numpy as np
from contextlib import nullcontext as does_not_raise

from tests.conftest import DATDIR, TMPDIR, remove_dir

from mysca.run_preprocessing import parse_args as prep_parse_args
from mysca.run_preprocessing import main as prep_main
from mysca.run_sca import parse_args, main


#####################
##  Configuration  ##
#####################
 
def get_args(fpath):
    with open(fpath, 'r') as f:
        argstring = f.readline()
        arglist = argstring.split(" ")
        return arglist
        
###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize(
        "prep_argstring_fpath, argstring_fpath, sca_results_exp, " \
        "sca_eigendecomp_exp", [
    [f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring5a.txt",
     f"{DATDIR}/entrypoint_tests/sca_run/argstrings/argstring5a.txt",
     {
        "conservation": np.array([
            1.913113904, 1.913113904, 1.913113904, 1.913113904, 1.913113904, 
            1.913113904, 1.913113904, 1.913113904, 1.913113904, 1.913113904, 
            0.937572444, 1.913113904, 1.913113904, 1.913113904, 1.913113904, 
            1.913113904, 1.913113904, 0.937572444, 0.937572444
        ]),
        "Dia": np.array([
            [0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 1.904593605, 0.11767738],
            [0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 1.904593605, 0.11767738],
            [0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 1.904593605, 0.11767738],
            [0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 1.904593605, 0.11767738],
            [0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 1.904593605, 0.11767738],
            [0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 1.904593605, 0.11767738],
            [0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 1.904593605, 0.11767738],
            [0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 1.904593605, 0.11767738],
            [0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 1.904593605, 0.11767738],
            [0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 1.904593605, 0.11767738],
            [0.390909248, 0.11767738, 0.054189847, 0.054189847, 0.11767738, 0.11767738, 0.11767738, 0.11767738],
            [0.11767738, 1.904593605, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738],
            [0.11767738, 1.904593605, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738],
            [0.11767738, 0.11767738, 1.904593605, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738],
            [0.11767738, 0.11767738, 1.904593605, 0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.11767738],
            [0.11767738, 0.11767738, 0.11767738, 1.904593605, 0.11767738, 0.11767738, 0.11767738, 0.11767738],
            [0.11767738, 0.11767738, 0.11767738, 1.904593605, 0.11767738, 0.11767738, 0.11767738, 0.11767738],
            [0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.054189847, 0.054189847, 0.390909248, 0.11767738],
            [0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.054189847, 0.390909248, 0.054189847, 0.11767738 ],
        ]),
     },
     {
         
     },
    ],
    [f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring6a.txt",
     f"{DATDIR}/entrypoint_tests/sca_run/argstrings/argstring6a.txt",
     {
        "conservation": np.array([
            0.096636564, 0.3097553, 0.118204883, 0.52238543, 0.630603701
        ]),
        "Dia": np.array([
            [0.100661216, 0.014514623, 0.014514623, 0.014514623],
            [0.257595807, 0.119223059, 0.010170043, 0.007566241],
            [0.023978967, 0.023978967, 0.119223059, 0.007566241],
            [0.020531044, 0.040935192, 0.257595807, 0.422611313],
            [0.257595807, 0.070242567, 0.040935192, 0.55350727],
        ]),
        "fijab": np.genfromtxt("tests/_data/test_msa06_fijab.txt").reshape([5,5,4,4]),
        "fia": np.array([
            [0.457386139, 0.178871287, 0.178871287, 0.178871287],
            [0.006, 0.476594059, 0.313326733, 0.198079208],
            [0.159663366, 0.159663366, 0.476594059, 0.198079208],
            [0.166066007, 0.134052805, 0.006, 0.687881188],
            [0.006, 0.102039604, 0.134052805, 0.751907591],
        ]),
        "Cijab_raw": np.genfromtxt("tests/_data/test_msa06_Cijab_raw.txt").reshape([5,5,4,4]),
     },
     {
        "evals_sca": np.array([
            1.96531907, 0.32630997, 0.16828179, 0.10295909, 0.01436446
        ]),
        "evecs_sca": np.array([
            [-0.22902231, -0.03376003, -0.67626318,  0.6993266 ,  0.00441214],
            [-0.26546513, -0.62630952,  0.21078685,  0.08226435,  0.69718443],
            [-0.26501772, -0.61098485,  0.1926129 ,  0.07449714, -0.7168079 ],
            [-0.6101291 ,  0.02967662, -0.46063535, -0.64388272,  0.00958601],
            [-0.65923071,  0.48209279,  0.49895127,  0.28989671, -0.00298915]
        ]),
     },
    ],
    [f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring7a.txt",
     f"{DATDIR}/entrypoint_tests/sca_run/argstrings/argstring7a.txt",
     {
        "conservation": np.array([
            0.062503348, 0.284389726, 0.018995106, 0.560311, 0.473135715
        ]),
        "Dia": np.array([
            [0.026614074, 0.009036957, 0.053994624, 0.000363874],
            [0.257595807, 0.018398313, 0.011688344, 0.003419621],
            [0.018117296, 0.003419621, 0.011688344, 0.000312037],
            [0.035861065, 0.257595807, 0.27563541, 0.117071369],
            [0.257595807, 0.095516127, 0.192533067, 0.011688344],
        ]),
        "fijab": np.genfromtxt("tests/_data/test_msa07_fijab.txt").reshape([5,5,4,4]),
        "fia": np.array([
            [0.353820711, 0.1934034, 0.11844204, 0.238380216],
            [0.006, 0.335829985, 0.185907264, 0.286355487],
            [0.170914992, 0.286355487, 0.185907264, 0.260868624],
            [0.140930448, 0.006, 0.601193199, 0.065969088],
            [0.006, 0.08096136, 0.541224111, 0.185907264],
        ]),
     },
     {
         
     },
    ],
    [f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring8a.txt",
     f"{DATDIR}/entrypoint_tests/sca_run/argstrings/argstring8a.txt",
     {
        "conservation": np.array([
            0.045838237, 0.045838237, 0.045131096, 0.215246981, 0.119592364
        ]),
        "Dia": np.array([
            [0.029101866, 0.013070285, 0.013070285, 0.001106026],
            [0.001106026, 0.029101866, 0.013070285, 0.013070285],
            [9.07449E-05, 0.006386235, 0.007874818, 0.007874818],
            [0.079043658, 0.079043658, 0.029101866, 0.091162641],
            [0.079043658, 0.013070285, 0.029101866, 0.029101866],
        ]),
        "fijab": np.genfromtxt("tests/_data/test_msa08_fijab.txt").reshape([5,5,4,4]),
        "fi0": np.array([
            0, 0, 0.045454545, 0, 0
        ]),
        "fia": np.array([
            [0.358727273, 0.182363636, 0.182363636, 0.270545455],
            [0.270545455, 0.358727273, 0.182363636, 0.182363636],
            [0.255848485, 0.299939394, 0.197060606, 0.197060606],
            [0.094181818, 0.094181818, 0.358727273, 0.446909091],
            [0.094181818, 0.182363636, 0.358727273, 0.358727273],
        ]),
        "Cijab_raw": np.genfromtxt("tests/_data/test_msa08_Cijab_raw.txt").reshape([5,5,4,4]),
     },
     {
        "evals_sca": np.array([
            0.46532286, 0.11290216, 0.02699881, 0.01944467, 0.00371416  # out of order due to 0 determinant
        ]),
        # "evecs_sca": np.array([
        #     [-2.21111457e-01, -6.71532828e-01,  3.11692589e-02, -4.33832890e-02, -7.05194833e-01],
        #     [-2.21761069e-01, -6.69773908e-01, -2.11915621e-05, 1.04795084e-01,  7.00887245e-01],
        #     [-1.21132473e-01,  7.76741944e-02, -2.61973572e-01, 9.48391617e-01, -1.05909544e-01],
        #     [-7.70174297e-01,  2.29821166e-01, -5.33552833e-01, -2.62875237e-01,  1.52239479e-02],
        #     [-5.42300866e-01,  2.03949808e-01,  8.03566860e-01, 1.36329860e-01,  2.95250123e-03],
        # ]),
     },
    ],
])
@pytest.mark.parametrize("seed", [None, 42, 123])
@pytest.mark.parametrize("n_boot", [None, -1, 0, 1, 2, 4])
def test_main(
        prep_argstring_fpath, argstring_fpath, 
        sca_results_exp, sca_eigendecomp_exp, 
        seed, n_boot, 
):
    # Run preprocessing...
    prep_argstring = get_args(prep_argstring_fpath)
    prep_args = prep_parse_args(prep_argstring)
    prep_args.msa_fpath = f"{DATDIR}/{prep_args.msa_fpath}"
    prep_args.outdir = f"{TMPDIR}/{prep_args.outdir}"
    prep_main(prep_args)

    # Run SCA
    argstring = get_args(argstring_fpath)
    args = parse_args(argstring)
    args.indir = f"{TMPDIR}/{args.indir}"
    args.outdir = f"{TMPDIR}/{args.outdir}"
    args.save_all = True  # Need this to be able to check Cijab_raw and fijab
    if seed:
        args.seed = seed
    if n_boot:
        args.n_boot = n_boot
    if isinstance(args.background, str) and args.background.endswith(".json"):
        args.background = f"{DATDIR}/{args.background}"
    # Run the entrypoint
    main(args)

    # Open the scarun results archive
    fpath = os.path.join(args.outdir,  "scarun_results.npz")
    if not os.path.exists(fpath):
        raise FileNotFoundError(fpath)
    results = np.load(fpath)
    errors = []
    keys_to_check = [
        "Dia",
        "conservation",
        "sca_matrix",
        "phi_ia",
        "fi0",
        "fia",
        "Cijab_raw"
    ]
    for k in keys_to_check:
        if k not in results:
            msg = f"key {k} not found in results!"
            errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
    # Check values
    errors = []
    for k, v_exp in sca_results_exp.items():
        v = results[k]
        if not np.allclose(v, v_exp, atol=1e-5):
            msg = f"Mismatch in result {k}\n"
            msg += f"Expected:\n{v_exp}\nGot:\n{v}"
            errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    # Open the sca eigendecomp results archive
    fpath = os.path.join(args.outdir,  "sca_eigendecomp.npz")
    if not os.path.exists(fpath):
        raise FileNotFoundError(fpath)
    results = np.load(fpath)
    errors = []
    keys_to_check = [
        "evals_sca",
        "evecs_sca",
        "significant_evals_sca",
        "significant_evecs_sca",
    ]
    for k in keys_to_check:
        if k not in results:
            msg = f"key {k} not found in results!"
            errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
    # Check values
    errors = []
    for k, v_exp in sca_eigendecomp_exp.items():
        v = results[k]
        if not np.allclose(v, v_exp, atol=1e-5):
            msg = f"Mismatch in result {k}\n"
            msg += f"Expected:\n{v_exp}\nGot:\n{v}"
            errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
    
    # Remove the output directory
    remove_dir(prep_args.outdir)
    # remove_dir(args.outdir)


def test_main_writes_component_coverage_per_seq(tmp_path):
    """End-to-end: with the default --coverage_for ('all'), sca-core
    writes component_coverage_per_seq.npz with one key per *input* MSA
    sequence (including any dropped during preprocessing) and each value
    is a length-n_components float vector with values in [0, 1]."""
    from mysca.io import load_msa
    from mysca.mappings import SymMap
    from mysca.results import COMPONENT_COVERAGE_PER_SEQ_FNAME

    prep_argstring = get_args(
        f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring6a.txt"
    )
    prep_args = prep_parse_args(prep_argstring)
    prep_args.msa_fpath = f"{DATDIR}/{prep_args.msa_fpath}"
    prep_args.outdir = f"{tmp_path}/{prep_args.outdir}"
    prep_main(prep_args)

    sca_argstring = get_args(
        f"{DATDIR}/entrypoint_tests/sca_run/argstrings/argstring6a.txt"
    )
    sca_args = parse_args(sca_argstring)
    sca_args.indir = f"{tmp_path}/{sca_args.indir}"
    sca_args.outdir = f"{tmp_path}/{sca_args.outdir}"
    if isinstance(sca_args.background, str) and sca_args.background.endswith(".json"):
        sca_args.background = f"{DATDIR}/{sca_args.background}"
    sca_args.n_boot = 2
    main(sca_args)

    cov_path = os.path.join(sca_args.outdir, COMPONENT_COVERAGE_PER_SEQ_FNAME)
    assert os.path.isfile(cov_path), (
        f"component_coverage_per_seq.npz missing at {cov_path}"
    )

    # Default --coverage_for=all → one key per *input* MSA sequence
    # (post-load_msa). Compare against load_msa's view of the input.
    sym_map = SymMap("ABCD", "-", gap_value=4)
    _, _, msa_ids, _, _, _, _ = load_msa(
        prep_args.msa_fpath, format="fasta", mapping=sym_map,
    )
    expected_ids = set(msa_ids)

    with np.load(cov_path, allow_pickle=True) as cov:
        keys = set(cov.files)
        assert keys == expected_ids, (
            f"coverage keys differ from input MSA ids. "
            f"missing={expected_ids - keys}, extra={keys - expected_ids}"
        )
        # Determine n_components from the saved IC positions dir.
        ic_dir = os.path.join(sca_args.outdir, "ic_positions")
        n_comp = sum(
            1 for f in os.listdir(ic_dir)
            if f.startswith("ic_") and f.endswith("_msaproc.npy")
        )
        assert n_comp >= 1
        for sid in keys:
            vec = cov[sid]
            assert vec.shape == (n_comp,), (
                f"{sid}: expected ({n_comp},), got {vec.shape}"
            )
            # Values in [0, 1] (or NaN for empty IC groups).
            finite = np.isfinite(vec)
            assert np.all((vec[finite] >= 0.0) & (vec[finite] <= 1.0))


def test_main_coverage_for_reference_only(tmp_path):
    """--coverage_for='reference' restricts the keyset to the single
    reference id."""
    from mysca.results import COMPONENT_COVERAGE_PER_SEQ_FNAME

    prep_argstring = get_args(
        f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring6a.txt"
    )
    prep_args = prep_parse_args(prep_argstring)
    prep_args.msa_fpath = f"{DATDIR}/{prep_args.msa_fpath}"
    prep_args.outdir = f"{tmp_path}/{prep_args.outdir}"
    prep_main(prep_args)

    sca_argstring = get_args(
        f"{DATDIR}/entrypoint_tests/sca_run/argstrings/argstring6a.txt"
    )
    sca_args = parse_args(sca_argstring)
    sca_args.indir = f"{tmp_path}/{sca_args.indir}"
    sca_args.outdir = f"{tmp_path}/{sca_args.outdir}"
    if isinstance(sca_args.background, str) and sca_args.background.endswith(".json"):
        sca_args.background = f"{DATDIR}/{sca_args.background}"
    sca_args.n_boot = 2
    sca_args.coverage_for = "reference"
    main(sca_args)

    cov_path = os.path.join(sca_args.outdir, COMPONENT_COVERAGE_PER_SEQ_FNAME)
    with np.load(cov_path, allow_pickle=True) as cov:
        keys = list(cov.files)
        # The reference id from the prep argstring is "msa06_sequence0".
        assert keys == ["msa06_sequence0"], f"got keys={keys}"


def test_main_coverage_for_id_list(tmp_path):
    """--coverage_for=<path> restricts coverage keys to listed input IDs,
    including IDs filtered during preprocessing (a difference from
    --sectors_for, which is gated on retained sequences only)."""
    from mysca.results import COMPONENT_COVERAGE_PER_SEQ_FNAME

    prep_argstring = get_args(
        f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring6a.txt"
    )
    prep_args = prep_parse_args(prep_argstring)
    prep_args.msa_fpath = f"{DATDIR}/{prep_args.msa_fpath}"
    prep_args.outdir = f"{tmp_path}/{prep_args.outdir}"
    prep_main(prep_args)

    # msa06.faa has 23 sequences (msa06_sequence0..22). Pick the
    # reference plus another one and a deliberately-missing id; the
    # missing id should be skipped silently with a log.
    coverage_list = tmp_path / "ids.txt"
    coverage_list.write_text(
        "msa06_sequence0\nmsa06_sequence5\nnonexistent_id\n"
    )

    sca_argstring = get_args(
        f"{DATDIR}/entrypoint_tests/sca_run/argstrings/argstring6a.txt"
    )
    sca_args = parse_args(sca_argstring)
    sca_args.indir = f"{tmp_path}/{sca_args.indir}"
    sca_args.outdir = f"{tmp_path}/{sca_args.outdir}"
    if isinstance(sca_args.background, str) and sca_args.background.endswith(".json"):
        sca_args.background = f"{DATDIR}/{sca_args.background}"
    sca_args.n_boot = 2
    sca_args.coverage_for = str(coverage_list)
    main(sca_args)

    cov_path = os.path.join(sca_args.outdir, COMPONENT_COVERAGE_PER_SEQ_FNAME)
    with np.load(cov_path, allow_pickle=True) as cov:
        assert set(cov.files) == {"msa06_sequence0", "msa06_sequence5"}
