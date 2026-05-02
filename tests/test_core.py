"""Core tests

"""

import pytest
from contextlib import nullcontext as does_not_raise
from tests.conftest import DATDIR, TMPDIR, remove_dir

import numpy as np

from mysca.io import load_msa
from mysca.mappings import SymMap
from mysca.preprocess import preprocess_msa
from mysca.core import (
    run_sca,
    _compute_fijab_v1,
    _compute_fijab_v2,
    _compute_fijab_v3,
    _compute_fijab_v4_jax,
    _compute_fijab_gpu,
    _compute_eigvalsh_bootstrap_gpu,
)
# from mysca.helpers import map_msa_positions_to_sequence


#####################
##  Configuration  ##
#####################

TEST_MSA5 = f"{DATDIR}/msas/msa05.faa"
TEST_MSA6 = f"{DATDIR}/msas/msa06.faa"
TEST_MSA7 = f"{DATDIR}/msas/msa07.faa"

SYMMAP5 = SymMap("ABCDEFGH", '-')
BACKGROUND_MAP5 = {
    "A": 1/8,
    "B": 1/8,
    "C": 1/8,
    "D": 1/8,
    "E": 1/8,
    "F": 1/8,
    "G": 1/8,
    "H": 1/8,
}

SYMMAP6 = SymMap("ABCD", '-')
BACKGROUND_MAP6 = {
    "A": 0.25,
    "B": 0.25,
    "C": 0.25,
    "D": 0.25,
}

SYMMAP7 = SymMap("ACDE", '-')
BACKGROUND_MAP7 = {
    "A": 0.25,
    "C": 0.25,
    "D": 0.25,
    "E": 0.25,
}


LAMBDA1 = 0.03

###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize(
        "fa_fpath, symmap, " \
        "gap_truncation_thresh, sequence_gap_thresh, " \
        "reference_id, reference_similarity_thresh, " \
        "sequence_similarity_thresh, position_gap_thresh, " \
        "background_map, regularization, " \
        "expected_results", [
    # Test MSA: msa06.faa
    [
        TEST_MSA6, SYMMAP6,
        0.4, 0.2, 
        "msa06_sequence0", 0.2, 
        0.8, 0.2,
        BACKGROUND_MAP6, LAMBDA1,
        {
            "fi0": [0, 0, 0, 0, 0],
            "fia": np.array([
                [0.457386139, 0.178871287, 0.178871287, 0.178871287],
                [0.006, 0.476594059, 0.313326733, 0.198079208],
                [0.159663366, 0.159663366, 0.476594059, 0.198079208],
                [0.166066007, 0.134052805, 0.006, 0.687881188],
                [0.006, 0.102039604, 0.134052805, 0.751907591],
            ]),
            "Di": np.array([0.096636564, 0.3097553, 0.118204883, 0.52238543, 0.630603701]),
            "Dia": np.array([
                [0.100661216, 0.014514623, 0.014514623, 0.014514623],
                [0.257595807, 0.119223059, 0.010170043, 0.007566241],
                [0.023978967, 0.023978967, 0.119223059, 0.007566241],
                [0.020531044, 0.040935192, 0.257595807, 0.422611313],
                [0.257595807, 0.070242567, 0.040935192, 0.55350727],
            ]),
            "fijab": np.genfromtxt("tests/_data/test_msa06_fijab.txt").reshape([5,5,4,4])
        },
    ],
    # Test MSA: msa07.faa
    [
        TEST_MSA7, SYMMAP7,
        0.4, 0.6, 
        "msa07_sequence0", 0.2, 
        0.8, 0.2,
        BACKGROUND_MAP7, LAMBDA1,
        {
            "fi0": [0.092735703, 0.185471406, 0.092735703, 0.185471406, 0.185471406],
            "fia": np.array([
                [0.353820711, 0.1934034, 0.11844204, 0.238380216],
                [0.006, 0.335829985, 0.185907264, 0.286355487],
                [0.170914992, 0.286355487, 0.185907264, 0.260868624],
                [0.140930448, 0.006, 0.601193199, 0.065969088],
                [0.006, 0.08096136, 0.541224111, 0.185907264],
            ]),
            "Di": np.array([0.062503348, 0.284389726, 0.018995106, 0.560311, 0.473135715]),
            "Dia": np.array([
                [0.026614074, 0.009036957, 0.053994624, 0.000363874],
                [0.257595807, 0.018398313, 0.011688344, 0.003419621],
                [0.018117296, 0.003419621, 0.011688344, 0.000312037],
                [0.035861065, 0.257595807, 0.27563541, 0.117071369],
                [0.257595807, 0.095516127, 0.192533067, 0.011688344],
            ]),
            "fijab": np.genfromtxt("tests/_data/test_msa07_fijab.txt").reshape([5,5,4,4])
        },
    ],
    # Test MSA: msa05.faa
    [
        TEST_MSA5, SYMMAP5,
        0.4, 0.2, 
        "msa05_sequence1", 0.499, 
        1.0, 0.2,
        BACKGROUND_MAP5, LAMBDA1,
        {
            "fi0": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
            "fia": np.array([
                [0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.973333333, 0.003333333],
                [0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.973333333, 0.003333333],
                [0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.973333333, 0.003333333],
                [0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.973333333, 0.003333333],
                [0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.973333333, 0.003333333],
                [0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.973333333, 0.003333333],
                [0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.973333333, 0.003333333],
                [0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.973333333, 0.003333333],
                [0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.973333333, 0.003333333],
                [0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.973333333, 0.003333333],
                [0.488333333, 0.003333333, 0.245833333, 0.245833333, 0.003333333, 0.003333333, 0.003333333, 0.003333333],
                [0.003333333, 0.973333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333],
                [0.003333333, 0.973333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333],
                [0.003333333, 0.003333333, 0.973333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333],
                [0.003333333, 0.003333333, 0.973333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333],
                [0.003333333, 0.003333333, 0.003333333, 0.973333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333],
                [0.003333333, 0.003333333, 0.003333333, 0.973333333, 0.003333333, 0.003333333, 0.003333333, 0.003333333],
                [0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.245833333, 0.245833333, 0.488333333, 0.003333333],
                [0.003333333, 0.003333333, 0.003333333, 0.003333333, 0.245833333, 0.488333333, 0.245833333, 0.003333333],
            ]),
            "Di": np.array([
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
                [0.11767738, 0.11767738, 0.11767738, 0.11767738, 0.054189847, 0.390909248, 0.054189847, 0.11767738],
            ]),
            "fijab": np.genfromtxt("tests/_data/test_msa05_fijab.txt").reshape([19,19,8,8])
        },
    ],

])
@pytest.mark.parametrize("gap_value", [0, "mid", "end"])
def test_run_sca(
    fa_fpath, symmap,
    gap_truncation_thresh,
    sequence_gap_thresh,
    sequence_similarity_thresh,
    reference_id,
    reference_similarity_thresh,
    position_gap_thresh,
    background_map,
    regularization,
    expected_results,
    gap_value,
):
    # Re-build the SymMap at the requested gap position. Ground-truth
    # expectations are indexed by aa_list order, which is invariant under
    # gap repositioning, so the same expected_results applies. "end" pins
    # gap to len(aa_list) — the pre-default-change layout that matches how
    # the Excel ground truth was originally computed.
    if gap_value == "mid":
        gap_value = len(symmap.aa_list) // 2
    elif gap_value == "end":
        gap_value = len(symmap.aa_list)
    symmap = SymMap(
        "".join(symmap.aa_list), symmap.gapsym, gap_value=gap_value,
    )

    # Equal background probability distribution if background_map is None
    if background_map is None:
        background_map = {s: 1 / len(symmap.aa2int) for s in symmap.aa2int}
    
    msa_obj, msa_loaded, msa_ids_loaded, _, _, _ = load_msa(
        fa_fpath, format="fasta", mapping=symmap,
    )

    msa, preprocessing_results = preprocess_msa(
        msa_loaded, msa_ids_loaded, 
        mapping=symmap, 
        gap_truncation_thresh=gap_truncation_thresh,
        sequence_gap_thresh=sequence_gap_thresh,
        reference_id=reference_id,
        reference_similarity_thresh=reference_similarity_thresh,
        sequence_similarity_thresh=sequence_similarity_thresh,
        position_gap_thresh=position_gap_thresh,
        verbosity=2
    )
    xmsa = preprocessing_results["msa_binary3d"]
    weights = preprocessing_results["sequence_weights"]

    sca_res = run_sca(
        xmsa, weights, background_map,
        mapping=symmap,
        regularization=regularization,
        return_keys="all",
        pbar=False,
    )

    errors = []
    for key in expected_results:
        print(key, np.array(expected_results[key]).shape, np.array(sca_res[key]).shape)
        if np.shape(expected_results[key]) != np.shape(sca_res[key]):
            msg = f"Shape mismatch in {key}\n"
            msg += f"Expected: {np.shape(expected_results[key])}\n"
            msg += f"Got: {np.shape(sca_res[key])}."
            errors.append(msg)
        if not np.allclose(expected_results[key], sca_res[key]):
            msg = f"Mismatch in {key}\n"
            msg += f"{np.max(np.abs(np.array(expected_results[key]) - np.array(sca_res[key])))}"
            # msg += f"Expected:\n{expected_results[key]}\n"
            # msg += f"     Got:\n{sca_res[key]}\n"
            errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.parametrize("nseq, npos, naas", [
    (50, 12, 8),
    (200, 25, 6),
])
def test_compute_fijab_kernels_agree(nseq, npos, naas):
    """v1 (numpy double-loop), v2 (JAX), and v3 (tensordot, the
    default) must produce identical fijab tensors up to fp64
    rounding on synthetic input. Locks in the contract that the
    fast tensordot path doesn't drift from the reference v1 kernel.
    """
    rng = np.random.default_rng(seed=12345)
    xmsa = rng.integers(0, 2, size=(nseq, npos, naas)).astype(bool)
    # Ensure no all-zero rows (each "sequence" should have at least
    # one symbol per position so the contraction isn't degenerate).
    for i in range(nseq):
        for j in range(npos):
            if not xmsa[i, j].any():
                xmsa[i, j, rng.integers(0, naas)] = True
    ws_norm = rng.random(nseq)
    ws_norm = ws_norm / ws_norm.sum()
    lam = 0.03
    nsyms = naas + 1

    f1 = _compute_fijab_v1(xmsa, ws_norm, lam, nsyms)
    f2 = _compute_fijab_v2(xmsa, ws_norm, lam, nsyms)
    f3 = _compute_fijab_v3(xmsa, ws_norm, lam, nsyms)
    f4 = _compute_fijab_v4_jax(xmsa, ws_norm, lam, nsyms)
    # GPU kernel falls back to v3 internally on no-GPU machines (with a
    # WARNING); on GPU-equipped machines this exercises torch.tensordot
    # at fp64. Either way the answer must match v1.
    f5 = _compute_fijab_gpu(xmsa, ws_norm, lam, nsyms)

    assert np.allclose(f1, f3, atol=1e-12), (
        f"v3 disagrees with v1; max abs diff = {np.max(np.abs(f1 - f3)):.3e}"
    )
    assert np.allclose(f1, f2, atol=1e-12), (
        f"v2 disagrees with v1; max abs diff = {np.max(np.abs(f1 - f2)):.3e}"
    )
    assert np.allclose(f1, f4, atol=1e-12), (
        f"v4_jax disagrees with v1; max abs diff = {np.max(np.abs(f1 - f4)):.3e}"
    )
    assert np.allclose(f1, f5, atol=1e-10), (
        f"gpu kernel disagrees with v1; max abs diff = "
        f"{np.max(np.abs(f1 - f5)):.3e}"
    )
    # v3 must preserve the (j, i) symmetry of v1 — fijab[j, i, b, a] is
    # the transpose of fijab[i, j, a, b]. v1 enforces this explicitly;
    # tensordot does it implicitly. Verify.
    assert np.allclose(f3, f3.transpose(1, 0, 3, 2)), (
        "v3 fijab is not symmetric under (i,a)<->(j,b) swap"
    )


def test_bootstrap_gpu_eigvals_match_per_iter():
    """The batched-GPU bootstrap helper must produce eigenvalues that
    rank-match the per-iter (CPU/v3) path on the same input.

    On no-GPU machines the helper raises RuntimeError; the test then
    just exercises the per-iter path and verifies its own self-
    consistency (which is trivially true). On GPU machines it
    additionally asserts the batched + per-iter paths agree on each
    iter's eigenvalues within fp64 tolerance.

    Why we test eigenvalues directly (not kstar): kstar depends on a
    statistical-significance threshold that's noisy on small synthetic
    inputs. Eigenvalue-array allclose is the right contract — kstar
    stability follows.
    """
    rng = np.random.default_rng(seed=20260427)
    nseq, npos, naas = 60, 10, 6
    nsyms = naas + 1
    lam = 0.03
    qa = np.full(naas, 1.0 / naas)

    # Build a synthetic int MSA in [1..naas] (gap=0); construct the
    # one-hot batch of `B` shuffled iters by permuting columns.
    msa = rng.integers(1, naas + 1, size=(nseq, npos)).astype(np.int8)
    B = 4
    shuffled = np.empty((B, nseq, npos), dtype=msa.dtype)
    for b in range(B):
        # Same per-column shuffling shape as run_sca's shuffle_columns.
        shuf = msa.copy()
        for col in range(npos):
            shuf[:, col] = rng.permutation(shuf[:, col])
        shuffled[b] = shuf

    # One-hot encode (drop the gap channel).
    onehot = np.eye(nsyms, dtype=bool)[shuffled]
    xmsa_batch = np.delete(onehot, 0, axis=-1).astype(bool)
    weights = rng.random(nseq)
    ws_norm = weights / weights.sum()

    # Per-iter CPU reference: run the same Cij_corr derivation per iter
    # via run_sca, take eigvalsh, sort descending.
    per_iter_evals = np.empty((B, npos))
    for b in range(B):
        res = run_sca(
            xmsa_batch[b], weights, background_map={},
            background_arr=qa, regularization=lam, return_keys=["Cij_corr"],
            pbar=False,
        )
        evals = np.linalg.eigvalsh(res["Cij_corr"])
        per_iter_evals[b] = np.flip(evals)  # descending

    try:
        batched_evals = _compute_eigvalsh_bootstrap_gpu(
            xmsa_batch, ws_norm, qa=qa, lam=lam, nsyms=nsyms,
        )
    except RuntimeError:
        pytest.skip("No GPU available for batched-bootstrap test.")

    assert batched_evals.shape == per_iter_evals.shape
    # GPU path may diverge in low-order bits; tolerance loose enough to
    # tolerate non-deterministic reductions on some accelerators while
    # still catching real bugs (sign flips, broadcast mismatches, etc.).
    assert np.allclose(batched_evals, per_iter_evals, atol=1e-9), (
        f"batched-GPU eigenvalues disagree with per-iter; "
        f"max abs diff = {np.max(np.abs(batched_evals - per_iter_evals)):.3e}"
    )


def test_resolve_torch_dtype_choices():
    """resolve_torch_dtype maps each precision choice to the expected
    torch dtype and rejects unknown values."""
    pytest.importorskip("torch")
    import torch
    from mysca._acceleration import (
        resolve_torch_dtype, PRECISION_CHOICES,
    )
    expected = {
        "fp64": torch.float64,
        "fp32": torch.float32,
        "fp16": torch.float16,
    }
    for p in PRECISION_CHOICES:
        assert resolve_torch_dtype(p) is expected[p]
    with pytest.raises(ValueError, match="Unknown precision"):
        resolve_torch_dtype("bf16")


def test_compute_fijab_gpu_precision_match():
    """The fp32 GPU kernel must agree with the fp64 GPU kernel within
    ~1e-5 relative tolerance. Skips when no GPU is available (the
    fallback CPU path ignores precision)."""
    from mysca._acceleration import detect_device
    try:
        device = detect_device()
    except Exception:
        pytest.skip("torch not available")
    if device.type == "cpu":
        pytest.skip("No GPU available for fp32-vs-fp64 GPU kernel test.")

    rng = np.random.default_rng(seed=20260428)
    nseq, npos, naas = 80, 10, 6
    xmsa = np.zeros((nseq, npos, naas), dtype=bool)
    msa_int = rng.integers(0, naas, size=(nseq, npos))
    for i in range(nseq):
        for j in range(npos):
            xmsa[i, j, msa_int[i, j]] = True
    ws_norm = rng.random(nseq)
    ws_norm = ws_norm / ws_norm.sum()
    lam = 0.03
    nsyms = naas + 1

    f64 = _compute_fijab_gpu(xmsa, ws_norm, lam, nsyms, precision="fp64")
    f32 = _compute_fijab_gpu(xmsa, ws_norm, lam, nsyms, precision="fp32")
    assert np.allclose(f32, f64, rtol=1e-4, atol=1e-6), (
        f"fp32 GPU fijab disagrees with fp64 beyond fp32 precision; "
        f"max abs diff = {np.max(np.abs(f32 - f64)):.3e}"
    )
