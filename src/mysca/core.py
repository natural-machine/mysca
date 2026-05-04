"""Core SCA functionality.

"""

import logging
import warnings

import numpy as np
from numpy.typing import NDArray
import tqdm

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from mysca.mappings import SymMap, DEFAULT_MAP

logger = logging.getLogger(__name__)


def run_sca(
        xmsa: NDArray[np.bool],
        ws: NDArray[np.float64],
        background_map: dict[str, float],
        mapping: SymMap = DEFAULT_MAP,
        background_arr: NDArray[np.float64] | None = None,
        regularization: float = 0.03,
        return_keys: str = "all",
        pbar: bool = True,
        leave_pbar: bool = True,
        verbosity: int = 1,
        use_jax: bool = False,
        freq_method: str | None = None,
        precision: str = "fp64",
):
    """Run SCA algorithm on given MSA matrix

    Args: # TODO
        xmsa (_type_): _description_
        ws (_type_): _description_
        background_map (_type_): _description_
        mapping (SymMap): SymMap mapping symbols to integer values.
        background_arr (_type_, optional): _description_. Defaults to None.
        regularization (float, optional): _description_. Defaults to 0.03.
        return_keys (str, optional): _description_. Defaults to "all".
        pbar (bool, optional): _description_. Defaults to True.
        leave_pbar (bool, optional): _description_. Defaults to True.
        verbosity (int, optional): _description_. Defaults to 1.
        use_jax (bool, optional): Use JAX for computations. Defaults to False.

    Returns:  # TODO
        _type_: _description_
    """
    lam = regularization  # brevity
    qa = background_arr  # brevity
    results = {}

    nseq, npos, naas = xmsa.shape

    # Dictionary size
    nsyms = naas + 1

    # Compute positional conservation
    logger.info(
        "Frequency regularization: λ=%g, nsyms=%d (uniform pseudocount "
        "λ/nsyms=%g per amino-acid).", lam, nsyms, lam / nsyms,
    )
    ws_norm = ws / ws.sum()
    fi0 = 1 - np.sum(ws[:,None,None] * xmsa, axis=(0,2)) / ws.sum()
    n_fi0_zero = int(np.sum(np.isclose(fi0, 0)))
    if n_fi0_zero > 0:
        # fi0 ≈ 0 just means "this column has effectively no gaps after
        # weighting" — the normal state of any well-conserved column. The
        # only fi0 consumer in run_sca is the Di term below (line ~108),
        # which already guards the case via np.where(fi0 > 0, ..., 0)
        # under errstate('ignore'). DEBUG-only diagnostic so a future
        # NaN-chase has a breadcrumb without polluting healthy runs.
        logger.debug(
            "fi0 ≈ 0 at %d position(s) (well-conserved columns); "
            "downstream Di guards this case.", n_fi0_zero,
        )
    fia = (1 - lam) * np.sum(ws_norm[:,None,None] * xmsa, axis=0) + lam / nsyms

    # Resolve fijab kernel. freq_method is the user-facing surface
    # ("numpy" | "jax" | "gpu"); use_jax is the deprecated legacy flag.
    freq_method = _resolve_freq_method(freq_method=freq_method, use_jax=use_jax)
    fijab_version = _FREQ_METHOD_TO_VERSION[freq_method]

    # Compute correlated conservation. The default (numpy -> v3,
    # tensordot) is ~9x faster than v1 (numpy double-loop) on
    # SH3-scale input with bit-stable fp64 numerics; see
    # docs/.claude_sessions/session_2026-04-27_scacore_perf_phase1.md.
    fijab = compute_fijab(
        xmsa, ws_norm, lam, nsyms,
        pbar=pbar,
        leave_pbar=leave_pbar,
        version=fijab_version,
        precision=precision,
    )

    if qa is None:
        qa = np.array(
            [background_map.get(a, 0.0) for a in mapping.aa_list]
        )
        qa = qa / qa.sum()
    
    
    # Separate frequencies for Dia, Di, and phi_ia, regularized with UniProt
    # background (lam * qa) to match pySCA's posWeights. This differs from fia
    # (line 62) which uses uniform regularization (lam / nsyms) to match pySCA's
    # scaMat covariance. See docs/pysca_frequency_inconsistency.md for details.
    #fia_pw = (1 - lam) * np.sum(ws_norm[:,None,None] * xmsa, axis=0) + lam * qa

    Dia = np.nan * np.ones([npos, naas])
    Dia[:] = fia * np.log(fia / qa) + (1 - fia) * np.log((1 - fia) / (1 - qa))
    q0hat = np.sum(~np.any(xmsa, axis=-1)) / (nseq * npos)
    qahat = (1 - q0hat) * qa
    Di = np.sum(fia * np.log(fia / qahat), axis=1)
    if q0hat > 0:
        with np.errstate(divide='ignore'):
            Di += fi0 * np.where(fi0 > 0, np.log(fi0 / q0hat), 0)

    Cijab_raw = fijab - fia[:,None,:,None] * fia[None,:,None,:]
    Cij_raw = np.sqrt(np.sum(np.square(Cijab_raw), axis=(-1, -2)))
    phi_ia = np.log((fia * (1 - qa)) / ((1 - fia) * qa))
    Cijab_corr = phi_ia[:,None,:,None] * phi_ia[None,:,None,:] * Cijab_raw
    Cij_corr = np.sqrt(np.sum(np.square(Cijab_corr), axis=(-1,-2)))

    if return_keys == "all":
        results["fi0"] = fi0
        results["fia"] = fia
        results["fijab"] = fijab
        results["Dia"] = Dia
        results["Di"] = Di
        results["Cijab_raw"] = Cijab_raw
        results["Cij_raw"] = Cij_raw
        results["phi_ia"] = phi_ia
        results["Cijab_corr"] = Cijab_corr
        results["Cij_corr"] = Cij_corr
    else:
        for k in return_keys:
            results[k] = eval(k)
    return results


def run_ica(
        v: NDArray, 
        rho: float = 1e-1, 
        tol: float = 1e-6, 
        maxiter: int = 1000000,
        verbosity: int = 1,
) -> tuple[NDArray[np.float64], float]:
    """Implements ICA using the infomax algorithm.

    Independent components V^* are computed by applying the returned matrix
    W to the eigenvectors in input V: V* = WV, with V of shape (k, m). If 
    working instead with eigenvectors in the columns of V, then the 
    corresponding transform is given by the matrix product V* = VW^T.

    Refs: 
        [1] Bell and Sejnowski, 1995
        [2] SI to Rivoire et al., 2016
    
    Args:
        (NDArray) v: 2d eigenvector array. Shape (k, m), where k is the number 
            of eigenvectors.
        (float) rho: ICA stepsize parameter. Default 1e-4.
        (float) tol: Convergence threshold. Default 1e-7
        (int) maxiter: Maximum steps before halting. Default 1000000.
        (int) verbosity: Verbosity level. Default 1.

    Returns:
        (NDArray) Independent Component vectors W. Shape (k, k).
        (float) Achieved delta W value.
    """
    k, m = v.shape
    id = np.eye(k)
    w = np.eye(k)
    itercount = 0
    while itercount < maxiter:
        y = w @ v
        g = 1 / (1 + np.exp(-y))
        dw = rho * (id + (1 - 2 * g) @ y.T) @ w
        w += dw
        if np.max(np.abs(dw)) < tol:
            logger.info("Converged in %d iterations", itercount)
            return w, np.max(np.abs(dw))
        itercount += 1
    logger.warning("Did not converge in %d iterations", maxiter)
    return None, np.max(np.abs(dw))


# Mapping from user-facing --freq_method values to internal compute_fijab
# version codes. Centralized so run_sca and compute_fijab agree.
FREQ_METHOD_CHOICES = ("numpy", "jax", "gpu")
_FREQ_METHOD_TO_VERSION = {
    "numpy": "v3",
    "jax":   "v4",
    "gpu":   "v5",
}


def _resolve_freq_method(*, freq_method, use_jax):
    """Resolve the final freq_method given both `freq_method` (new) and
    `use_jax` (deprecated). Emits a DeprecationWarning if `use_jax`
    is set; freq_method wins on conflict."""
    if freq_method is None:
        if use_jax:
            warnings.warn(
                "`use_jax=True` is deprecated; pass `freq_method='jax'` "
                "instead. Routing to freq_method='jax' for now.",
                DeprecationWarning, stacklevel=3,
            )
            return "jax"
        return "numpy"
    if freq_method not in _FREQ_METHOD_TO_VERSION:
        raise ValueError(
            f"Unknown freq_method: {freq_method!r}. Choices: "
            f"{FREQ_METHOD_CHOICES}"
        )
    if use_jax:
        warnings.warn(
            "Both `freq_method` and `use_jax` were set; `use_jax` is "
            "deprecated and is being ignored.",
            DeprecationWarning, stacklevel=3,
        )
    return freq_method


def compute_fijab(
        xmsa, ws_norm, lam, nsyms, pbar=False, leave_pbar=False,
        version="v3", precision="fp64",
):
    if version == "v1":
        return _compute_fijab_v1(
            xmsa, ws_norm, lam, nsyms, pbar=pbar, leave_pbar=leave_pbar
        )
    elif version == "v2":
        return _compute_fijab_v2(
            xmsa, ws_norm, lam, nsyms, pbar=pbar, leave_pbar=leave_pbar
        )
    elif version == "v3":
        return _compute_fijab_v3(xmsa, ws_norm, lam, nsyms)
    elif version == "v4":
        return _compute_fijab_v4_jax(xmsa, ws_norm, lam, nsyms)
    elif version == "v5":
        return _compute_fijab_gpu(
            xmsa, ws_norm, lam, nsyms, precision=precision,
        )
    else:
        raise RuntimeError(f"Unknown version to compute fijab: {version}")


def _compute_fijab_v1(xmsa, ws_norm, lam, nsyms, pbar=False, leave_pbar=False):
    """Standard implementation"""
    _, npos, naas = xmsa.shape
    fijab = np.full([npos, npos, naas, naas], np.nan)
    for i in tqdm.trange(npos, disable=not pbar, leave=leave_pbar):
        ci = xmsa[:,i,:]
        for j in range(i, npos):
            cj = xmsa[:,j,:]
            f = (1 - lam) * (ci.T @ (ws_norm[:, None] * cj)) \
                + lam / (nsyms**2 * (i != j) + nsyms * (i == j))
            fijab[i,j,:,:] = f
            fijab[j,i,:,:] = f.T
    return fijab


def _compute_fijab_v2(xmsa, ws_norm, lam, nsyms, pbar=False, leave_pbar=False):
    """JAX implementation"""
    _, npos, naas = xmsa.shape
    fijab = np.full([npos, npos, naas, naas], np.nan)

    @jax.jit
    def compute_f(ci, cj, ws_norm, lam, nsyms, regterm):
        return (1 - lam) * (ci.T @ (ws_norm[:, None] * cj)) + regterm

    for i in tqdm.trange(npos, disable=not pbar, leave=leave_pbar):
        ci = xmsa[:,i,:]
        for j in range(i, npos):
            cj = xmsa[:,j,:]
            if i == j:
                regterm = np.array(lam / nsyms)
            else:
                regterm = np.array(lam / (nsyms * nsyms))
            f = compute_f(ci, cj, ws_norm, lam, nsyms, regterm)
            fijab[i,j,:,:] = f
            fijab[j,i,:,:] = f.T
    return fijab


def _compute_fijab_v3(xmsa, ws_norm, lam, nsyms):
    """Vectorized tensordot equivalent of v1.

    Roughly 9x faster than v1 on SH3-scale input (npos=62) with
    bit-stable fp64 numerics (max abs diff vs v1 ~1e-16). Replaces
    v1's `O(npos^2)` Python-driven inner loops with a single
    `np.tensordot` over the sequence axis. Memory footprint is the
    two operands plus the result — no large intermediate.

    Naive `np.einsum('s,sia,sjb->ijab', ...)` (without `optimize=`)
    materializes a `(nseq, npos, naas, npos, naas)` intermediate and
    is materially slower than v1 on small npos; that path was tried
    and rejected. tensordot avoids the issue because numpy contracts
    the sequence axis directly.
    """
    npos = xmsa.shape[1]
    xf = xmsa.astype(np.float64, copy=False)
    weighted = ws_norm[:, None, None] * xf
    # tensordot returns shape (i, a, j, b); transpose to (i, j, a, b)
    # to match v1's layout.
    fijab = np.tensordot(weighted, xf, axes=([0], [0])).transpose(0, 2, 1, 3)
    fijab *= (1.0 - lam)
    diag = np.eye(npos, dtype=bool)
    fijab[~diag] += lam / (nsyms * nsyms)
    fijab[diag] += lam / nsyms
    return fijab


@jax.jit
def _fijab_jax_kernel(xf, ws_norm, lam, reg_diag, reg_offdiag):
    weighted = ws_norm[:, None, None] * xf
    fijab = jnp.tensordot(weighted, xf, axes=([0], [0])).transpose(0, 2, 1, 3)
    fijab = fijab * (1.0 - lam)
    npos = xf.shape[1]
    diag = jnp.eye(npos, dtype=bool)
    return jnp.where(
        diag[:, :, None, None],
        fijab + reg_diag,
        fijab + reg_offdiag,
    )


def _compute_fijab_v4_jax(xmsa, ws_norm, lam, nsyms):
    """JAX-jitted equivalent of v3.

    The whole tensordot + regularization is lifted under `jax.jit` so
    JAX can compile and (on accelerator-equipped backends) execute it
    on-device. Comparable speed to v3 on CPU; the GPU path lives in
    `_compute_fijab_gpu` so this stays a CPU-flavored alternative
    for users on JAX-only setups.

    Numerics match v1/v3 within fp64 tolerance; verified by
    tests/test_core.py::test_compute_fijab_kernels_agree.
    """
    xf = jnp.asarray(xmsa, dtype=jnp.float64)
    ws_jnp = jnp.asarray(ws_norm, dtype=jnp.float64)
    fijab = _fijab_jax_kernel(
        xf, ws_jnp,
        float(lam),
        float(lam / nsyms),
        float(lam / (nsyms * nsyms)),
    )
    return np.asarray(fijab)


def _compute_fijab_gpu(xmsa, ws_norm, lam, nsyms, *, precision="fp64"):
    """torch GPU equivalent of v3.

    Lazy-imports torch and uses ``torch.tensordot`` on the first
    available accelerator (CUDA / MPS / XPU). On no-GPU detection,
    logs a WARNING and falls back to ``_compute_fijab_v3`` — same
    graceful-fallback pattern as ``_compute_weights_gpu`` in
    ``mysca.preprocess``.

    `precision` selects the on-device floating-point dtype: ``fp64``
    (default, bit-stable vs the CPU path), ``fp32`` (~2× faster on
    most GPUs, ~7-decimal precision — adequate for routine
    Cij_corr/eigvalsh use), or ``fp16`` (highest throughput on TF32 /
    half-precision tensor cores; ~10⁻³ relative precision —
    numerically risky for downstream eigvalsh on small eigenvalues,
    treat as preview-only).
    """
    from mysca._acceleration import detect_device, resolve_torch_dtype
    import torch

    device = detect_device()
    if device.type == "cpu":
        logger.warning(
            "No GPU device found; falling back to CPU tensordot "
            "(_compute_fijab_v3)."
        )
        return _compute_fijab_v3(xmsa, ws_norm, lam, nsyms)

    dtype = resolve_torch_dtype(precision)

    npos = xmsa.shape[1]
    xf = torch.as_tensor(np.asarray(xmsa, dtype=np.float64),
                         dtype=dtype, device=device)
    ws_t = torch.as_tensor(np.asarray(ws_norm, dtype=np.float64),
                           dtype=dtype, device=device)
    weighted = ws_t[:, None, None] * xf
    fijab = torch.tensordot(weighted, xf, dims=([0], [0])).permute(0, 2, 1, 3)
    fijab = fijab * (1.0 - lam)
    diag = torch.eye(npos, dtype=torch.bool, device=device)
    reg_diag = torch.tensor(lam / nsyms, dtype=dtype, device=device)
    reg_off = torch.tensor(lam / (nsyms * nsyms), dtype=dtype,
                           device=device)
    fijab = torch.where(diag[:, :, None, None],
                        fijab + reg_diag, fijab + reg_off)
    # Always return float64 to the caller — downstream pipeline assumes
    # fp64 for Cij_corr / eigvalsh / bootstrap.
    return fijab.to(torch.float64).cpu().numpy()


def _compute_eigvalsh_bootstrap_gpu(
        xmsa_batch, ws_norm, *, qa, lam, nsyms, precision="fp64",
):
    """Batched GPU bootstrap kernel.

    Takes a stack of one-hot-encoded shuffled MSAs (shape
    ``(B, nseq, npos, naas)``) and returns the descending-sorted
    eigenvalues of each iter's `Cij_corr` (shape ``(B, npos)``).

    Lifts the full per-iter pipeline onto torch:
        fia -> fijab -> Cijab_raw -> phi_ia -> Cijab_corr
            -> Cij_corr -> eigvalsh
    so intermediate tensors stay on the device across the batch.

    Raises:
        RuntimeError: when no GPU is detected. Caller is expected to
            fall back to the per-iter CPU path.
    """
    from mysca._acceleration import detect_device, resolve_torch_dtype
    import torch

    device = detect_device()
    if device.type == "cpu":
        raise RuntimeError(
            "No GPU device found; "
            "_compute_eigvalsh_bootstrap_gpu cannot run."
        )

    dtype = resolve_torch_dtype(precision)
    # eigvalsh requires fp32+ on most backends and is numerically
    # treacherous in fp16; promote to fp32 for the eigendecomposition
    # while keeping the cheap fijab/Cij_corr ops in the chosen dtype.
    eig_dtype = torch.float32 if dtype == torch.float16 else dtype

    B, nseq, npos, naas = xmsa_batch.shape
    xf = torch.as_tensor(
        np.asarray(xmsa_batch, dtype=np.float64),
        dtype=dtype, device=device,
    )
    ws_t = torch.as_tensor(
        np.asarray(ws_norm, dtype=np.float64),
        dtype=dtype, device=device,
    )
    qa_t = torch.as_tensor(
        np.asarray(qa, dtype=np.float64),
        dtype=dtype, device=device,
    )

    # fia[b, p, a] = (1-lam) * sum_s ws_norm[s] * xf[b, s, p, a] + lam/nsyms
    fia = (1.0 - lam) * torch.einsum('s,bspa->bpa', ws_t, xf) + (lam / nsyms)

    # fijab[b, i, j, a, b'] = (1-lam) * sum_s ws_norm[s] * xf[b,s,i,a] * xf[b,s,j,b']
    weighted = ws_t[None, :, None, None] * xf  # (B, S, P, A)
    fijab = torch.einsum('bspa,bsqc->bpqac', weighted, xf)
    fijab = fijab * (1.0 - lam)
    diag = torch.eye(npos, dtype=torch.bool, device=device)
    reg_diag = lam / nsyms
    reg_off = lam / (nsyms * nsyms)
    fijab = torch.where(
        diag[None, :, :, None, None],
        fijab + reg_diag, fijab + reg_off,
    )

    # Cijab_raw[b, i, j, a, b'] = fijab[b, i, j, a, b'] - fia[b, i, a] * fia[b, j, b']
    Cijab_raw = fijab - fia[:, :, None, :, None] * fia[:, None, :, None, :]
    # phi_ia[b, p, a] = log((fia*(1-qa)) / ((1-fia)*qa))
    phi_ia = torch.log((fia * (1.0 - qa_t)) / ((1.0 - fia) * qa_t))
    Cijab_corr = (
        phi_ia[:, :, None, :, None] * phi_ia[:, None, :, None, :] * Cijab_raw
    )
    # Cij_corr[b, i, j] = sqrt(sum_{a,b'} Cijab_corr[b, i, j, a, b']^2)
    Cij_corr = torch.sqrt(torch.sum(Cijab_corr ** 2, dim=(-1, -2)))

    # Batched symmetric eigvalsh; ascending order. Flip to descending.
    # Promote to eig_dtype (fp32 minimum) — eigvalsh on fp16 is unstable
    # and unsupported on most backends.
    evals_asc = torch.linalg.eigvalsh(Cij_corr.to(eig_dtype))  # (B, P)
    evals_desc = torch.flip(evals_asc, dims=[-1])
    # Always return float64 to the caller for consistent downstream
    # numerics (kstar selection, cutoff comparisons).
    return evals_desc.to(torch.float64).cpu().numpy()
