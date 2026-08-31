"""Spatial-clustering analysis of SCA independent components on a structure.

Given a structure with each IC's positions mapped onto its residues, this
module measures whether an IC is *contiguous* in 3D (its residues sit close
together, vs a size-matched random null) and how close ICs sit to one another
(cross-neighbour distances), which is the basis for merging ICs into sectors.

Distances are minimum heavy-atom distances between residues. When predicted
aligned error (PAE) is available — e.g. from AlphaFold — residue pairs whose
inter-domain geometry is uncertain (PAE >= ``max_err``) are masked out, so a
spurious cross-domain contact in a low-confidence linker does not make two ICs
look adjacent. PAE is optional: without it, all pairs are used.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

logger = logging.getLogger("mysca.examine.structure")


# ---------------------------------------------------------------------------
# Distance matrix + PAE
# ---------------------------------------------------------------------------

def residue_coords(pdb):
    """Per-residue heavy-atom coordinate arrays for a ``PDBStructure``, in
    ``pdb.residue_ids`` order (the order the distance matrix is indexed by).

    Coordinates are pulled from the underlying Bio.PDB chain by residue number
    so they stay aligned to ``residue_ids`` even when non-standard residues were
    skipped while building the sequence. Raises if a modelled residue id has no
    coordinates (mismatched structure object).
    """
    chain = next(iter(pdb.structure))[pdb.chain_id]
    coord_by_resid = {
        int(res.id[1]): np.array([a.coord for a in res.get_atoms()],
                                 dtype=np.float64)
        for res in chain if res.id[0] == " "
    }
    missing = [int(r) for r in pdb.residue_ids if int(r) not in coord_by_resid]
    if missing:
        raise ValueError(
            f"{len(missing)} residue id(s) in PDBStructure.residue_ids have no "
            f"coordinates (e.g. {missing[:5]}); structure object mismatch.")
    return [coord_by_resid[int(r)] for r in pdb.residue_ids]


def min_atom_distance_matrix(coords):
    """Symmetric (N, N) matrix of minimum inter-residue heavy-atom distances,
    NaN on the diagonal (a residue is not its own neighbour).

    ``coords`` is a length-N list of (n_atoms_i, 3) arrays, one per residue, in
    the order distances are indexed (typically ``PDBStructure.residue_ids``)."""
    n = len(coords)
    D = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        ci = coords[i]
        for j in range(i + 1, n):
            d = float(cdist(ci, coords[j]).min())
            D[i, j] = d
            D[j, i] = d
    return D


def align_pae_to_residues(pae_full, residue_ids):
    """Reindex a full (R, R) PAE matrix (rows/cols = residue number - 1) onto
    the modelled-residue order ``residue_ids``.

    Returns an (N, N) array aligned to the distance matrix, or None when the
    PAE cannot be aligned (residue ids out of range), in which case masking is
    simply skipped by the caller.
    """
    pae_full = np.asarray(pae_full, dtype=np.float64)
    if pae_full.ndim != 2 or pae_full.shape[0] != pae_full.shape[1]:
        logger.warning("PAE is not a square matrix (shape %s); ignoring it.",
                       pae_full.shape)
        return None
    idx = np.asarray(residue_ids, dtype=int) - 1     # residue number -> 0-based
    if idx.min() < 0 or idx.max() >= pae_full.shape[0]:
        logger.warning(
            "PAE matrix (%dx%d) does not cover residue ids %d..%d; "
            "skipping PAE masking.",
            pae_full.shape[0], pae_full.shape[1],
            int(np.asarray(residue_ids).min()), int(np.asarray(residue_ids).max()))
        return None
    return pae_full[np.ix_(idx, idx)]


def _masked(dist_mat, ali_err, max_err):
    """dist_mat with pairs of PAE >= max_err set to NaN. No-op when
    ``ali_err`` is None (PAE unavailable)."""
    if ali_err is None:
        return dist_mat
    return np.where(ali_err < max_err, dist_mat, np.nan)


# ---------------------------------------------------------------------------
# Neighbour-distance kernels
# ---------------------------------------------------------------------------

def nearest_within(set_pos, dist_mat, ali_err, max_err):
    """For each position in ``set_pos``, the distance to its closest neighbour
    within the set (PAE-masked, symmetrised). NaN where no usable neighbour."""
    masked = _masked(dist_mat, ali_err, max_err)
    sub = masked[:, set_pos][set_pos, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sub = np.nanmean(np.array([sub, sub.T]), axis=0)     # symmetrise
        return np.nanmin(sub, axis=0)


def nearest_between(set1, set2, dist_mat, ali_err, max_err):
    """For each position in ``set2``, the distance to its closest neighbour in
    ``set1`` (PAE-masked). Directional: set1 is the target, set2 the query."""
    masked = _masked(dist_mat, ali_err, max_err)
    sub = masked[:, set1][set2, :]                            # (len2, len1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmin(sub, axis=1)


def mean_neighbor_distance(set_pos, dist_mat, ali_err, max_err):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(nearest_within(set_pos, dist_mat, ali_err, max_err)))


# ---------------------------------------------------------------------------
# Per-IC contiguity vs size-matched null
# ---------------------------------------------------------------------------

def per_ic_contiguity(ic_residues, dist_mat, ali_err, *, max_err=5.0,
                      n_null=1000, seed=0):
    """Per-IC mean nearest-neighbour distance vs a size-matched null drawn from
    the pool of all IC positions.

    Returns a DataFrame: ``IC, n_res, mean_nbr_dist, null_mean, null_sd, z,
    p_clustered`` (small distance / negative z = spatially clustered).
    """
    rng = np.random.default_rng(seed)
    pool = np.unique(np.concatenate([np.asarray(r, dtype=int) for r in ic_residues]))
    rows = []
    for k, residues in enumerate(ic_residues):
        residues = np.asarray(residues, dtype=int)
        obs = mean_neighbor_distance(residues, dist_mat, ali_err, max_err)
        nk = len(residues)
        null = np.empty(n_null, dtype=np.float64)
        for b in range(n_null):
            samp = rng.choice(pool, size=min(nk, len(pool)), replace=False)
            null[b] = mean_neighbor_distance(samp, dist_mat, ali_err, max_err)
        nm = float(np.nanmean(null))
        ns = float(np.nanstd(null, ddof=1)) if n_null > 1 else 0.0
        z = (obs - nm) / ns if ns > 0 else np.nan
        p_emp = (1 + int(np.sum(null <= obs))) / (n_null + 1)   # clustered = small
        rows.append({"IC": f"IC{k}", "n_res": nk, "mean_nbr_dist": obs,
                     "null_mean": nm, "null_sd": ns, "z": z, "p_clustered": p_emp})
        logger.info("  IC%d: n=%d obs=%.2f null=%.2f+/-%.2f z=%+.2f p=%.3f",
                    k, nk, obs, nm, ns, z, p_emp)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-IC neighbour distances
# ---------------------------------------------------------------------------

def cross_ic_distances(ic_residues, dist_mat, ali_err, *, max_err=5.0):
    """(K, K) matrix where ``cross[i, j]`` is the mean nearest distance to IC i
    over IC j's residues (so column j shares IC j's residue basis, comparable to
    IC j's own within-IC distance)."""
    K = len(ic_residues)
    cross = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(K):
            l = nearest_between(np.asarray(ic_residues[i], dtype=int),
                                np.asarray(ic_residues[j], dtype=int),
                                dist_mat, ali_err, max_err)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                cross[i, j] = np.nanmean(l)
    return cross


def touching_pairs(cross, cross_thresh):
    """IC pairs whose mean cross-neighbour distance is below ``cross_thresh`` in
    at least one direction. Each entry: (i, j, d_ij, d_ji, kind) where kind is
    'reciprocal' (both directions below) or 'one-sided'."""
    K = cross.shape[0]
    out = []
    for i in range(K):
        for j in range(i + 1, K):
            d_ij = cross[i, j]      # IC_j -> IC_i
            d_ji = cross[j, i]      # IC_i -> IC_j
            below = [np.isfinite(d) and d < cross_thresh for d in (d_ij, d_ji)]
            if any(below):
                kind = "reciprocal" if all(below) else "one-sided"
                out.append((i, j, d_ij, d_ji, kind))
    return out
