"""Pagel's lambda phylogenetic-signal test for SCA independent components.

For each IC (a ``Up_k`` sequence-projection axis) we ask whether the projection
behaves like a trait evolving by Brownian motion along the family guide tree:

  * lambda ~ 1  -> the IC largely tracks phylogeny.
  * lambda ~ 0  -> the IC is independent of the tree (consistent with function).

Because *any* linear projection of an alignment inherits the family's overall
phylogenetic autocorrelation, a large lambda is the expected baseline rather
than IC-specific evidence. We therefore also calibrate each IC's lambda /
likelihood-ratio statistic against a null built from many random linear
projections of the same alignment (Andrews, "How to characterize independent
components after performing SCA"). The LRT, unlike lambda (which saturates at
1.0 and loses per-IC resolution at large n), keeps scaling with how strongly a
trait tracks the tree, and its z-score is the statistic the sector-merging
strategy thresholds on (``Z_LRT > 7`` -> phylogeny-associated).

Lambda is fit by maximum likelihood using Felsenstein's independent contrasts
on the lambda-rescaled tree: O(N) per evaluation, no N*N covariance matrix is
ever formed, so it scales to 10k+ tips.
"""

import logging
import math

import numpy as np
import pandas as pd
from scipy import optimize, stats

from mysca.examine.newick import bh_fdr

logger = logging.getLogger("mysca.examine.pagel")

_EPS = 1e-12
_ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"


# ---------------------------------------------------------------------------
# Geometry precompute (postorder, child lists, root-to-node distances)
# ---------------------------------------------------------------------------

def build_geometry(tree):
    """Precompute everything the per-lambda traversal needs, once."""
    from mysca.examine.newick import iterative_postorder
    children = tree["children"]
    is_leaf = tree["is_leaf"]
    root = tree["root"]
    n = tree["n_nodes"]

    blen = np.array(tree["branch_len"], dtype=np.float64)
    blen[~np.isfinite(blen)] = 0.0          # root + any missing lengths -> 0

    post = iterative_postorder(tree)        # children before parents

    # Root-to-node distance h[node] via preorder (parents before children).
    h = np.zeros(n, dtype=np.float64)
    stack = [root]
    while stack:
        v = stack.pop()
        for c in children[v]:
            h[c] = h[v] + blen[c]
            stack.append(c)

    return {
        "post": post,
        "children": children,
        "is_leaf": is_leaf,
        "blen": blen,
        "h": h,
        "root": root,
        "n_nodes": n,
        "leaf_ids": tree["leaf_ids"],
    }


# ---------------------------------------------------------------------------
# Brownian-motion log-likelihood on the lambda-rescaled tree (contrasts)
# ---------------------------------------------------------------------------

def bm_loglik(lam, geom, y_by_node, n_tips):
    """ML log-likelihood of the data under Brownian motion on the
    Pagel-lambda transformed tree, by Felsenstein's independent contrasts in a
    single post-order pass.

    Pagel's lambda scales off-diagonal phylogenetic covariances by ``lam`` while
    keeping tip variances (root-to-tip distances) fixed. Equivalent per-edge
    transform used here:
        internal edge length  -> lam * original
        terminal edge length  -> original + (1 - lam) * h(parent)
    so each tip's root-to-tip total is preserved and shared paths shrink by lam.
    """
    post = geom["post"]
    children = geom["children"]
    is_leaf = geom["is_leaf"]
    blen = geom["blen"]
    h = geom["h"]

    Xhat = np.empty(geom["n_nodes"], dtype=np.float64)
    Vacc = np.empty(geom["n_nodes"], dtype=np.float64)

    SS = 0.0          # sum of squared standardized contrasts
    logdet = 0.0      # log|C| accumulator

    for node in post:
        if is_leaf[node]:
            Xhat[node] = y_by_node[node]
            Vacc[node] = 0.0
            continue

        ch = children[node]
        hp = h[node]                       # root-to-parent distance for leaves

        def eff_len(c):
            if is_leaf[c]:
                return blen[c] + (1.0 - lam) * hp
            return lam * blen[c]

        first = ch[0]
        Vr = eff_len(first) + Vacc[first]
        Xr = Xhat[first]

        for c in ch[1:]:
            tr = Vr if Vr > _EPS else _EPS
            tc = eff_len(c) + Vacc[c]
            tc = tc if tc > _EPS else _EPS
            contrast = Xr - Xhat[c]
            vcon = tr + tc
            SS += contrast * contrast / vcon
            logdet += math.log(vcon)
            wr = 1.0 / tr
            wc = 1.0 / tc
            Xr = (wr * Xr + wc * Xhat[c]) / (wr + wc)
            Vr = 1.0 / (wr + wc)

        Xhat[node] = Xr
        Vacc[node] = Vr

    logdet += math.log(Vacc[geom["root"]] if Vacc[geom["root"]] > _EPS else _EPS)

    sigma2 = SS / n_tips
    if sigma2 < _EPS:
        sigma2 = _EPS
    return -0.5 * (n_tips * math.log(2.0 * math.pi * sigma2) + logdet + n_tips)


def fit_lambda(geom, y_by_node, n_tips):
    """ML estimate of Pagel's lambda in [0, 1], with logLik at the optimum and
    at the lambda = 0 (no phylogenetic signal) null."""
    res = optimize.minimize_scalar(
        lambda lam: -bm_loglik(lam, geom, y_by_node, n_tips),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-5},
    )
    lam_hat = float(res.x)
    ll_hat = bm_loglik(lam_hat, geom, y_by_node, n_tips)
    ll_null = bm_loglik(0.0, geom, y_by_node, n_tips)
    # Guard: the optimizer can stop just shy of a boundary optimum.
    ll_one = bm_loglik(1.0, geom, y_by_node, n_tips)
    if ll_one > ll_hat:
        lam_hat, ll_hat = 1.0, ll_one
    if ll_null > ll_hat:
        lam_hat, ll_hat = 0.0, ll_null
    return lam_hat, ll_hat, ll_null


# ---------------------------------------------------------------------------
# Van der Waerden / normal-scores transform + random-projection null encoding
# ---------------------------------------------------------------------------

def van_der_waerden(x):
    """Rank-based inverse-normal transform: ``norm.ppf(rank / (n + 1))``."""
    n = len(x)
    ranks = stats.rankdata(x, method="average")
    return stats.norm.ppf(ranks / (n + 1.0))


def build_code_matrix(seqs):
    """Encode aligned sequences as a uint8 matrix (n x L) of symbol indices
    over ``_ALPHABET`` (+1 catch-all). A random projection of the implied
    one-hot tensor is then ``sum_pos R[pos, code[:, pos]]`` for a random R of
    shape (L, n_symbols)."""
    n = len(seqs)
    L = len(seqs[0])
    for s in seqs:
        if len(s) != L:
            raise ValueError("Aligned sequences differ in length; not an alignment.")
    other = len(_ALPHABET)                      # index for symbols not in alphabet
    lut = np.full(256, other, dtype=np.uint8)
    for idx, ch in enumerate(_ALPHABET):
        lut[ord(ch)] = idx
        lut[ord(ch.lower())] = idx
    lut[ord(".")] = _ALPHABET.index("-")        # '.' is a gap/insert variant
    code = np.empty((n, L), dtype=np.uint8)
    for i, s in enumerate(seqs):
        code[i] = lut[np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)]
    return code, other + 1


def _null_lambda_lrt(geom, leaf_ids, code, n_symbols, n_tips, n_null,
                     transform, rng):
    """Per random alignment projection, fit lambda AND record the lambda>0
    likelihood-ratio statistic, for calibrating an IC against the generic
    phylogenetic autocorrelation of any sequence projection."""
    L = code.shape[1]
    col_idx = np.arange(L)
    y_by_node = np.zeros(geom["n_nodes"], dtype=np.float64)
    lams = np.empty(n_null, dtype=np.float64)
    lrts = np.empty(n_null, dtype=np.float64)
    for d in range(n_null):
        R = rng.standard_normal((L, n_symbols))
        proj = R[col_idx, code].sum(axis=1)
        vals = van_der_waerden(proj) if transform else proj
        y_by_node[leaf_ids] = vals
        _lam, ll_hat, ll_null = fit_lambda(geom, y_by_node, n_tips)
        lams[d] = _lam
        lrts[d] = max(0.0, 2.0 * (ll_hat - ll_null))
        if (d + 1) % 50 == 0 or (d + 1) == n_null:
            logger.debug("null draw %d/%d", d + 1, n_null)
    return lams, lrts


# ---------------------------------------------------------------------------
# Per-IC driver
# ---------------------------------------------------------------------------

def run_pagel(tree, acc_to_row, comp_names, *, transform=True,
              boundary_correction=True, aligned_seqs=None, n_null=0, seed=0):
    """Fit Pagel's lambda for every IC on ``tree``.

    Parameters
    ----------
    tree : dict
        Parsed Newick (``mysca.examine.newick.parse_newick``). Pruned in place
        to the leaves shared with ``acc_to_row``.
    acc_to_row : dict[str, np.ndarray]
        Accession/label -> length-K array of that sequence's IC projections.
    comp_names : list[str]
        One name per IC column (e.g. ``"PF00001_Up_0"``), length K.
    transform : bool
        Apply the van der Waerden normal-scores transform (default True).
    boundary_correction : bool
        Use the ``0.5*chi2(1)`` boundary mixture for the LRT p-value
        (lambda = 0 sits on the parameter boundary).
    aligned_seqs : dict[str, str] or None
        Accession/label -> gapped processed-coordinate sequence. Required when
        ``n_null > 0`` (feeds the random-projection null).
    n_null : int
        Random alignment projections used to calibrate each IC's lambda/LRT
        (0 disables).
    seed : int
        RNG seed for the null draws.

    Returns
    -------
    pandas.DataFrame
        One row per IC: ``component, lamP, LRT, p_value, p_adj_BH, n`` plus,
        when ``n_null > 0``, the ``lamP_*`` / ``lrt_*`` null-calibration columns
        (notably ``lrt_z``, thresholded by the sector strategy).
    """
    from mysca.examine.newick import (
        extract_accession_from_leaf,
        prune_to_accessions,
    )
    tree_acc = {extract_accession_from_leaf(tree["name"][i])
                for i in tree["leaf_ids"]}
    common = tree_acc & set(acc_to_row)
    if len(common) < 3:
        raise ValueError("Need >=3 shared sequences for Pagel's lambda.")

    prune_to_accessions(tree, common)
    geom = build_geometry(tree)
    leaf_ids = geom["leaf_ids"]
    n_tips = len(leaf_ids)
    leaf_acc = [extract_accession_from_leaf(tree["name"][i]) for i in leaf_ids]
    proj = np.array([acc_to_row[a] for a in leaf_acc], dtype=np.float64)

    rows = []
    y_by_node = np.zeros(geom["n_nodes"], dtype=np.float64)
    for k, col in enumerate(comp_names):
        vals = proj[:, k]
        if transform:
            vals = van_der_waerden(vals)
        y_by_node[leaf_ids] = vals
        lam, ll_hat, ll_null = fit_lambda(geom, y_by_node, n_tips)
        lrt = 2.0 * (ll_hat - ll_null)
        if lrt <= 0:
            pval = 1.0
        elif boundary_correction:
            pval = 0.5 * float(stats.chi2.sf(lrt, df=1))
        else:
            pval = float(stats.chi2.sf(lrt, df=1))
        # lamP = Pagel's lambda, named to avoid clashing with the SCA component
        # magnitude "lambda" (an eigenvalue) reported elsewhere.
        rows.append({"component": col, "lamP": lam, "LRT": lrt,
                     "p_value": pval, "n": n_tips})
        logger.info("  %s: lamP=%.4f LRT=%.2f p=%.3e", col, lam, lrt, pval)

    out = pd.DataFrame(rows)
    out["p_adj_BH"] = bh_fdr(out["p_value"].to_numpy())

    if n_null > 0:
        if aligned_seqs is None:
            raise ValueError("null projections requested but aligned_seqs is None")
        seqs = [aligned_seqs[a] for a in leaf_acc]   # leaf_ids order
        code, n_symbols = build_code_matrix(seqs)
        rng = np.random.default_rng(seed)
        logger.info("  calibrating against %d random projections "
                    "(L=%d, %d symbols)...", n_null, code.shape[1], n_symbols)
        null_lams, null_lrts = _null_lambda_lrt(
            geom, leaf_ids, code, n_symbols, n_tips, n_null, transform, rng)

        # --- lambda vs null (saturating; kept as a diagnostic) --------------
        nm = float(null_lams.mean())
        ns = float(null_lams.std(ddof=1)) if n_null > 1 else 0.0
        lam_obs = out["lamP"].to_numpy()
        out["lamP_null_mean"] = nm
        out["lamP_null_sd"] = ns
        out["lamP_z"] = (lam_obs - nm) / ns if ns > 0 else np.nan
        out["p_vs_null"] = [(1 + int(np.sum(null_lams >= lo))) / (n_null + 1)
                            for lo in lam_obs]
        out["p_vs_null_BH"] = bh_fdr(out["p_vs_null"].to_numpy())

        # --- LRT vs null (non-saturating; the discriminating statistic) -----
        lm = float(null_lrts.mean())
        ls = float(null_lrts.std(ddof=1)) if n_null > 1 else 0.0
        lrt_obs = out["LRT"].to_numpy()
        out["lrt_null_mean"] = lm
        out["lrt_null_sd"] = ls
        out["lrt_z"] = (lrt_obs - lm) / ls if ls > 0 else np.nan
        out["p_lrt_vs_null"] = [(1 + int(np.sum(null_lrts >= lo))) / (n_null + 1)
                                for lo in lrt_obs]
        out["p_lrt_vs_null_BH"] = bh_fdr(out["p_lrt_vs_null"].to_numpy())
        logger.info("  null LRT: mean=%.2f sd=%.2f max=%.2f",
                    lm, ls, float(null_lrts.max()))
    return out
