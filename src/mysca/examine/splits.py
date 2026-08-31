"""Best clade-split test for SCA independent components.

For each IC (``Up_k`` axis) we find the branch in the family guide tree whose
removal best separates leaves into two clades along that axis, judged by a
Mann-Whitney U two-sided test, and report the effect size as Cliff's delta
(``|delta|`` in [0, 1]). A large effect means a single subclade dominates the
IC's tail — evidence the IC is a subclade artifact rather than a distributed
functional signal. The sector-merging strategy thresholds ``|delta| > 0.95``
to mark an IC subclade-associated.

Algorithm (guaranteed optimum, O(N*K)):

  1. Re-root at an arbitrary matched leaf so every non-root, non-unifurcation
     node maps 1:1 to a unique unrooted bipartition (2N-3 total).
  2. Rank projection values per axis (tie-corrected).
  3. One iterative post-order pass accumulates per-subtree validity counts and
     sums of (values, ranks).
  4. Vectorized MWU across all candidate splits and axes at once.
  5. Per-axis best split, with Benjamini-Hochberg FDR over (splits * axes).
"""

import hashlib
import logging

import numpy as np
import pandas as pd
from scipy import stats

from mysca.examine.newick import (
    bh_fdr,
    descendants_of,
    extract_accession_from_leaf,
    iterative_postorder,
    reroot_at_leaf,
    sample_accessions,
)

logger = logging.getLogger("mysca.examine.splits")


def build_leaf_value_matrix(tree, acc_to_row, n_axes):
    """(X, valid, tree_accs, matched): per-leaf projection values and a mask of
    which leaves matched an accession in ``acc_to_row``."""
    leaf_ids = tree["leaf_ids"]
    names = tree["name"]
    n_leaves = len(leaf_ids)
    X = np.zeros((n_leaves, n_axes), dtype=np.float64)
    valid = np.zeros((n_leaves, n_axes), dtype=bool)
    tree_accs = np.empty(n_leaves, dtype=object)
    matched = 0
    seen_acc = set()
    dup_acc = []
    for i, leaf_id in enumerate(leaf_ids):
        acc = extract_accession_from_leaf(names[leaf_id])
        tree_accs[i] = acc
        if acc in seen_acc:
            dup_acc.append(acc)
        else:
            seen_acc.add(acc)
        row = acc_to_row.get(acc)
        if row is not None:
            X[i] = row
            valid[i] = True
            matched += 1
    if dup_acc:
        raise ValueError(
            f"Duplicate accessions among tree leaves (first 5): {dup_acc[:5]}")
    return X, valid, tree_accs, matched


def rank_per_axis(X, valid):
    """Per-axis average-rank of valid values. Invalid rows get rank 0.
    Returns (R, T_k, N_valid) with T_k = sum(t**3 - t) over tie groups."""
    n_leaves, K = X.shape
    R = np.zeros_like(X)
    T = np.zeros(K, dtype=np.float64)
    N_valid = np.zeros(K, dtype=np.int64)
    for k in range(K):
        m = valid[:, k]
        vals = X[m, k]
        if vals.size:
            R[m, k] = stats.rankdata(vals, method="average")
            _, counts = np.unique(vals, return_counts=True)
            T[k] = np.sum(counts ** 3 - counts)
        N_valid[k] = m.sum()
    return R, T, N_valid


def accumulate_subtree_stats(tree, postorder, X, R, valid):
    n_nodes = tree["n_nodes"]
    children = tree["children"]
    leaf_ids = tree["leaf_ids"]
    K = X.shape[1]

    n_val = np.zeros((n_nodes, K), dtype=np.int64)
    sum_x = np.zeros((n_nodes, K), dtype=np.float64)
    sum_r = np.zeros((n_nodes, K), dtype=np.float64)

    leaf_idx_of_node = -np.ones(n_nodes, dtype=np.int64)
    leaf_idx_of_node[leaf_ids] = np.arange(len(leaf_ids))

    leaf_mask = leaf_idx_of_node >= 0
    n_val[leaf_mask] = valid[leaf_idx_of_node[leaf_mask]].astype(np.int64)
    sum_x[leaf_mask] = np.where(valid[leaf_idx_of_node[leaf_mask]],
                                X[leaf_idx_of_node[leaf_mask]], 0.0)
    sum_r[leaf_mask] = R[leaf_idx_of_node[leaf_mask]]

    for v in postorder:
        if leaf_idx_of_node[v] >= 0:
            continue
        for c in children[v]:
            n_val[v] += n_val[c]
            sum_x[v] += sum_x[c]
            sum_r[v] += sum_r[c]
    return n_val, sum_x, sum_r


def candidate_nodes(tree):
    """Non-root, non-unifurcation nodes. Each maps to a unique unrooted
    bipartition (2*n_leaves - 3 for a binary tree)."""
    new_root = tree["root"]
    children = tree["children"]
    n = tree["n_nodes"]
    cand = [v for v in range(n) if v != new_root and len(children[v]) != 1]
    return np.asarray(cand, dtype=np.int32)


def evaluate_all_splits(node_ids, n_val, sum_x, sum_r,
                        N_total, sum_x_total, T_k, min_side_size):
    """MWU statistic and two-sided asymptotic p-value for every (candidate
    node, axis) pair. Returns a dict of (m, K) arrays."""
    nA = n_val[node_ids].astype(np.float64)               # (m, K)
    sxA = sum_x[node_ids]
    srA = sum_r[node_ids]

    nB = N_total.astype(np.float64) - nA                   # (m, K)
    nT = N_total.astype(np.float64)                        # (K,)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_A = sxA / np.where(nA > 0, nA, np.nan)
        mean_B = (sum_x_total - sxA) / np.where(nB > 0, nB, np.nan)

        U_A = srA - nA * (nA + 1.0) / 2.0
        var_U = (nA * nB / 12.0) * ((nT + 1.0) - T_k / (nT * (nT - 1.0)))
        mean_U = nA * nB / 2.0
        Z = (U_A - mean_U) / np.sqrt(var_U)
        p_mwu = 2.0 * stats.norm.sf(np.abs(Z))

    valid_split = (nA >= min_side_size) & (nB >= min_side_size)
    p_mwu = np.where(valid_split, p_mwu, np.nan)
    Z = np.where(valid_split, Z, np.nan)
    U_A = np.where(valid_split, U_A, np.nan)
    return {"node_ids": node_ids, "nA": nA, "nB": nB,
            "mean_A": mean_A, "mean_B": mean_B,
            "U_A": U_A, "Z": Z, "p_mwu": p_mwu, "valid": valid_split}


def bipartition_id(tree, smaller_side_leaf_node_ids):
    """Content-based ID for a bipartition: 12-hex MD5 of the comma-joined
    sorted accessions on the smaller side. Two axes whose optimal splits
    separate the same sequences share an ID even if the chosen node differs."""
    names = tree["name"]
    acc = sorted(extract_accession_from_leaf(names[int(n)])
                 for n in smaller_side_leaf_node_ids)
    return hashlib.md5(",".join(acc).encode("utf-8")).hexdigest()[:12]


def run_splits(tree, acc_to_row, comp_names, *, min_side_size=10, seed=0):
    """Best clade split per IC on ``tree``.

    Parameters mirror :func:`mysca.examine.pagel.run_pagel`. Returns a
    DataFrame with one row per IC that has a valid split: ``component,
    bipartition_id, clade_size, complement_size, mean_clade, mean_complement,
    abs_cliffs_delta, Z, p_mwu, q_bh, rep_clade``.
    """
    rng = np.random.default_rng(seed)
    K = len(comp_names)

    X, valid, _accs, matched = build_leaf_value_matrix(tree, acc_to_row, K)
    if matched < 3:
        raise ValueError("Need >=3 shared sequences for split analysis.")

    matched_leaf_idx = int(np.argmax(valid.any(axis=1)))
    reroot_at_leaf(tree, int(tree["leaf_ids"][matched_leaf_idx]))

    R, T_k, _N_valid = rank_per_axis(X, valid)
    N_valid = valid.sum(axis=0)
    sum_x_total = (X * valid).sum(axis=0)
    postorder = iterative_postorder(tree)
    n_val, sum_x, sum_r = accumulate_subtree_stats(tree, postorder, X, R, valid)
    cand = candidate_nodes(tree)
    res = evaluate_all_splits(cand, n_val, sum_x, sum_r,
                              N_valid, sum_x_total, T_k, min_side_size)
    p_mwu = res["p_mwu"]                                   # (m, K)
    q_bh = bh_fdr(p_mwu.ravel()).reshape(p_mwu.shape)

    rows = []
    for k in range(K):
        col = p_mwu[:, k]
        if not np.any(np.isfinite(col)):
            logger.info("  %s: no valid split", comp_names[k])
            continue
        idx = int(np.nanargmin(col))
        node_id = int(cand[idx])
        nA = int(res["nA"][idx, k]); nB = int(res["nB"][idx, k])
        mA = float(res["mean_A"][idx, k]); mB = float(res["mean_B"][idx, k])
        U = float(res["U_A"][idx, k]); Z = float(res["Z"][idx, k])
        cliffs_delta_A = 2.0 * U / (nA * nB) - 1.0
        under = set(descendants_of(tree, node_id))
        complement = [l for l in (int(x) for x in tree["leaf_ids"]) if l not in under]
        if nA <= nB:
            clade_leaves, clade_size, comp_size = list(under), nA, nB
            mean_clade, mean_comp, cliffs = mA, mB, cliffs_delta_A
        else:
            clade_leaves, clade_size, comp_size = complement, nB, nA
            mean_clade, mean_comp, cliffs = mB, mA, -cliffs_delta_A
        rows.append({
            "component": comp_names[k],
            "bipartition_id": bipartition_id(tree, clade_leaves),
            "clade_size": clade_size,
            "complement_size": comp_size,
            "mean_clade": mean_clade,
            "mean_complement": mean_comp,
            # The sign of Cliff's delta only reflects the arbitrary clade-vs-
            # complement orientation, so report magnitude (mean_clade /
            # mean_complement still carry the direction if needed).
            "abs_cliffs_delta": abs(cliffs),
            "Z": Z,
            "p_mwu": float(res["p_mwu"][idx, k]),
            "q_bh": float(q_bh[idx, k]),
            "rep_clade": ";".join(sample_accessions(tree, clade_leaves, 5, rng)),
        })
        logger.info("  %s: clade=%d/%d |delta|=%.3f p=%.3e",
                    comp_names[k], clade_size, comp_size, abs(cliffs),
                    float(res["p_mwu"][idx, k]))
    return pd.DataFrame(rows)
