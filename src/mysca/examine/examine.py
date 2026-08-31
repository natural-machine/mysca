"""Orchestration helpers and the result container for ``sca-examine``.

The per-IC characterisation pipeline has three independent legs — conservation
correlation, phylogeny (Pagel's lambda + clade splits), and structural
contiguity — that are combined into a per-IC table and then grouped into
sectors (:mod:`mysca.examine.sectors`). This module holds the glue that turns a
loaded :class:`~mysca.results.SCAResults` into the inputs each leg needs, plus
:class:`ExamineResults`, which collects the per-IC table, the sector table, and
the supporting intermediates and writes them under an output directory.

Coordinate conventions follow the rest of ``mysca``: ``ic_positions`` are
processed-MSA columns; sequence projections (Uᵖ) come from
:meth:`SCAResults.project_sequences`; structural mapping reuses
:func:`mysca.structure.project_pdb` so IC residues land on the structure the
same robust, alignment-based way ``sca-structure`` does.
"""

import logging
import os

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("mysca.examine")


# ---------------------------------------------------------------------------
# Significant-IC selection + sequence projection
# ---------------------------------------------------------------------------

def n_significant_ics(sca):
    """Number of significant ICs to characterise: ``kstar`` clamped to
    [1, n_components]. ICs are ordered by descending SCA eigenvalue, so the
    first ``kstar`` columns of every per-IC array are exactly the significant
    ones (mysca convention: IC i is significant iff i < kstar)."""
    n_comp = len(sca.ic_positions) if sca.ic_positions is not None else (
        0 if sca.v_ica is None else sca.v_ica.shape[1])
    if n_comp == 0:
        raise ValueError("SCAResults carries no ICs (ic_positions / v_ica).")
    kstar = int(sca.kstar) if sca.kstar is not None else n_comp
    return max(1, min(kstar, n_comp))


def int2char_from_sym2int(sym2int):
    """Build an ``int -> symbol`` lookup array from a ``{symbol: int}`` map
    (the on-disk ``sym2int.json`` format). Index 0 is the gap by mysca
    convention; ``np.where(int2char == "-")`` recovers the gap index."""
    n_sym = max(sym2int.values()) + 1
    int2char = np.empty(n_sym, dtype="<U1")
    for ch, idx in sym2int.items():
        int2char[idx] = ch
    return int2char


def onehot_from_int_msa(msa_int, n_aa):
    """(M, L) ints (0 = gap, 1..n_aa = residues, per sym2int) -> (M, L, n_aa)
    bool one-hot with all-zero rows at gaps. Mirrors the onehot_without_gap
    convention :meth:`SCAResults.project_sequences` expects."""
    M, L = msa_int.shape
    xmsa = np.zeros((M, L, n_aa), dtype=bool)
    rows, cols = np.where(msa_int > 0)
    xmsa[rows, cols, msa_int[rows, cols] - 1] = True
    return xmsa


def project_subsample(sca, msa_int_sub):
    """Uᵖ scores (n_sub x n_components) for an integerized processed MSA."""
    n_aa = sca.fia.shape[1]
    xmsa = onehot_from_int_msa(msa_int_sub, n_aa)
    return sca.project_sequences(xmsa)


def subsample_indices(n_rows, n_target, rng):
    """Sorted row indices for a uniform subsample (all rows if n_rows <=
    n_target). The processed alignment is already redundancy-reduced upstream,
    so uniform sampling is reasonable."""
    if n_target is None or n_rows <= n_target:
        return np.arange(n_rows)
    return np.sort(rng.choice(n_rows, size=n_target, replace=False))


# ---------------------------------------------------------------------------
# Per-IC base table (SCA magnitude + conservation correlation)
# ---------------------------------------------------------------------------

def per_ic_table(sca, kstar, comp_names, *, n_seqs, align_length):
    """Per-IC SCA quantities + conservation correlation for the first
    ``kstar`` ICs.

    Columns: ``component, ic_idx, lambda`` (SCA eigenvalue), ``ic_n_residues,
    ic_frac_residues, align_length, n_seqs, cons_corr_r, cons_corr_p,
    cons_corr_p_adj``. The conservation correlation is Pearson's r between the
    IC's position-wise loading (``v_ica[:, k]``) and per-position conservation
    (relative entropy D_i), with a within-family Benjamini-Hochberg correction.
    """
    conservation = np.asarray(sca.conservation, dtype=np.float64)
    v_ica = np.asarray(sca.v_ica, dtype=np.float64)        # (L, n_components)
    evals_sca = np.asarray(sca.evals_sca, dtype=np.float64)
    groups = sca.ic_positions
    rows = []
    for k in range(kstar):
        r, p = stats.pearsonr(v_ica[:, k], conservation)
        n_res = int(len(groups[k]))
        rows.append({"component": comp_names[k], "ic_idx": k,
                     "lambda": float(evals_sca[k]),
                     "ic_n_residues": n_res,
                     "ic_frac_residues": n_res / align_length,
                     "align_length": align_length, "n_seqs": n_seqs,
                     "cons_corr_r": float(r), "cons_corr_p": float(p)})
    df = pd.DataFrame(rows)
    df["cons_corr_p_adj"] = stats.false_discovery_control(
        df["cons_corr_p"].to_numpy())
    return df


# ---------------------------------------------------------------------------
# Map IC residues onto a structure's distance matrix
# ---------------------------------------------------------------------------

def ic_residues_on_structure(pdb_projection, pdb, kstar):
    """Per-IC 0-based indices into a distance matrix built over
    ``pdb.residue_ids``, for the first ``kstar`` ICs.

    ``pdb_projection`` is a :class:`mysca.structure.PdbProjection`;
    ``ic_pdb_residues[i]`` lists PDB residue *numbers* for IC i. We invert
    ``pdb.residue_ids`` (structure order) to recover the row/column index each
    residue number occupies in the distance matrix. Residue numbers absent from
    the modelled chain are dropped (they have no coordinates).
    """
    resid_to_idx = {int(r): i for i, r in enumerate(pdb.residue_ids)}
    ic_residues = []
    for residues in pdb_projection.ic_pdb_residues[:kstar]:
        idx = np.array(sorted(resid_to_idx[int(r)] for r in residues
                              if int(r) in resid_to_idx), dtype=int)
        ic_residues.append(idx)
    return ic_residues


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ExamineResults:
    """Collected outputs of an ``sca-examine`` run for one family.

    Attributes
    ----------
    family : str
    per_ic : pandas.DataFrame
        One row per significant IC, the merge of the conservation table with
        the Pagel/split statistics and the structural within-IC distance.
    sectors : pandas.DataFrame or None
        Feature table from :func:`mysca.examine.sectors.group_ics`.
    cross : np.ndarray or None
        (K, K) cross-IC neighbour-distance matrix.
    subsample_projections : pandas.DataFrame or None
        label / seq_id / Up_* table for the phylogeny subsample.
    tree_newick : str or None
    structure_meta : dict or None
        Source metadata for the structure used (paths, accession, ...).
    """

    def __init__(self, family, per_ic, *, sectors=None, cross=None,
                 subsample_projections=None, tree_newick=None,
                 structure_meta=None):
        self.family = family
        self.per_ic = per_ic
        self.sectors = sectors
        self.cross = cross
        self.subsample_projections = subsample_projections
        self.tree_newick = tree_newick
        self.structure_meta = structure_meta

    def save(self, outdir):
        """Write every populated output under ``outdir`` (created if needed)."""
        os.makedirs(outdir, exist_ok=True)
        written = []

        per_ic_path = os.path.join(outdir, "per_ic_characterization.tsv")
        self.per_ic.to_csv(per_ic_path, sep="\t", index=False,
                           float_format="%.6g")
        written.append(per_ic_path)

        if self.sectors is not None:
            p = os.path.join(outdir, "sector_features.tsv")
            self.sectors.to_csv(p, sep="\t", index=False, float_format="%.6g")
            written.append(p)
        if self.cross is not None:
            p = os.path.join(outdir, "ic_cross_neighbor_distance.tsv")
            np.savetxt(p, self.cross, delimiter="\t", fmt="%.4g")
            written.append(p)
        if self.subsample_projections is not None:
            p = os.path.join(outdir, "subsample_projections.tsv")
            self.subsample_projections.to_csv(p, sep="\t", index=False,
                                              float_format="%.6g")
            written.append(p)
        if self.tree_newick is not None:
            p = os.path.join(outdir, "guide_tree.nwk")
            with open(p, "w") as fh:
                fh.write(self.tree_newick)
            written.append(p)
        return written
