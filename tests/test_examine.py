"""Tests for the mysca.examine subpackage (sca-examine).

Covers the novel per-IC characterization kernels (Pagel's lambda, clade
splits), the structural-contiguity kernels, and the IC -> sector grouping
strategy from Andrews, "How to characterize independent components after
performing SCA". Uses small synthetic inputs so the numerics are deterministic
and no full SCA fixture is required.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from mysca.examine import (
    cross_ic_distances,
    group_ics,
    min_atom_distance_matrix,
    parse_newick,
    per_ic_contiguity,
    run_pagel,
    run_splits,
    touching_pairs,
)


# A balanced 8-leaf tree; one well-separated bipartition {A,B,C,D}|{E,F,G,H}.
_TREE = ("(((A:1,B:1):1,(C:1,D:1):1):1,"
         "((E:1,F:1):1,(G:1,H:1):1):1):0;")


def _acc_to_row():
    # Up_0 cleanly separates the two halves; Up_1 is unstructured noise.
    hi, lo = 10.0, 0.0
    rng = np.random.default_rng(0)
    rows = {}
    for a in "ABCD":
        rows[a] = np.array([hi + rng.normal(0, 0.01), rng.normal()])
    for a in "EFGH":
        rows[a] = np.array([lo + rng.normal(0, 0.01), rng.normal()])
    return rows


class TestPhylogenyKernels:

    def test_pagel_columns_and_signal(self):
        comp_names = ["fam_Up_0", "fam_Up_1"]
        df = run_pagel(parse_newick(_TREE), _acc_to_row(), comp_names,
                       n_null=0)
        assert list(df["component"]) == comp_names
        for col in ("lamP", "LRT", "p_value", "p_adj_BH", "n"):
            assert col in df.columns
        # lambda in [0, 1]; the clade-correlated axis carries strong signal.
        assert ((df["lamP"] >= 0) & (df["lamP"] <= 1)).all()
        assert df.loc[df.component == "fam_Up_0", "LRT"].iloc[0] > 0

    def test_pagel_null_calibration_columns(self):
        comp_names = ["fam_Up_0", "fam_Up_1"]
        aligned = {a: "ACDEFGHIKL" for a in "ABCDEFGH"}
        df = run_pagel(parse_newick(_TREE), _acc_to_row(), comp_names,
                       aligned_seqs=aligned, n_null=20, seed=1)
        assert "lrt_z" in df.columns and "p_lrt_vs_null_BH" in df.columns

    def test_splits_finds_clean_bipartition(self):
        comp_names = ["fam_Up_0", "fam_Up_1"]
        df = run_splits(parse_newick(_TREE), _acc_to_row(), comp_names,
                        min_side_size=2, seed=0)
        row0 = df[df.component == "fam_Up_0"].iloc[0]
        # The {A,B,C,D}|{E,F,G,H} split is a perfect separation -> |delta| ~ 1.
        assert row0["clade_size"] == 4
        assert row0["abs_cliffs_delta"] > 0.99
        # The noise axis should not produce a near-perfect split.
        row1 = df[df.component == "fam_Up_1"].iloc[0]
        assert row1["abs_cliffs_delta"] < row0["abs_cliffs_delta"]


class TestStructureKernels:

    def _line_coords(self, n, spacing=1.0):
        """n residues, each a single atom, spaced ``spacing`` apart on a line."""
        return [np.array([[i * spacing, 0.0, 0.0]]) for i in range(n)]

    def test_distance_matrix(self):
        D = min_atom_distance_matrix(self._line_coords(4))
        assert np.isnan(D[0, 0])
        assert D[0, 1] == pytest.approx(1.0)
        assert D[0, 3] == pytest.approx(3.0)
        assert D[1, 0] == D[0, 1]

    def test_contiguity_compact_vs_spread(self):
        # 8 residues on a line. A compact IC (adjacent) should sit below the
        # size-matched null; a spread IC (every other residue) above it.
        coords = self._line_coords(8)
        D = min_atom_distance_matrix(coords)
        ic_residues = [np.array([0, 1, 2]), np.array([0, 4, 7])]
        df = per_ic_contiguity(ic_residues, D, None, n_null=200, seed=0)
        assert df.loc[0, "mean_nbr_dist"] < df.loc[1, "mean_nbr_dist"]
        assert df.loc[0, "z"] < df.loc[1, "z"]

    def test_cross_and_touching(self):
        coords = self._line_coords(10)
        D = min_atom_distance_matrix(coords)
        ic_residues = [np.array([0, 1]), np.array([2, 3]), np.array([8, 9])]
        cross = cross_ic_distances(ic_residues, D, None, max_err=5.0)
        assert cross.shape == (3, 3)
        # IC0 and IC1 are adjacent (within ~2 A); IC2 is far.
        pairs = {(i, j) for i, j, *_ in touching_pairs(cross, cross_thresh=3.0)}
        assert (0, 1) in pairs
        assert (0, 2) not in pairs


class TestSectorGrouping:

    def _per_ic(self):
        return pd.DataFrame([
            # IC0: conservation-associated core (protected, no edges).
            {"component": "F_Up_0", "ic_idx": 0, "cons_corr_r": 0.7,
             "abs_cliffs_delta": 0.1, "lrt_z": 0.0, "within_dist": 3.0,
             "ic_n_residues": 10},
            # IC1, IC2: reciprocally closer-than-self -> merged sector.
            {"component": "F_Up_1", "ic_idx": 1, "cons_corr_r": 0.1,
             "abs_cliffs_delta": 0.1, "lrt_z": 0.0, "within_dist": 3.0,
             "ic_n_residues": 8},
            {"component": "F_Up_2", "ic_idx": 2, "cons_corr_r": 0.1,
             "abs_cliffs_delta": 0.1, "lrt_z": 0.0, "within_dist": 3.0,
             "ic_n_residues": 7},
            # IC3: one-sided reach into IC1's sector -> co-sector.
            {"component": "F_Up_3", "ic_idx": 3, "cons_corr_r": 0.1,
             "abs_cliffs_delta": 0.1, "lrt_z": 0.0, "within_dist": 3.0,
             "ic_n_residues": 6},
            # IC4: subclade pseudo-sector.
            {"component": "F_Up_4", "ic_idx": 4, "cons_corr_r": 0.1,
             "abs_cliffs_delta": 0.99, "lrt_z": 0.0, "within_dist": 3.0,
             "ic_n_residues": 5},
            # IC5: phylogeny pseudo-sector.
            {"component": "F_Up_5", "ic_idx": 5, "cons_corr_r": 0.1,
             "abs_cliffs_delta": 0.1, "lrt_z": 10.0, "within_dist": 3.0,
             "ic_n_residues": 4},
        ])

    def _cross(self):
        # cross[i, j] = mean nearest distance to IC i over IC j's residues.
        # edge x->y iff cross[idx_y, idx_x] < within_dist[x] (=3 everywhere).
        c = np.full((6, 6), 5.0)
        np.fill_diagonal(c, 3.0)
        c[2, 1] = 1.0   # IC1 -> IC2
        c[1, 2] = 1.0   # IC2 -> IC1   (reciprocal -> merge)
        c[1, 3] = 1.0   # IC3 -> IC1   (one-sided; c[3,1] stays 5 -> co-sector)
        return c

    def test_full_strategy(self):
        feats = group_ics(self._per_ic(), self._cross(), family="F")
        by_comp = {r["member_components"]: r for _, r in feats.iterrows()}
        types = dict(zip(feats["feature_name"], feats["feature_type"]))

        # Merged sector contains IC1 + IC2.
        merged = [r for _, r in feats.iterrows()
                  if r["feature_type"] == "sector" and r["n_ics"] == 2]
        assert len(merged) == 1
        assert set(merged[0]["member_components"].split(",")) == {"F_Up_1", "F_Up_2"}

        # IC0 is its own conservation-associated sector.
        ic0 = by_comp["F_Up_0"]
        assert ic0["feature_type"] == "sector"
        assert "conservation-associated" in ic0["labels"]

        # IC3 is a co-sector pointing at the merged sector.
        ic3 = by_comp["F_Up_3"]
        assert ic3["feature_type"] == "co-sector"
        assert ic3["associated_sector"] == merged[0]["feature_name"]

        # IC4 / IC5 are pseudo-sectors with the right provenance labels.
        assert by_comp["F_Up_4"]["feature_type"] == "pseudo-sector"
        assert "subclade origin" in by_comp["F_Up_4"]["labels"]
        assert "phylogeny origin" in by_comp["F_Up_5"]["labels"]

    def test_no_structure_all_standalone(self):
        # Without a cross matrix, every surviving IC is its own sector.
        feats = group_ics(self._per_ic(), None, family="F")
        survivors = feats[feats["feature_type"] == "sector"]
        assert len(survivors) == 4          # IC0, IC1, IC2, IC3 (IC4/5 pseudo)
        assert (survivors["n_ics"] == 1).all()


@pytest.mark.skipif(shutil.which("famsa") is None
                    and shutil.which("FAMSA") is None,
                    reason="FAMSA not on PATH")
def test_guide_tree_build(tmp_path):
    from mysca.examine.guide_tree import build_guide_tree, write_subsample_fasta
    int2char = np.array(list("-ACDEFGHIKLMNPQRSTVWY"), dtype="<U1")
    # 5 distinct short sequences under synthetic labels.
    rng = np.random.default_rng(0)
    msa = rng.integers(1, 21, size=(5, 12))
    labels = [f"s{i}" for i in range(5)]
    fasta = tmp_path / "sub.fasta"
    tree = tmp_path / "tree.nwk"
    write_subsample_fasta(str(fasta), labels, msa, int2char)
    text = build_guide_tree(str(fasta), str(tree), gt_method="nj")
    parsed = parse_newick(text)
    assert int(parsed["is_leaf"].sum()) == 5
