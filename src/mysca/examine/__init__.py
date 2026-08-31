"""Characterize SCA independent components (ICs) after a core run.

Given a finished ``sca-core`` result, ``sca-examine`` asks, per IC:

  * Is it conservation-associated? (position loading vs conservation)
  * Is it phylogeny / subclade driven? (Pagel's lambda vs a random-projection
    null; best clade split by Mann-Whitney U + Cliff's delta)
  * Is it a contiguous structural unit, and which ICs sit next to each other?
    (PAE-masked min heavy-atom distances on a supplied or AlphaFold structure)

It then groups the surviving ICs into sectors, co-sectors and pseudo-sectors
following Andrews, "How to characterize independent components after performing
SCA".

Public surface:

    from mysca.examine import (
        parse_newick, run_pagel, run_splits,
        build_guide_tree, fetch_alphafold_structure,
        per_ic_contiguity, cross_ic_distances,
        group_ics, ExamineResults,
    )
"""

from mysca.examine.newick import parse_newick
from mysca.examine.pagel import run_pagel
from mysca.examine.splits import run_splits
from mysca.examine.guide_tree import build_guide_tree, GT_METHODS
from mysca.examine.alphafold import fetch_alphafold_structure, load_pae_json
from mysca.examine.structure import (
    residue_coords,
    min_atom_distance_matrix,
    align_pae_to_residues,
    per_ic_contiguity,
    cross_ic_distances,
    touching_pairs,
)
from mysca.examine.sectors import group_ics
from mysca.examine.examine import (
    ExamineResults,
    n_significant_ics,
    per_ic_table,
    project_subsample,
    subsample_indices,
    int2char_from_sym2int,
    ic_residues_on_structure,
)

__all__ = [
    "parse_newick",
    "run_pagel",
    "run_splits",
    "build_guide_tree",
    "GT_METHODS",
    "fetch_alphafold_structure",
    "load_pae_json",
    "residue_coords",
    "min_atom_distance_matrix",
    "align_pae_to_residues",
    "per_ic_contiguity",
    "cross_ic_distances",
    "touching_pairs",
    "group_ics",
    "ExamineResults",
    "n_significant_ics",
    "per_ic_table",
    "project_subsample",
    "subsample_indices",
    "int2char_from_sym2int",
    "ic_residues_on_structure",
]
