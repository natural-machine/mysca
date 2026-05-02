#!/usr/bin/env bash
set -euo pipefail

outdir=out/from_msa

# Demonstrate sca-pymol's --features_py / --features plumbing on the
# SH3 demo structures. Unlike the NarG (1Q16) case, neither 1SHF nor
# 2ABL has a bound cofactor — these features highlight the canonical
# *apo* PXXP peptide-binding surface (hydrophobic pocket + RT, n-Src,
# and distal loops) where SH3 sectors typically map.
#
# The features files live at demo/pymol_features/sh3_1shf.py and
# demo/pymol_features/sh3_2abl.py; selectors are PDB-specific because
# 1SHF (Fyn) and 2ABL (Abl) use different residue numbering.

if ! python -c "import pymol" >/dev/null 2>&1; then
    echo "[step7c_pymol_features] pymol not importable; skipping. Install with:" >&2
    echo "  conda install -c conda-forge pymol-open-source" >&2
    exit 0
fi

# Render IC 0 + IC 1 on Fyn SH3 (1SHF) with the apo binding-surface
# features overlaid.
sca-pymol \
    --structure ${outdir}/structure \
    --structure_id Fyn_1SHF \
    --groups 0 1 \
    --features_py ../pymol_features/sh3_1shf.py \
    --features show_pxxp_pocket,show_rt_loop,show_n_src_loop,show_specificity_tyr \
    -o ${outdir}/pymol_features_1SHF

# Same overlay on Abl SH3 (2ABL).
sca-pymol \
    --structure ${outdir}/structure \
    --structure_id Abl1_2ABL \
    --groups 0 1 \
    --features_py ../pymol_features/sh3_2abl.py \
    --features show_pxxp_pocket,show_rt_loop,show_n_src_loop,show_specificity_tyr \
    -o ${outdir}/pymol_features_2ABL
