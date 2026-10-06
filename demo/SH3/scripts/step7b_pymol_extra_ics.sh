#!/usr/bin/env bash
set -euo pipefail

outdir=out/from_msa

# Beyond-kstar demo. SH3 has only ~2 statistically significant ICs
# (kstar=2), and step2_scacore computes just those. This script reruns
# sca-core with --n_components 10 into its own directory, so the
# projection carries IC residue lists 0..9, and renders IC groups 0..3
# to show what beyond-kstar ICs look like — kstar bounds significance,
# but all computed ICs are projectable and renderable.
#
# sca-core warns about this run, as intended: ICA is solved jointly
# over all 10 eigenvectors, so ICs 0 and 1 here are not the ICs 0 and 1
# of step2_scacore.
groups=(0 1 2 3)

if ! python -c "import pymol" >/dev/null 2>&1; then
    echo "[step7b_pymol_extra_ics] pymol not importable; skipping. Install with:" >&2
    echo "  conda install -c conda-forge pymol-open-source" >&2
    exit 0
fi

sca-core \
    -i ${outdir}/preprocessing \
    -o ${outdir}/scacore_extra_ics \
    --regularization 0.03 \
    --n_components 10 \
    --seed 42

sca-structure \
    --seq_map data/structures.tsv \
    --preprocessing ${outdir}/preprocessing \
    --scacore ${outdir}/scacore_extra_ics \
    -o ${outdir}/structure_extra_ics

# Static still per IC group (one PNG each).
sca-pymol \
    --structure ${outdir}/structure_extra_ics \
    --groups "${groups[@]}" \
    -o ${outdir}/pymol_extra_ics

if ! python -c "import imageio, PIL" >/dev/null 2>&1; then
    echo "[step7b_pymol_extra_ics] imageio+PIL not importable; skipping animate pass." >&2
    exit 0
fi

# Combined rotation with all selected ICs lit at once.
sca-pymol \
    --structure ${outdir}/structure_extra_ics \
    --groups "${groups[@]}" \
    --multisector \
    --struct_style cartoon \
    --animate --nframes 36 --duration 3.6 \
    -o ${outdir}/pymol_extra_ics_anim_multi
