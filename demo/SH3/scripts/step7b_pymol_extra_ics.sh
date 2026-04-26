#!/usr/bin/env bash
set -euo pipefail

outdir=out/from_msa

# Beyond-kstar demo. SH3 has only ~2 statistically significant ICs
# (kstar=2), but step2_scacore was told to compute --n_components 10
# so the projection carries IC residue lists 0..9. This script renders
# IC groups 0..3 to show what beyond-kstar ICs look like — kstar
# bounds significance, but all computed ICs are projectable and
# renderable.
groups=(0 1 2 3)

if ! python -c "import pymol" >/dev/null 2>&1; then
    echo "[step7b_pymol_extra_ics] pymol not importable; skipping. Install with:" >&2
    echo "  conda install -c conda-forge pymol-open-source" >&2
    exit 0
fi

# Static still per IC group (one PNG each).
sca-pymol \
    --structure ${outdir}/structure \
    --groups "${groups[@]}" \
    -o ${outdir}/pymol_extra_ics

if ! python -c "import imageio, PIL" >/dev/null 2>&1; then
    echo "[step7b_pymol_extra_ics] imageio+PIL not importable; skipping animate pass." >&2
    exit 0
fi

# Combined rotation with all selected ICs lit at once.
sca-pymol \
    --structure ${outdir}/structure \
    --groups "${groups[@]}" \
    --multisector \
    --struct_style cartoon \
    --animate --nframes 36 --duration 3.6 \
    -o ${outdir}/pymol_extra_ics_anim_multi
