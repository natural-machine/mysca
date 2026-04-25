#!/usr/bin/env bash
set -euo pipefail

outdir=out/from_msa

# sca-pymol requires the optional pymol-open-source dependency. If it's
# not importable in the active env, skip this step with a hint rather
# than failing the whole demo. Install it via:
#   conda install -c conda-forge pymol-open-source
if ! python -c "import pymol" >/dev/null 2>&1; then
    echo "[step7_pymol] pymol not importable; skipping. Install with:" >&2
    echo "  conda install -c conda-forge pymol-open-source" >&2
    exit 0
fi

# Render the top 2 ICs from the 1SHF structure projection produced by
# step5_structure. sca-pymol loads the PDB via the pdb_path recorded
# in structure_projection.json and uses authoritative PDB residue
# numbers from ic_pdb_residues — no --pdb_dir / --modes / 1+idx
# fudging.
sca-pymol \
    --structure ${outdir}/structure \
    --groups 0 1 \
    --views \
    -o ${outdir}/pymol

# Animated passes: one GIF per IC group (default mode), plus one GIF
# covering both groups lit up together (--multisector). Requires
# imageio + PIL (both are deps of pymol-open-source on conda-forge,
# but guard anyway so a minimal install doesn't break the demo
# mid-run).
if ! python -c "import imageio, PIL" >/dev/null 2>&1; then
    echo "[step7_pymol] imageio+PIL not importable; skipping animate pass." >&2
    echo "  pip install imageio pillow" >&2
    exit 0
fi

# Per-group rotations: one GIF each for IC 0 and IC 1.
sca-pymol \
    --structure ${outdir}/structure \
    --groups 0 1 \
    --animate \
    -o ${outdir}/pymol_anim

# Combined rotation: single GIF with both ICs lit up at once.
sca-pymol \
    --structure ${outdir}/structure \
    --groups 0 1 \
    --multisector \
    --animate \
    -o ${outdir}/pymol_anim_multi

# MP4 pass: same combined rotation also written as MP4 alongside the
# GIF. Requires the optional imageio-ffmpeg dep; skip gracefully if
# the env doesn't have it.
if python -c "import imageio_ffmpeg" >/dev/null 2>&1; then
    sca-pymol \
        --structure ${outdir}/structure \
        --groups 0 1 \
        --multisector \
        --animate \
        --format both \
        -o ${outdir}/pymol_anim_mp4
else
    echo "[step7_pymol] imageio-ffmpeg not importable; skipping mp4 pass." >&2
    echo "  pip install imageio-ffmpeg  (or: pip install -e '.[mp4]')" >&2
fi

# Reveal-mode passes: still-camera narrative animations that walk the
# top 2 ICs through stages of which groups are visible. Three sub-
# modes exercised — cumulative (groups stack), sequential (one at a
# time), and custom (explicit stage list).
sca-pymol \
    --structure ${outdir}/structure \
    --groups 0 1 \
    --animate --mode reveal --reveal_schedule cumulative \
    -o ${outdir}/pymol_reveal_cum

sca-pymol \
    --structure ${outdir}/structure \
    --groups 0 1 \
    --animate --mode reveal --reveal_schedule sequential \
    -o ${outdir}/pymol_reveal_seq

sca-pymol \
    --structure ${outdir}/structure \
    --groups 0 1 \
    --animate --mode reveal --reveal_schedule custom \
    --reveal_custom "0" "0,1" "1" \
    -o ${outdir}/pymol_reveal_custom
