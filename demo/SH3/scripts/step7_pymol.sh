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
