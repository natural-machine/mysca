#!/usr/bin/env bash
set -euo pipefail

outdir=out/from_msa

# PDB-level projection of two SH3 structures in a single batch via
# --seq_map: 1SHF (human Fyn SH3, residues 84-142) and 2ABL (human
# Abl1 SH3-SH2 fragment, residues 75-237). sca-structure aligns each
# PDB's primary sequence to the reference MSA (default: mafft --add
# --keeplength) and composes with each PDB's residue_ids, producing
# one per-IC list of PDB residue numbers per structure. Two scaffolds
# in different residue-numbering frames is a concrete demo of why
# raw-residue-index != PDB residue number in the general case.
sca-structure \
    --seq_map data/structures.tsv \
    --preprocessing ${outdir}/preprocessing \
    --scacore ${outdir}/scacore \
    -o ${outdir}/structure
