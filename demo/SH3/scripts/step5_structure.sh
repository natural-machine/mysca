#!/usr/bin/env bash
set -euo pipefail

outdir=out/from_msa

# PDB-level projection: 2ABL (human Abl1 SH3-SH2 fragment, chain A).
# sca-structure aligns the PDB's primary sequence to the reference MSA
# (default: mafft --add --keeplength) and then composes with the PDB's
# residue_ids to produce per-IC lists of PDB residue numbers. 2ABL has
# residues 75-237 (SH3 + SH2), which is a concrete demo of why
# raw-residue-index != PDB residue number in the general case.
sca-structure \
    -s data/pdbs/2ABL.pdb \
    --chain A \
    --preprocessing ${outdir}/preprocessing \
    --scacore ${outdir}/scacore \
    -o ${outdir}/structure
