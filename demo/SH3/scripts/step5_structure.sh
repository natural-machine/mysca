#!/usr/bin/env bash
set -euo pipefail

outdir=out/from_msa

# PDB-level projection: 1SHF (human Fyn SH3 domain, chain A).
# sca-structure aligns the PDB's primary sequence to the reference MSA
# (default: mafft --add --keeplength) and then composes with the PDB's
# residue_ids to produce per-IC lists of PDB residue numbers. 1SHF has
# residues 84-142, which is a concrete demo of why raw-residue-index
# != PDB residue number in the general case.
sca-structure \
    -s data/pdbs/1SHF.pdb \
    --chain A \
    --preprocessing ${outdir}/preprocessing \
    --scacore ${outdir}/scacore \
    -o ${outdir}/structure
