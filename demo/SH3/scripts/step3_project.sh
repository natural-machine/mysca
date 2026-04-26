#!/usr/bin/env bash
set -euo pipefail

outdir=out/from_msa
reference='4837_jgi||3708||Equilibrative'

# Project the named reference sequence back onto its own SCA result.
# --from_msa pulls the record out of the training MSA, ungaps it, and
# feeds it to the in-sample short-circuit (no alignment at runtime).
sca-project \
    --from_msa ${outdir}/preprocessing/msa_orig.fasta-aln \
    --seq_id "${reference}" \
    --preprocessing ${outdir}/preprocessing \
    --scacore ${outdir}/scacore \
    -o ${outdir}/project
