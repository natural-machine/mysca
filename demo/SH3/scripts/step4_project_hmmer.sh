#!/usr/bin/env bash
set -euo pipefail

outdir=out/from_msa

# Primary-sequence projection of human Fyn SH3 (1SHF chain A) onto the
# SH3 SCA result using the HMMER aligner backend. 1SHF is NOT in the
# training MSA, so this exercises both the out-of-sample path and the
# hmmalign backend. Requires `hmmbuild` + `hmmalign` on PATH (install
# via `conda install -c bioconda hmmer`).
project_input=data/pdbs/1SHF.fasta

sca-project \
    -i ${project_input} \
    --preprocessing ${outdir}/preprocessing \
    --scacore ${outdir}/scacore \
    --aligner hmmalign \
    -o ${outdir}/project_hmmer
