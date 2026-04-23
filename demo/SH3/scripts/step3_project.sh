#!/usr/bin/env bash
set -euo pipefail

outdir=out/from_msa

infile=data/msas/SH3_demo_MSA_1.afa
reference='4837_jgi||3708||Equilibrative'

# Extract the reference sequence's gapless primary sequence from the MSA
# and write it as a one-record FASTA for sca-project to consume. Since
# the reference ID is in msa_obj_orig, sca-project will take the
# in-sample short-circuit (no mafft call at runtime).
mkdir -p ${outdir}/project
project_input=${outdir}/project/input.fasta

raw_seq=$(awk -v target=">${reference}" '
    /^>/ { want = ($0 == target); next }
    want { printf "%s", $0 }
' ${infile} | tr -d '-')

if [ -z "${raw_seq}" ]; then
    echo "error: reference sequence ${reference} not found in ${infile}" >&2
    exit 1
fi

printf ">%s\n%s\n" "${reference}" "${raw_seq}" > ${project_input}

sca-project \
    -i ${project_input} \
    --preprocessing ${outdir}/preprocessing \
    --scacore ${outdir}/scacore \
    -o ${outdir}/project
