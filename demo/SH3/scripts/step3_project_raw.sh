#!/usr/bin/env bash
set -euo pipefail

outdir=out/from_raw

# The raw path has no named reference. Pick the first sequence from the
# mafft-aligned output and project it back in-sample.
aligned=${outdir}/prealign/aligned.fasta
mkdir -p ${outdir}/project
project_input=${outdir}/project/input.fasta

first_id=$(awk '/^>/ { print substr($0, 2); exit }' ${aligned})
if [ -z "${first_id}" ]; then
    echo "error: could not read an ID from ${aligned}" >&2
    exit 1
fi

raw_seq=$(awk -v target=">${first_id}" '
    /^>/ { want = ($0 == target); next }
    want { printf "%s", $0 }
' ${aligned} | tr -d '-')

if [ -z "${raw_seq}" ]; then
    echo "error: sequence ${first_id} not found in ${aligned}" >&2
    exit 1
fi

printf ">%s\n%s\n" "${first_id}" "${raw_seq}" > ${project_input}

sca-project \
    -i ${project_input} \
    --preprocessing ${outdir}/preprocessing \
    --scacore ${outdir}/scacore \
    -o ${outdir}/project
