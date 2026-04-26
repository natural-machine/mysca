#!/usr/bin/env bash
set -euo pipefail

outdir=out/from_raw
aligned=${outdir}/prealign/aligned.fasta

# The raw path has no named reference. Pick the first record's ID
# (Biopython rec.id convention: first whitespace-delimited token of
# the header) and project it back in-sample via --from_msa.
first_id=$(awk '/^>/ { print substr($1, 2); exit }' ${aligned})
if [ -z "${first_id}" ]; then
    echo "error: could not read an ID from ${aligned}" >&2
    exit 1
fi

sca-project \
    --from_msa ${aligned} \
    --seq_id "${first_id}" \
    --preprocessing ${outdir}/preprocessing \
    --scacore ${outdir}/scacore \
    -o ${outdir}/project
