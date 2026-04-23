#!/usr/bin/env bash

outdir=out/from_raw

sca-preprocess \
    -i ${outdir}/prealign/aligned.fasta \
    -o ${outdir}/preprocessing \
    --gap_truncation_thresh 0.4 \
    --sequence_gap_thresh 0.2 \
    --reference_similarity_thresh 0.2 \
    --sequence_similarity_thresh 0.8 \
    --position_gap_thresh 0.2
