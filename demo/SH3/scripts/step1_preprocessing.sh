#!/usr/bin/env bash

outdir=out

infile=data/msas/SH3_demo_MSA_1.afa
reference='4837_jgi||3708||Equilibrative'

sca-preprocess \
    -i ${infile} \
    -o ${outdir}/preprocessing \
    --gap_truncation_thresh 0.4 \
    --sequence_gap_thresh 0.2 \
    --reference ${reference} \
    --reference_similarity_thresh 0.2 \
    --sequence_similarity_thresh 0.8 \
    --position_gap_thresh 0.2
