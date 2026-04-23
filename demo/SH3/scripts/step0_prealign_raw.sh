#!/usr/bin/env bash

outdir=out/from_raw

infile=data/seqs/PF00018_raw.fasta

sca-prealign \
    -i ${infile} \
    -o ${outdir}/prealign \
    --align mafft \
    --align_threads 1
