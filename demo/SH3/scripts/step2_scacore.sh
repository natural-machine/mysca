#!/usr/bin/env bash

outdir=out/from_msa

sca-core \
    -i ${outdir}/preprocessing \
    -o ${outdir}/scacore \
    --regularization 0.03 \
    --n_components 10 \
    --seed 42
