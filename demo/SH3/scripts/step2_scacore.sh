#!/usr/bin/env bash

outdir=out

sca-core \
    -i ${outdir}/preprocessing \
    -o ${outdir}/scacore \
    --regularization 0.03 \
    --seed 42
