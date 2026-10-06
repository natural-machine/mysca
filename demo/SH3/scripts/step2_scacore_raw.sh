#!/usr/bin/env bash

outdir=out/from_raw

sca-core \
    -i ${outdir}/preprocessing \
    -o ${outdir}/scacore \
    --regularization 0.03 \
    --seed 42
