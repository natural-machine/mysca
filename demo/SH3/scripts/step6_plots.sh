#!/usr/bin/env bash
set -euo pipefail

outdir=out/from_msa

# Regenerate every replayable diagnostic figure from the persisted
# preprocess + scacore outputs, without rerunning the pipeline.
# Passing --preprocessing alongside --scacore enables the positional
# conservation variants that need retained_positions + the original
# MSA length.
sca-plots \
    --preprocessing ${outdir}/preprocessing \
    --scacore ${outdir}/scacore \
    --imgdir ${outdir}/plots_replay
