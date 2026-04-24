#!/usr/bin/env bash
set -euo pipefail

outdir=out/from_raw

# Same as step6_plots.sh but for the from_raw path. Also includes the
# prealign diagnostic (sequence counts per stage) since prealign
# persists a filter_history.json.
sca-plots \
    --prealign ${outdir}/prealign \
    --preprocessing ${outdir}/preprocessing \
    --scacore ${outdir}/scacore \
    --imgdir ${outdir}/plots_replay
