#!/usr/bin/env bash
set -e

cd SH3

# Path A: start from the pre-aligned SH3 MSA.
./scripts/step1_preprocessing.sh
./scripts/step2_scacore.sh

# Path B: start from raw (unaligned) SH3 sequences, prealign, then pipeline.
./scripts/step1_prealign_raw.sh
./scripts/step2_preprocessing_raw.sh
./scripts/step3_scacore_raw.sh
