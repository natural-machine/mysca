#!/usr/bin/env bash
set -e

cd SH3

# Path A: start from the pre-aligned SH3 MSA.
./scripts/step1_preprocessing.sh
./scripts/step2_scacore.sh
./scripts/step3_project.sh
./scripts/step4_project_hmmer.sh
./scripts/step5_structure.sh

# Path B: start from raw (unaligned) SH3 sequences, prealign, then pipeline.
./scripts/step0_prealign_raw.sh
./scripts/step1_preprocessing_raw.sh
./scripts/step2_scacore_raw.sh
./scripts/step3_project_raw.sh
