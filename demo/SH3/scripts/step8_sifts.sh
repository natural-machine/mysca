#!/usr/bin/env bash
set -euo pipefail

outdir=out/from_msa

# Resolve human Fyn (UniProt P06241) → 1SHF via SIFTS's best_structures
# endpoint, then project. A pre-populated cache under
# data/sifts_cache/P06241.json lets this step run offline: the SIFTS
# lookup hits the cache file and never touches the network. Passing
# --cache_dir data/sifts_cache steers mysca at that shipped cache
# instead of the default ./.sifts_cache.
#
# On first online run with a different --cache_dir (or after deleting
# the shipped JSON), sca-structure would fetch the live SIFTS
# response and cache it itself. The top-ranked best_structures entry
# may have changed by then; the shipped cache pins the demo to a
# known-good mapping.
sca-structure \
    --uniprot_ids P06241 \
    --pdb_dir data/pdbs \
    --cache_dir data/sifts_cache \
    --preprocessing ${outdir}/preprocessing \
    --scacore ${outdir}/scacore \
    -o ${outdir}/structure_sifts
