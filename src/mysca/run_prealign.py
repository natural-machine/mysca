"""Pre-alignment CLI for raw-FASTA input.

Produces an aligned FASTA (`aligned.fasta`) ready to feed into `sca-preprocess`.
Optionally runs a clustering step first to reduce redundancy before alignment.

External binaries (`mafft`, and `mmseqs` if `--cluster mmseqs2`) must be
available on PATH; otherwise the program raises FileNotFoundError and stops.

-------------------------------------------------------------------------------
EXAMPLE USAGE:

    sca-prealign -i raw.fasta -o prealign_out
    sca-prealign -i raw.fasta -o prealign_out --cluster mmseqs2 \
        --cluster_min_seq_id 0.9

Then chain into sca-preprocess:

    sca-preprocess -i prealign_out/aligned.fasta -o preprocess_out
"""

import argparse
import json
import logging
import os
import sys

from mysca.logging_config import configure_logging
from mysca.prealign import (
    ALIGNERS,
    CLUSTERERS,
    _resolve_bin,
    run_align,
    run_cluster,
)

PREALIGN_LOG_FNAME = "prealign.log"
PREALIGN_ARGS_FNAME = "prealign_args.json"
CLUSTERED_FASTA_FNAME = "clustered.fasta"
ALIGNED_FASTA_FNAME = "aligned.fasta"

logger = logging.getLogger("mysca.run_prealign")


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_fpath", type=str, required=True,
                        help="Filepath of input (raw) FASTA.")
    parser.add_argument("-o", "--outdir", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("-v", "--verbosity", type=int, default=1)
    parser.add_argument("--pbar", action="store_true")

    cluster = parser.add_argument_group("Clustering")
    cluster.add_argument(
        "--cluster", type=str, default="none",
        choices=["none"] + sorted(CLUSTERERS),
        help="Clustering method (or 'none' to skip). Default 'none'.",
    )
    cluster.add_argument("--cluster_min_seq_id", type=float, default=0.9)
    cluster.add_argument("--cluster_coverage", type=float, default=0.8)
    cluster.add_argument("--cluster_cov_mode", type=int, default=1)
    cluster.add_argument("--cluster_threads", type=int, default=1)
    cluster.add_argument("--cluster_bin", type=str, default=None,
                         help="Explicit path to the clustering binary.")

    align = parser.add_argument_group("Alignment")
    align.add_argument(
        "--align", type=str, default="mafft",
        choices=sorted(ALIGNERS),
        help="Alignment method. Default 'mafft'.",
    )
    align.add_argument("--align_threads", type=int, default=1)
    align.add_argument("--align_bin", type=str, default=None,
                       help="Explicit path to the alignment binary.")
    align.add_argument("--align_extra", nargs="*", default=[],
                       help="Extra arguments passed through to the aligner.")

    return parser.parse_args(args)


def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    configure_logging(
        verbosity=args.verbosity,
        logfile=os.path.join(outdir, PREALIGN_LOG_FNAME),
    )

    input_fpath = args.input_fpath
    if not os.path.isfile(input_fpath):
        raise FileNotFoundError(f"Input FASTA not found: {input_fpath}")

    # Resolve all needed binaries up front so a missing tool fails fast.
    if args.cluster != "none":
        _resolve_bin("mmseqs", override=args.cluster_bin)
    _resolve_bin("mafft", override=args.align_bin)

    # Persist resolved args for reproducibility.
    args_fpath = os.path.join(outdir, PREALIGN_ARGS_FNAME)
    with open(args_fpath, "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    # Stage 1: optional clustering.
    if args.cluster != "none":
        clustered_fpath = os.path.join(outdir, CLUSTERED_FASTA_FNAME)
        run_cluster(
            input_fpath, clustered_fpath,
            method=args.cluster,
            min_seq_id=args.cluster_min_seq_id,
            coverage=args.cluster_coverage,
            coverage_mode=args.cluster_cov_mode,
            threads=args.cluster_threads,
            bin_path=args.cluster_bin,
        )
        aligner_input = clustered_fpath
    else:
        logger.info("Clustering disabled (--cluster none).")
        aligner_input = input_fpath

    # Stage 2: alignment.
    aligned_fpath = os.path.join(outdir, ALIGNED_FASTA_FNAME)
    run_align(
        aligner_input, aligned_fpath,
        method=args.align,
        threads=args.align_threads,
        bin_path=args.align_bin,
        extra_args=args.align_extra,
    )

    logger.info("Prealign complete. Aligned FASTA at: %s", aligned_fpath)
    logger.info(
        "Next step: sca-preprocess -i %s -o <preprocess_outdir>", aligned_fpath,
    )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
