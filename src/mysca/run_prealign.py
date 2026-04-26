"""Pre-alignment CLI for raw-FASTA input.

Produces an aligned FASTA (`aligned.fasta`) ready to feed into `sca-preprocess`.
Optionally runs a clustering step first to reduce redundancy before alignment.

External binaries (`mafft` or `clustalo` for the alignment step, and `mmseqs`
if `--cluster mmseqs2`) must be available on PATH; otherwise the program
raises FileNotFoundError and stops.

-------------------------------------------------------------------------------
EXAMPLE USAGE:

    sca-prealign -i raw.fasta -o prealign_out
    sca-prealign -i raw.fasta -o prealign_out --cluster mmseqs2 \
        --cluster_min_seq_id 0.9

    # Use Clustal Omega instead of MAFFT, with aligner-specific kwargs.
    sca-prealign -i raw.fasta -o prealign_out --align clustalo \
        --align_args guidetree_out=true output_order=tree-order

Then chain into sca-preprocess:

    sca-preprocess -i prealign_out/aligned.fasta -o preprocess_out

-------------------------------------------------------------------------------
COMMAND LINE ARGUMENTS:

    Alignment group:
        --align {mafft, clustalo}     alignment method (default mafft)
        --align_threads INT           threads for the aligner (default 1)
        --align_bin PATH              explicit path to the alignment binary
        --align_extra ...             raw passthrough args appended to the
                                      aligner CLI verbatim
        --align_args KEY[=VAL] ...    aligner-specific structured kwargs.
                                      The wrapper for --align consumes the
                                      keys it knows; unknown keys raise.
                                      Bare KEY is treated as KEY=true.
        --output_format {fasta, stockholm}  format for the aligned output

    Per-aligner --align_args keys:
        clustalo:
            guidetree_out=true        write guide tree to
                                      <outdir>/guidetree.dnd
            output_order={tree-order, input-order}
                                      order of aligned output
        mafft: (none currently)

OUTPUTS (under outdir):

    aligned.fasta or aligned.sto      aligned MSA
    clustered.fasta                   only when --cluster is not 'none'
    guidetree.dnd                     only with --align clustalo
                                      --align_args guidetree_out=true
    filter_history.json               per-stage sequence counts
    prealign_args.json                resolved arguments
    prealign.log                      run log
    images/prealign_filter_history.png  only when --plot is passed
"""

import argparse
import json
import logging
import os
import sys

from mysca.logging_config import configure_logging
from mysca.prealign import (
    ALIGNER_BINARIES,
    ALIGNERS,
    CLUSTERERS,
    SUPPORTED_ALIGNMENT_FORMATS,
    _resolve_bin,
    run_align,
    run_cluster,
)

PREALIGN_LOG_FNAME = "prealign.log"
PREALIGN_ARGS_FNAME = "prealign_args.json"
PREALIGN_FILTER_HISTORY_FNAME = "filter_history.json"
CLUSTERED_FASTA_FNAME = "clustered.fasta"
ALIGNED_BASENAME = "aligned"

_ALIGNED_EXT = {
    "fasta": ".fasta",
    "stockholm": ".sto",
}

logger = logging.getLogger("mysca.run_prealign")


def _parse_align_args(items):
    """Parse `--align_args` items into a dict of aligner-specific kwargs.

    Each item is `KEY` (treated as `KEY=true`) or `KEY=VAL`. Values stay
    as strings; the chosen aligner's wrapper handles type coercion and
    validation, and raises on keys it doesn't recognize.
    """
    out = {}
    for raw in items:
        if "=" in raw:
            key, _, val = raw.partition("=")
        else:
            key, val = raw, "true"
        if not key:
            raise ValueError(f"Empty key in --align_args item: {raw!r}")
        if key in out:
            raise ValueError(f"Duplicate --align_args key: {key!r}")
        out[key] = val
    return out


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_fpath", type=str, required=True,
                        help="Filepath of input (raw) FASTA.")
    parser.add_argument("-o", "--outdir", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="Verbosity level (0=warnings only).")
    parser.add_argument("--pbar", action="store_true",
                        help="Enable tqdm progress bars.")
    parser.add_argument(
        "--plot", action="store_true",
        help="Write a per-stage sequence-count diagnostic plot to outdir/images/",
    )

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
        help="Alignment method. Choices: 'mafft' (default), 'clustalo'.",
    )
    align.add_argument("--align_threads", type=int, default=1)
    align.add_argument("--align_bin", type=str, default=None,
                       help="Explicit path to the alignment binary.")
    align.add_argument("--align_extra", nargs="*", default=[],
                       help="Extra arguments passed through to the aligner.")
    align.add_argument(
        "--output_format", type=str, default="fasta",
        choices=list(SUPPORTED_ALIGNMENT_FORMATS),
        help="Format for the aligned output. Default 'fasta'.",
    )
    align.add_argument(
        "--align_args", nargs="*", default=[], metavar="KEY[=VAL]",
        help=("Aligner-specific structured kwargs. Each item is either "
              "a bare KEY (treated as KEY=true) or KEY=VAL. The wrapper "
              "for the chosen --align consumes the keys it knows and "
              "raises on unknown keys. Use this for options that need "
              "post-processing (e.g. clustalo guidetree_out=true writes "
              "a guide tree under outdir). For raw passthrough to the "
              "aligner CLI, use --align_extra instead. Examples: "
              "`--align_args guidetree_out=true output_order=tree-order` "
              "(clustalo)."),
    )

    return parser.parse_args(args)


def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    configure_logging(
        verbosity=args.verbosity,
        logfile=os.path.join(outdir, PREALIGN_LOG_FNAME),
    )

    aligner_kwargs = _parse_align_args(args.align_args)

    input_fpath = args.input_fpath
    if not os.path.isfile(input_fpath):
        raise FileNotFoundError(f"Input FASTA not found: {input_fpath}")

    # Resolve all needed binaries up front so a missing tool fails fast.
    if args.cluster != "none":
        _resolve_bin("mmseqs", override=args.cluster_bin)
    _resolve_bin(ALIGNER_BINARIES[args.align], override=args.align_bin)

    # Persist resolved args for reproducibility.
    args_fpath = os.path.join(outdir, PREALIGN_ARGS_FNAME)
    with open(args_fpath, "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    # Track per-stage sequence counts for the diagnostic plot / filter history.
    from mysca.prealign import _count_fasta
    n_initial = _count_fasta(input_fpath)
    filter_history = [{
        "stage": "initial",
        "label": "initial",
        "n_sequences": n_initial,
        "n_filtered": 0,
    }]

    # Stage 1: optional clustering.
    if args.cluster != "none":
        clustered_fpath = os.path.join(outdir, CLUSTERED_FASTA_FNAME)
        cluster_info = run_cluster(
            input_fpath, clustered_fpath,
            method=args.cluster,
            min_seq_id=args.cluster_min_seq_id,
            coverage=args.cluster_coverage,
            coverage_mode=args.cluster_cov_mode,
            threads=args.cluster_threads,
            bin_path=args.cluster_bin,
        )
        filter_history.append({
            "stage": "cluster",
            "label": f"cluster ({args.cluster})",
            "n_sequences": cluster_info["n_out"],
            "n_filtered": cluster_info["n_in"] - cluster_info["n_out"],
        })
        aligner_input = clustered_fpath
    else:
        logger.info("Clustering disabled (--cluster none).")
        aligner_input = input_fpath

    # Stage 2: alignment.
    aligned_fname = ALIGNED_BASENAME + _ALIGNED_EXT[args.output_format]
    aligned_fpath = os.path.join(outdir, aligned_fname)
    align_info = run_align(
        aligner_input, aligned_fpath,
        method=args.align,
        threads=args.align_threads,
        bin_path=args.align_bin,
        extra_args=args.align_extra,
        output_format=args.output_format,
        aligner_kwargs=aligner_kwargs,
    )
    filter_history.append({
        "stage": "align",
        "label": f"align ({args.align})",
        "n_sequences": align_info["n_out"],
        "n_filtered": align_info["n_in"] - align_info["n_out"],
    })

    # Persist filter_history for replay.
    fh_path = os.path.join(outdir, PREALIGN_FILTER_HISTORY_FNAME)
    with open(fh_path, "w") as f:
        json.dump(filter_history, f)

    if args.plot:
        from mysca.pl import plot_prealign_filter_history
        imgdir = os.path.join(outdir, "images")
        os.makedirs(imgdir, exist_ok=True)
        logger.info("Writing prealign filter diagnostic plot to %s", imgdir)
        plot_prealign_filter_history(filter_history, imgdir)

    logger.info(
        "Prealign complete. Aligned output (%s) at: %s",
        args.output_format, aligned_fpath,
    )
    logger.info(
        "Next step: sca-preprocess -i %s --input_format %s -o <preprocess_outdir>",
        aligned_fpath, args.output_format,
    )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
