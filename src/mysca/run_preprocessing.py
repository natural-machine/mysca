"""SCA preprocessing pipeline

Runs the preprocessing steps of SCA, given an input MSA. Creates and populates
an output directory with the preprocessed results. See references:
    [1] SI to Rivoire et al., 2016

-------------------------------------------------------------------------------
COMMAND LINE ARGUMENTS:

SCA parameters (see [1] for definitions):
    --gap_truncation_thresh   (τ; default 0.4) Remove columns with gap
        frequency above this threshold, applied before any sequence-level
        filtering.
    --sequence_gap_thresh     (γ_seq; default 0.2) Remove sequences with
        gap frequency above this threshold.
    --reference               (default None) Reference sequence ID in the
        MSA. Used both to anchor similarity filtering and to derive raw-
        residue coordinates in downstream logs.
    --reference_similarity_thresh   (Δ; default 0.2) Minimum similarity to
        the reference sequence. Requires --reference.
    --sequence_similarity_thresh    (δ; default 0.8) Clustering threshold
        for sequence weighting.
    --position_gap_thresh     (γ_pos; default 0.2) Remove columns with
        *weighted* gap frequency above this threshold. Applied after
        sequence weighting.

See `docs/cli_reference.md` for the full list (input/output format, symbol
alphabet, gap-value convention, weight method, plotting, etc.).

-------------------------------------------------------------------------------
OUTPUTS:

preprocessing_results.npz
    msa : preprocessed MSA. Integer 2d array of shape (M x L)
    retained_sequences : Indices of retained sequences in the original MSA.
    retained_positions : Indices of retained positions in the original MSA.
    retained_sequence_ids : IDs of retained sequences in the original MSA.
    sequence_weights : Sampling weights for the retained sequences.
    fi0_pretruncation : Gap frequency per position, prior to truncation.

msa_binary2d_sp.npz
    MSA in a 2-dimensional sparse one-hot format of shape (M x DL), with D
    the alphabet size (e.g. D=20 for the canonical amino acids).

sym2int.json
    Mapping from sequence characters (i.e. amino acid symbols) to their
    integer representation.

preprocessing_args.json
    Mapping from command line arguments to their values.

msa_orig.fasta-aln
    Original MSA in fasta format, before any filtering.

filter_history.json
    Per-stage filter diagnostics (counts + threshold + stat distribution).
    Always persisted so `sca-plots` can replay the diagnostic plots later.

images/ (only when --plot is passed)
    filter_history.png, filter_distributions.png.

-------------------------------------------------------------------------------
EXAMPLE USAGE:

sca-preprocess -i </path/to/msa.fasta> -o </path/to/outdir> \\
    --gap_truncation_thresh 0.4 \\
    --sequence_gap_thresh 0.2 \\
    --reference <reference-fasta-id> \\
    --reference_similarity_thresh 0.2 \\
    --sequence_similarity_thresh 0.8 \\
    --position_gap_thresh 0.2

"""

import argparse
import logging
import os, sys
import numpy as np
import tqdm as tqdm

from mysca.io import load_msa
from mysca.logging_config import configure_logging
from mysca.mappings import SymMap
from mysca.constants import AA_STD20
from mysca.preprocess import preprocess_msa
from mysca.pl import plot_filter_history, plot_filter_distributions
from mysca.results import (
    PreprocessingResults,
    PREPROCESSING_RESULTS_FNAME as OUTPUT_RESULTS_FNAME,
    PREPROCESSING_SYMMAP_FNAME as OUTPUT_SYMMAP_FNAME,
    PREPROCESSING_ARGS_FNAME as OUTPUT_ARGS_FNAME,
    PREPROCESSING_MSAORIG_FNAME as OUTPUT_MSAORIG_FNAME,
)

PREPROCESSING_LOG_FNAME = "preprocessing.log"

logger = logging.getLogger("mysca.run_preprocessing")


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--msa_fpath", type=str, required=True,
                        help="Filepath of input MSA.")
    parser.add_argument("-o", "--outdir", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="Verbosity level (0=warnings only).")
    parser.add_argument("--pbar", action="store_true",
                        help="Enable tqdm progress bars.")
    parser.add_argument("--plot", action="store_true",
                        help="Emit filter_history.png and "
                        "filter_distributions.png to outdir/images/.")

    parser.add_argument(
        "--input_format", type=str, default="fasta",
        choices=["fasta", "stockholm"],
        help="Format of the input MSA file. Default 'fasta'. "
        "Format is never inferred from the filename.",
    )
    parser.add_argument(
        "--syms", type=str, default="default",
        help="Symbol alphabet. 'default' → standard 20 amino acids; "
        "'none' → disable excluded-symbol filtering and auto-detect; "
        "any other string is treated as an explicit character set.",
    )
    parser.add_argument(
        "--gapsym", type=str, default="-",
        help="Gap symbol in the input MSA.",
    )
    parser.add_argument(
        "--gap_value", type=int, default=0,
        help="Integer value assigned to the gap symbol in the SymMap. "
        "Default 0 (gap first). Pass len(aa_syms) to place gap at the end "
        "(legacy behavior).",
    )
    parser.add_argument(
        "--weight_method", type=str, default="sparse",
        choices=["sparse", "gpu"],
        help="Sequence-weight computation backend. 'sparse' (default) "
        "uses a CPU sparse-CSR implementation; 'gpu' dispatches to torch "
        "(CUDA/MPS/XPU), falling back to 'sparse' if no accelerator is "
        "detected. Non-production benchmark variants ('_v3', '_v4', "
        "'_v6') remain callable via the preprocess_msa library API but "
        "are intentionally not exposed here.",
    )
    parser.add_argument(
        "--block_size", type=int, default=512,
        help="Row-block size for pairwise sequence-similarity "
        "computation inside weight_method='sparse'. Smaller blocks "
        "cap peak memory; larger blocks amortize overhead. Default 512.",
    )
    
    sca_params = parser.add_argument_group("SCA parameters")
    sca_params.add_argument(
        "--gap_truncation_thresh", type=float, default=0.4,
        help="τ: drop columns with gap frequency above this threshold. "
        "Applied before any sequence-level filtering.",
    )
    sca_params.add_argument(
        "--sequence_gap_thresh", type=float, default=0.2,
        help="γ_seq: drop sequences with gap frequency above this threshold.",
    )
    sca_params.add_argument(
        "--reference", type=str, default=None,
        help="Reference sequence ID in the MSA. Anchors similarity "
        "filtering and supplies raw-residue coordinates downstream. "
        "Required by --reference_similarity_thresh.",
    )
    sca_params.add_argument(
        "--reference_similarity_thresh", type=float, default=0.2,
        help="Δ: minimum fractional identity to the reference sequence. "
        "Sequences below this similarity are dropped. Requires --reference.",
    )
    sca_params.add_argument(
        "--sequence_similarity_thresh", type=float, default=0.8,
        help="δ: clustering threshold for sequence weighting. Sequences "
        "within this pairwise similarity contribute down-weighted.",
    )
    sca_params.add_argument(
        "--position_gap_thresh", type=float, default=0.2,
        help="γ_pos: drop columns with *weighted* gap frequency above "
        "this threshold. Applied after sequence weighting.",
    )
    return parser.parse_args(args)


def main(args):

    # Process command line args
    msa_fpath = args.msa_fpath
    outdir = args.outdir
    verbosity = args.verbosity
    pbar = args.pbar
    do_plot = args.plot
    weight_computation_version = args.weight_method
    block_size = args.block_size
    
    syms = args.syms
    gapsym = args.gapsym
    gap_value = args.gap_value

    gap_truncation_thresh = args.gap_truncation_thresh
    sequence_gap_thresh = args.sequence_gap_thresh
    reference_id = args.reference
    reference_similarity_thresh = args.reference_similarity_thresh
    sequence_similarity_thresh = args.sequence_similarity_thresh
    position_gap_thresh = args.position_gap_thresh
    
    os.makedirs(outdir, exist_ok=True)
    configure_logging(
        verbosity=verbosity,
        logfile=os.path.join(outdir, PREPROCESSING_LOG_FNAME),
    )

    # Housekeeping
    if reference_id is None or reference_id.lower() == "none":
        logger.info("No reference entry specified.")
        reference_id = None

    if do_plot:
        imgdir = os.path.join(outdir, "images")
        os.makedirs(imgdir, exist_ok=True)

    if syms.lower() in ["default"]:
        sym_map = SymMap(
            aa_syms=AA_STD20, gapsym=gapsym, gap_value=gap_value,
        )
    elif syms.lower() in ["none"]:
        logger.warning(
            "--syms none disables excluded-symbol filtering; "
            "using auto-detected alphabet."
        )
        sym_map = None
    else:
        sym_map = SymMap(
            aa_syms=syms, gapsym=gapsym, gap_value=gap_value,
        )

    # Load MSA
    logger.info("Loading MSA (%s) from: %s", args.input_format, msa_fpath)
    msa_obj_orig, msa_orig, seqids_orig, sym_map = load_msa(
        msa_fpath, format=args.input_format,
        mapping=sym_map,
    )
    num_seq_orig, num_pos_orig = msa_orig.shape

    logger.info(
        "Loaded MSA. shape: %s (sequences x positions)", msa_orig.shape
    )
    logger.info("Symbols: %s", sym_map.aa_list)

    # Run preprocessing script
    msa, preprocessing_results = preprocess_msa(
        msa_orig, seqids_orig,
        mapping=sym_map,
        gap_truncation_thresh=gap_truncation_thresh,
        sequence_gap_thresh=sequence_gap_thresh,
        reference_id=reference_id,
        reference_similarity_thresh=reference_similarity_thresh,
        sequence_similarity_thresh=sequence_similarity_thresh,
        position_gap_thresh=position_gap_thresh,
        use_pbar=pbar,
        verbosity=verbosity,
        weight_computation_version=weight_computation_version,
        block_size=block_size,
    )

    results = PreprocessingResults.from_preprocess_output(
        msa, preprocessing_results,
        sym_map=sym_map,
        msa_obj_orig=msa_obj_orig,
    )
    results.save(outdir)

    if do_plot:
        filter_history = preprocessing_results.get("filter_history", [])
        if filter_history:
            logger.info("Writing filter diagnostic plots to %s", imgdir)
            plot_filter_history(filter_history, imgdir)
            plot_filter_distributions(filter_history, imgdir)

    logger.info("Output saved to %s", outdir)
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
