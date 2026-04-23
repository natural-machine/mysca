"""Project primary sequences onto an existing SCA result.

Given an input FASTA plus the output directories of a prior
``sca-preprocess`` + ``sca-core`` run, map each input sequence's raw
residues onto the IC groups and write a summary + per-sequence detail
under the output directory.

-------------------------------------------------------------------------------
COMMAND LINE ARGUMENTS:

    -i --input_fpath : Path to an input FASTA. Each record is projected
        independently. Records whose ID matches an entry in the reference
        MSA (``msa_orig.fasta-aln`` under --preprocessing) are resolved
        in-sample (no external alignment performed).
    --scacore : Path to an ``sca-core`` output directory
        (contains ``sca_results/msa_sectors/sector_*_msapos.npy`` and
        ``sca_results/v_ica_normalized.npy``).
    --preprocessing : Path to an ``sca-preprocess`` output directory
        (contains ``preprocessing_results.npz`` and ``msa_orig.fasta-aln``).
    -o --outdir : Output directory.
    --aligner : Registered entry in ``mysca.project.alignment.ALIGNERS``.
        Default ``mafft_add``.
    --align_bin : Explicit path to the alignment binary.
    --align_threads : Threads for the alignment tool. Default 1.

-------------------------------------------------------------------------------
OUTPUTS:

projection.json
    Top-level dict containing run args plus a list of per-sequence
    dicts: seq_id, raw_sequence, aligned_sequence,
    residue_by_processed_col (length L_proc), ic_memberships (per IC
    raw-residue indices), ic_loadings, ic_processed_cols, in_sample.

per_sequence/<seqid>_residues.tsv
    One row per (IC, residue) pairing for readable inspection.

projection_args.json
    Mapping from CLI argument to value.

projection.log
    Run log.

-------------------------------------------------------------------------------
EXAMPLE USAGE:

    sca-project -i new_seqs.fasta \\
        --preprocessing preprocess_out \\
        --scacore scacore_out \\
        -o projection_out

"""

import argparse
import json
import logging
import os
import sys

from mysca.logging_config import configure_logging
from mysca.project import project_sequences, ALIGNERS

PROJECT_LOG_FNAME = "projection.log"
PROJECT_RESULTS_FNAME = "projection.json"
PROJECT_ARGS_FNAME = "projection_args.json"
PER_SEQUENCE_DIRNAME = "per_sequence"

logger = logging.getLogger("mysca.run_project")


def parse_args(args):
    parser = argparse.ArgumentParser(
        description=(
            "Project primary sequences onto an existing SCA result "
            "(in-sample short-circuit + out-of-sample alignment)."
        ),
    )
    parser.add_argument(
        "-i", "--input_fpath", type=str, required=True,
        help="Path to an input FASTA of sequences to project.",
    )
    parser.add_argument(
        "--preprocessing", type=str, required=True, metavar="DIR",
        help="sca-preprocess output directory (must include "
        "msa_orig.fasta-aln).",
    )
    parser.add_argument(
        "--scacore", type=str, required=True, metavar="DIR",
        help="sca-core output directory (must include "
        "sca_results/msa_sectors/sector_*_msapos.npy and "
        "sca_results/v_ica_normalized.npy).",
    )
    parser.add_argument(
        "-o", "--outdir", type=str, required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--aligner", type=str, default="mafft_add",
        choices=sorted(ALIGNERS),
        help="Out-of-sample alignment method. 'mafft_add' "
        "(default) uses `mafft --add --keeplength`. 'hmmalign' is "
        "registered as a name but not yet implemented. In-sample "
        "records bypass alignment entirely.",
    )
    parser.add_argument(
        "--align_bin", type=str, default=None,
        help="Explicit path to the alignment binary (default: resolve "
        "from PATH).",
    )
    parser.add_argument(
        "--align_threads", type=int, default=1,
        help="Threads for the alignment tool. Default 1.",
    )
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="Verbosity level (0=warnings only).")
    return parser.parse_args(args)


def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    configure_logging(
        verbosity=args.verbosity,
        logfile=os.path.join(outdir, PROJECT_LOG_FNAME),
    )

    if not os.path.isfile(args.input_fpath):
        raise FileNotFoundError(f"Input FASTA not found: {args.input_fpath}")

    with open(os.path.join(outdir, PROJECT_ARGS_FNAME), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    aligner_kwargs = {
        "bin_path": args.align_bin,
        "threads": args.align_threads,
    }

    result = project_sequences(
        args.input_fpath,
        sca_result_dir=args.scacore,
        preproc_result_dir=args.preprocessing,
        aligner=args.aligner,
        workdir=os.path.join(outdir, "_align_workdir"),
        aligner_kwargs=aligner_kwargs,
    )

    with open(os.path.join(outdir, PROJECT_RESULTS_FNAME), "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    per_seq_dir = os.path.join(outdir, PER_SEQUENCE_DIRNAME)
    os.makedirs(per_seq_dir, exist_ok=True)
    for proj in result.projections:
        tsv_path = os.path.join(per_seq_dir, f"{proj.seq_id}_residues.tsv")
        with open(tsv_path, "w") as f:
            f.write("ic_index\traw_residue_idx\tprocessed_col\tv_ica_loading\n")
            for ic_idx, (members, loadings, cols) in enumerate(
                zip(proj.ic_memberships, proj.ic_loadings, proj.ic_processed_cols)
            ):
                for resi, loading, col in zip(members, loadings, cols):
                    f.write(
                        f"{ic_idx}\t{int(resi)}\t{int(col)}\t{float(loading)}\n"
                    )

    n_in = sum(1 for p in result.projections if p.in_sample)
    n_out = len(result.projections) - n_in
    logger.info(
        "Projected %d sequences (%d in-sample, %d via %s).",
        len(result.projections), n_in, n_out, args.aligner,
    )
    logger.info("sca-project done. Output at: %s", outdir)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
