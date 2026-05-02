"""Project primary sequences onto an existing SCA result.

Given an input FASTA (or a record selected from a reference MSA) plus
the output directories of a prior ``sca-preprocess`` + ``sca-core`` run,
map each input sequence's raw residues onto the IC groups and write a
summary + per-sequence detail under the output directory.

-------------------------------------------------------------------------------
COMMAND LINE ARGUMENTS:

    -i --input_fpath : Path to an input FASTA. Each record is projected
        independently. Records whose ID matches an entry in the reference
        MSA (``msa_orig.fasta-aln`` under --preprocessing) are resolved
        in-sample (no external alignment performed). Mutually exclusive
        with ``--from_msa`` / ``--seq_id``.
    --from_msa : Path to a (possibly aligned) FASTA from which to
        extract a single record by --seq_id, ungap, and project. Useful
        for in-sample replay without the FASTA-surgery boilerplate.
        Requires --seq_id; mutually exclusive with -i.
    --seq_id : First whitespace-delimited token of the target record's
        header in --from_msa. Required with --from_msa.
    --scacore : Path to an ``sca-core`` output directory
        (contains ``ic_positions/ic_*_msaproc.npy`` and
        ``sca_results/v_ica_normalized.npy``).
    --preprocessing : Path to an ``sca-preprocess`` output directory
        (contains ``preprocessing_results.npz`` and ``msa_orig.fasta-aln``).
    -o --outdir : Output directory.
    --aligner : Out-of-sample alignment method. Default ``mafft_add``
        (uses ``mafft --add --keeplength``). ``hmmalign`` builds a
        profile HMM (``hmmbuild --hand --amino``) treating every
        reference column as a match state, aligns new sequences with
        ``hmmalign --outformat afa``, and keeps only match columns.
        In-sample records bypass alignment entirely.
    --align_bin : Explicit path to the alignment binary. For
        ``--aligner hmmalign`` this overrides the ``hmmalign`` binary;
        ``hmmbuild`` is resolved from PATH.
    --align_threads : Threads for the alignment tool. Default 1
        (unused by ``hmmalign``).
    --save_dataframe : also write ``seq_projections.tsv`` to outdir,
        with columns seq_id, aligned_sequence, raw_sequence, in_sample,
        Up_0..Up_{n_components-1}. Requires pandas.

-------------------------------------------------------------------------------
OUTPUTS:

projection.json
    Top-level dict containing run args plus a list of per-sequence
    dicts: seq_id, raw_sequence, aligned_sequence,
    residue_by_processed_col (length L_proc), ic_residues (per IC
    raw-residue indices), ic_loadings, ic_processed_cols, in_sample,
    up_score (length n_components — the sequence's Uᵖ row, or None
    when the source SCAResults lacks the eigendecomposition fields).

per_sequence/<seqid>_residues.tsv
    One row per (IC, residue) pairing for readable inspection.

projection_args.json
    Mapping from CLI argument to value.

projection.log
    Run log.

seq_projections.tsv (only when --save_dataframe)
    Tab-separated table: seq_id, aligned_sequence, raw_sequence,
    in_sample, Up_0..Up_{n_components-1}.

-------------------------------------------------------------------------------
EXAMPLE USAGE:

    sca-project -i new_seqs.fasta \\
        --preprocessing preprocess_out \\
        --scacore scacore_out \\
        -o projection_out

    # In-sample replay: extract one record from the training MSA and
    # project it back, no FASTA-surgery glue required.
    sca-project --from_msa preprocess_out/msa_orig.fasta-aln \\
        --seq_id 'MYSEQ_HUMAN/1-100' \\
        --preprocessing preprocess_out \\
        --scacore scacore_out \\
        -o projection_out

"""

import argparse
import json
import logging
import os
import re
import sys

from Bio import SeqIO

from mysca.logging_config import configure_logging
from mysca.project import project_sequences, ALIGNERS


def _safe_filename_component(s: str) -> str:
    """Replace filesystem-unsafe characters so a FASTA ID can be used
    as a filename component. Pfam IDs like ``VAV_HUMAN/788-834`` and
    JGI IDs like ``4837_jgi||3708||...`` both land here."""
    return re.sub(r'[/\\|:*?"<>\s]+', "_", s)

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
        "-i", "--input_fpath", type=str, default=None,
        help="Path to an input FASTA of sequences to project. Mutually "
        "exclusive with --from_msa / --seq_id.",
    )
    parser.add_argument(
        "--from_msa", type=str, default=None, metavar="FASTA",
        help="Path to a (possibly aligned) FASTA. Combined with "
        "--seq_id, extract that single record, ungap it, and project. "
        "Mutually exclusive with -i.",
    )
    parser.add_argument(
        "--seq_id", type=str, default=None, metavar="ID",
        help="Header (first whitespace-delimited token) of the record "
        "to extract from --from_msa. Required with --from_msa.",
    )
    parser.add_argument(
        "--preprocessing", type=str, required=True, metavar="DIR",
        help="sca-preprocess output directory (must include "
        "msa_orig.fasta-aln).",
    )
    parser.add_argument(
        "--scacore", type=str, required=True, metavar="DIR",
        help="sca-core output directory (must include "
        "ic_positions/ic_*_msaproc.npy and "
        "sca_results/v_ica_normalized.npy).",
    )
    parser.add_argument(
        "-o", "--outdir", type=str, required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--aligner", type=str, default="mafft_add",
        choices=sorted(ALIGNERS),
        help="Out-of-sample alignment method. 'mafft_add' (default) "
        "uses `mafft --add --keeplength`. 'hmmalign' builds a profile "
        "HMM (`hmmbuild --hand --amino`) with every reference column "
        "as a match state, aligns new sequences (`hmmalign --outformat "
        "afa`) and keeps only match columns. In-sample records bypass "
        "alignment entirely.",
    )
    parser.add_argument(
        "--align_bin", type=str, default=None,
        help="Explicit path to the alignment binary (default: resolve "
        "from PATH). For --aligner hmmalign this is the `hmmalign` "
        "binary; `hmmbuild` is always resolved from PATH.",
    )
    parser.add_argument(
        "--align_threads", type=int, default=1,
        help="Threads for the alignment tool. Default 1 "
        "(unused by hmmalign).",
    )
    parser.add_argument(
        "--save_dataframe", action="store_true",
        help="Also write seq_projections.tsv to outdir, with columns "
             "seq_id, aligned_sequence, raw_sequence, in_sample, "
             "Up_0..Up_{n_components-1}. Requires pandas.",
    )
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="Verbosity level (0=warnings only).")
    parsed = parser.parse_args(args)
    has_input = parsed.input_fpath is not None
    has_from_msa = parsed.from_msa is not None or parsed.seq_id is not None
    if has_input and has_from_msa:
        parser.error("-i/--input_fpath is mutually exclusive with "
                     "--from_msa / --seq_id.")
    if not has_input and not has_from_msa:
        parser.error("must provide -i/--input_fpath OR "
                     "--from_msa + --seq_id.")
    if has_from_msa and (
        parsed.from_msa is None or parsed.seq_id is None
    ):
        parser.error("--from_msa requires --seq_id (and vice versa).")
    return parsed


def _materialize_from_msa(msa_path: str, seq_id: str, outdir: str) -> str:
    """Extract a single record by ID from a (possibly aligned) FASTA,
    ungap it, and write a one-record FASTA inside outdir. Returns the
    path to the new file."""
    if not os.path.isfile(msa_path):
        raise FileNotFoundError(f"--from_msa not found: {msa_path}")
    target = None
    for rec in SeqIO.parse(msa_path, "fasta"):
        if rec.id == seq_id:
            target = rec
            break
    if target is None:
        raise KeyError(
            f"--seq_id {seq_id!r} not found in --from_msa {msa_path!r}"
        )
    raw = str(target.seq).replace("-", "").replace(".", "")
    if not raw:
        raise ValueError(
            f"Record {seq_id!r} in {msa_path!r} is empty after ungapping."
        )
    out_fpath = os.path.join(outdir, "from_msa_input.fasta")
    with open(out_fpath, "w") as f:
        f.write(f">{seq_id}\n{raw}\n")
    return out_fpath


def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    configure_logging(
        verbosity=args.verbosity,
        logfile=os.path.join(outdir, PROJECT_LOG_FNAME),
    )

    if args.input_fpath is not None:
        input_fpath = args.input_fpath
        if not os.path.isfile(input_fpath):
            raise FileNotFoundError(f"Input FASTA not found: {input_fpath}")
    else:
        input_fpath = _materialize_from_msa(
            args.from_msa, args.seq_id, outdir,
        )
        logger.info(
            "Extracted seq_id %r from %s into %s",
            args.seq_id, args.from_msa, input_fpath,
        )

    with open(os.path.join(outdir, PROJECT_ARGS_FNAME), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    aligner_kwargs = {
        "bin_path": args.align_bin,
        "threads": args.align_threads,
    }

    result = project_sequences(
        input_fpath,
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
        tsv_path = os.path.join(
            per_seq_dir,
            f"{_safe_filename_component(proj.seq_id)}_residues.tsv",
        )
        with open(tsv_path, "w") as f:
            f.write("ic_index\traw_residue_idx\tprocessed_col\tv_ica_loading\n")
            for ic_idx, (members, loadings, cols) in enumerate(
                zip(proj.ic_residues, proj.ic_loadings, proj.ic_processed_cols)
            ):
                for resi, loading, col in zip(members, loadings, cols):
                    f.write(
                        f"{ic_idx}\t{int(resi)}\t{int(col)}\t{float(loading)}\n"
                    )

    if args.save_dataframe:
        df = result.to_dataframe()
        df_path = os.path.join(outdir, "seq_projections.tsv")
        df.to_csv(df_path, sep="\t", index=False)
        logger.info("Wrote sequence projection DataFrame to %s", df_path)

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
