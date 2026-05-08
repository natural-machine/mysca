"""Project primary sequences onto an existing SCA result.

Given an input FASTA (or a record selected from a reference MSA) plus
the output directories of a prior ``sca-preprocess`` + ``sca-core`` run,
map each input sequence's raw residues onto the IC groups and write a
summary + per-sequence detail under the output directory.

-------------------------------------------------------------------------------
COMMAND LINE ARGUMENTS:

    -i --input_fpath : Path to an input FASTA, OR (when --raw is set) a
        literal amino-acid sequence string. Each FASTA record is
        projected independently. Records whose ID matches an entry in
        the reference MSA (``msa_orig.fasta-aln`` under --preprocessing)
        are resolved in-sample (no external alignment performed).
        Mutually exclusive with ``--from_msa`` / ``--seq_id``.
    --raw : Treat -i/--input_fpath as a literal amino-acid sequence
        rather than a path. The string is uppercased and whitespace-
        stripped; no alphabet validation is performed (non-canonical
        characters are passed through to the projector, which may
        emit its own WARNINGs via ``load_msa``). Empty or all-gap
        inputs are rejected. Materialized as a one-record FASTA
        inside the output directory and fed through the standard
        projection path.
    --from_msa : Path to a (possibly aligned) FASTA from which to
        extract a single record by --seq_id, ungap, and project. Useful
        for in-sample replay without the FASTA-surgery boilerplate.
        Requires --seq_id; mutually exclusive with -i.
    --seq_id : First whitespace-delimited token of the target record's
        header in --from_msa. Required with --from_msa. Also used as
        the record ID when --raw is set (default: ``raw_input``).
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
    --align_target : Which reference MSA to align against. ``original``
        (default) = unfiltered loaded MSA (``msa_orig.fasta-aln``,
        length L_orig). ``processed`` = post-preprocessing MSA (length
        L_proc, sliced from msa_obj_loaded by retained_sequences /
        retained_positions). ``processed`` is denser and typically
        yields cleaner alignments but more aggressively clips input
        residues that exceed the reference column count; check
        ``input_coverage_fraction`` in the output to detect.
    --align_bin : Explicit path to the alignment binary. For
        ``--aligner hmmalign`` this overrides the ``hmmalign`` binary;
        ``hmmbuild`` is resolved from PATH.
    --align_threads : Threads for the alignment tool. Default 1
        (unused by ``hmmalign``).
    --save_dataframe : also write ``seq_projections.tsv`` to outdir,
        with columns seq_id, aligned_sequence, raw_sequence, in_sample,
        Up_0..Up_{n_components-1}, gap_frac_ic_0..gap_frac_ic_{...},
        n_inform_ic_0..n_inform_ic_{...}. Requires pandas.
    --seq_metadata : Optional path to a TSV with a ``seq_id`` column
        plus arbitrary user-supplied columns. Persisted alongside
        projection outputs as ``sequence_metadata.tsv`` and merged into
        ``seq_projections.tsv`` via left-join on ``seq_id`` when
        ``--save_dataframe`` is set. Mirrors sca-core's ``--seq_metadata``.
    --plot / --no-plot : write projection plots to ``outdir/images/``.
        Default: on. Pass ``--no-plot`` to skip plot generation
        entirely (no ``images/`` directory is created). Mirrors
        sca-core's ``--plot``/``--no-plot``.
    --seq_proj_axes : one or more ``i,j`` axis pairs (zero-indexed) for
        the sequence-projection scatter plot(s). Default: ``0,1``.
        Pairs that exceed ``n_components`` are skipped with a warning.
    --seq_proj_color_by : Optional column name in ``--seq_metadata`` to
        color the projection plot(s) by. Numeric → colorbar,
        categorical → legend. Mirrors sca-core's
        ``--seq_proj_color_by``; ignored with a warning when
        ``--seq_metadata`` is missing or the column is absent.

-------------------------------------------------------------------------------
OUTPUTS:

projection.json
    Top-level dict containing run args plus a list of per-sequence
    dicts: seq_id, raw_sequence, aligned_sequence,
    residue_by_processed_col (length L_proc), ic_residues (per IC
    raw-residue indices), ic_loadings, ic_processed_cols, in_sample,
    up_score (length n_components — the sequence's Uᵖ row, or None
    when the source SCAResults lacks the eigendecomposition fields),
    gap_fraction_per_ic (length n_components — fraction of each IC's
    training-time support that is gapped or non-canonical in this
    projection; 0.0 means full coverage),
    informative_positions_per_ic (length n_components — count of
    positions in each IC's support that contribute non-zero mass to
    the Uᵖ math), align_target ('original' or 'processed' — which
    reference MSA the aligner used; affects len(aligned_sequence)),
    n_input_residues_dropped (count of input residues that did not
    survive alignment), input_coverage_fraction (fraction of the
    user-provided input that survived alignment; 1.0 = no clipping).

per_sequence/<seqid>_residues.tsv
    One row per (IC, residue) pairing for readable inspection.

projection_args.json
    Mapping from CLI argument to value.

projection.log
    Run log.

raw_input.fasta (only when --raw is set)
    Materialized one-record FASTA for the literal sequence string
    passed via -i. The record's header is --seq_id (default
    'raw_input').

from_msa_input.fasta (only when --from_msa is used)
    Materialized one-record FASTA for the record extracted from
    --from_msa.

_align_workdir/processed_reference.fasta-aln (only when --align_target
    processed is set AND at least one record needed alignment)
    Materialized character-space FASTA of the processed MSA (rows =
    retained_sequences, cols = retained_positions). Provided for
    transparency / debug; safe to delete.

seq_projections.tsv (only when --save_dataframe)
    Tab-separated table: seq_id, aligned_sequence, raw_sequence,
    in_sample, Up_0..Up_{n_components-1},
    gap_frac_ic_0..gap_frac_ic_{n_components-1},
    n_inform_ic_0..n_inform_ic_{n_components-1}. With
    --seq_metadata, the metadata's non-`seq_id` columns are merged in
    via left-join on `seq_id`.

images/ (only when --plot, the default)
    One ``seq_proj_ic{i}v{j}[_by_<col>].png`` per axis pair from
    ``--seq_proj_axes`` (default ``0,1``), plotting the per-sequence
    Uᵖ scores. Optionally colored by ``--seq_proj_color_by`` (numeric
    column → colorbar, categorical → legend).

sequence_metadata.tsv (only when --seq_metadata is supplied)
    Verbatim copy of the user-supplied per-sequence metadata TSV.
    Carries a `seq_id` column plus arbitrary user columns.

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

    # Raw-string input: project a single sequence passed on the command
    # line. --seq_id is optional; defaults to "raw_input".
    sca-project -i ACDEFGHIKLMNPQRSTVWY --raw \\
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

import numpy as np
from Bio import SeqIO

from mysca.logging_config import configure_logging
from mysca.pl import plot_seq_projection_2d, resolve_color_values
from mysca.project import project_sequences, ALIGNERS
from mysca.project.projection import ALIGN_TARGET_CHOICES

DEFAULT_RAW_SEQ_ID = "raw_input"


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


def _parse_axis_pair(token: str) -> tuple[int, int]:
    """argparse type for ``--seq_proj_axes``: parse one ``i,j`` token."""
    parts = token.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--seq_proj_axes expects 'i,j' tokens; got {token!r}"
        )
    try:
        i, j = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--seq_proj_axes expects integer indices; got {token!r}"
        )
    if i < 0 or j < 0 or i == j:
        raise argparse.ArgumentTypeError(
            f"--seq_proj_axes expects distinct non-negative indices; "
            f"got {token!r}"
        )
    return (i, j)


def _resolve_seq_proj_color(sequence_metadata, seq_ids, column):
    """Mirror sca-core's argparse-warning paths for --seq_proj_color_by.

    Returns (color_values, color_label) — both None when the column
    can't be resolved (with a warning logged), or a numpy array + the
    column name when it can.
    """
    if sequence_metadata is None:
        logger.warning(
            "--seq_proj_color_by=%r ignored: --seq_metadata not "
            "supplied.", column,
        )
        return None, None
    if column not in sequence_metadata.columns:
        logger.warning(
            "--seq_proj_color_by=%r ignored: column not found. "
            "Available: %s",
            column, list(sequence_metadata.columns),
        )
        return None, None
    return resolve_color_values(sequence_metadata, seq_ids, column), column


def _emit_seq_proj_plots(result, outdir, args):
    """Render one PNG per ``--seq_proj_axes`` pair into ``<outdir>/images/``.

    Skips silently with a warning if any projection lacks ``up_score``
    (legacy sca-core bundles without the eigendecomposition fields).
    """
    if any(p.up_score is None for p in result.projections):
        logger.warning(
            "Skipping projection plots: up_score is None on some "
            "projections (the source SCAResults likely lacks the "
            "eigendecomposition fields)."
        )
        return

    up = np.array(
        [p.up_score for p in result.projections], dtype=float,
    )
    seq_ids = [p.seq_id for p in result.projections]
    n_components = up.shape[1]

    color_values, color_label = None, None
    if args.seq_proj_color_by is not None:
        color_values, color_label = _resolve_seq_proj_color(
            result.sequence_metadata, seq_ids, args.seq_proj_color_by,
        )

    imgdir = os.path.join(outdir, "images")
    os.makedirs(imgdir, exist_ok=True)
    n_emitted = 0
    for axi, axj in args.seq_proj_axes:
        if axi >= n_components or axj >= n_components:
            logger.warning(
                "Skipping seq_proj plot (%d, %d): only n_components=%d "
                "ICs available.", axi, axj, n_components,
            )
            continue
        plot_seq_projection_2d(
            up, (axi, axj), imgdir,
            color_values=color_values, color_label=color_label,
        )
        n_emitted += 1
    if n_emitted == 0:
        logger.warning(
            "No projection plots emitted (every requested axis pair "
            "exceeded n_components=%d).", n_components,
        )
    else:
        logger.info(
            "Wrote %d projection plot(s) to %s", n_emitted, imgdir,
        )


def parse_args(args):
    parser = argparse.ArgumentParser(
        description=(
            "Project primary sequences onto an existing SCA result "
            "(in-sample short-circuit + out-of-sample alignment)."
        ),
    )
    parser.add_argument(
        "-i", "--input_fpath", type=str, default=None,
        help="Path to an input FASTA of sequences to project, OR (when "
        "--raw is set) a literal amino-acid sequence string. Mutually "
        "exclusive with --from_msa / --seq_id.",
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Interpret -i/--input_fpath as a literal amino-acid "
        "sequence string instead of a file path. The string is "
        "uppercased and whitespace-stripped; no alphabet validation "
        "is performed. Empty / all-gap inputs are still rejected. "
        "The record's seq_id defaults to 'raw_input' but can be "
        "overridden with --seq_id.",
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
        "to extract from --from_msa. Required with --from_msa. Also "
        "used as the record's ID under --raw (default: 'raw_input').",
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
        "--align_target", type=str, default="original",
        choices=list(ALIGN_TARGET_CHOICES),
        help="Which reference MSA to align against. 'original' "
        "(default) uses the unfiltered loaded MSA "
        "(msa_orig.fasta-aln, length L_orig). 'processed' uses the "
        "post-preprocessing MSA (length L_proc, sliced from "
        "msa_obj_loaded by retained_sequences and retained_positions). "
        "'processed' is denser and typically yields cleaner alignments "
        "but more aggressively clips input residues that exceed the "
        "reference column count; check input_coverage_fraction in the "
        "output to detect.",
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
             "Up_0..Up_{n_components-1}, "
             "gap_frac_ic_0..gap_frac_ic_{n_components-1}, "
             "n_inform_ic_0..n_inform_ic_{n_components-1}. "
             "Requires pandas.",
    )
    parser.add_argument(
        "--seq_metadata", type=str, default=None, metavar="TSV",
        help="Optional path to a TSV with a 'seq_id' column plus any "
             "number of additional columns (e.g. taxid, kingdom, "
             "phylum). Persisted alongside projection outputs as "
             "sequence_metadata.tsv and merged into seq_projections.tsv "
             "via left-join on seq_id when --save_dataframe is set. "
             "Mirrors sca-core's --seq_metadata.",
    )
    parser.add_argument(
        "--plot", default=True, action=argparse.BooleanOptionalAction,
        help="Write projection plots to outdir/images/ "
             "(seq_proj_ic{i}v{j}[_by_<col>].png, one per axis pair from "
             "--seq_proj_axes). Default: on. Pass --no-plot to skip plot "
             "generation entirely (no images/ directory is created).",
    )
    parser.add_argument(
        "--seq_proj_axes", type=_parse_axis_pair, nargs="+",
        default=[(0, 1)], metavar="I,J",
        help="One or more 'i,j' axis pairs (zero-indexed) for the "
             "sequence-projection scatter plot(s). Default: '0,1'. "
             "Pairs that exceed n_components are skipped with a warning.",
    )
    parser.add_argument(
        "--seq_proj_color_by", type=str, default=None, metavar="COLUMN",
        help="Optional column name in --seq_metadata to color the "
             "seq_proj_ic*.png plot(s) by. Numeric columns get a "
             "colorbar; categorical columns get a legend. Mirrors "
             "sca-core's --seq_proj_color_by; ignored with a warning "
             "when --seq_metadata is missing or the column is absent.",
    )
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="Verbosity level (0=warnings only).")
    parsed = parser.parse_args(args)
    has_input = parsed.input_fpath is not None
    has_from_msa = parsed.from_msa is not None
    if parsed.raw:
        if not has_input:
            parser.error("--raw requires -i/--input_fpath (the sequence "
                         "string).")
        if has_from_msa:
            parser.error("--raw is mutually exclusive with --from_msa.")
        return parsed
    # Without --raw: keep the original FASTA-or-from_msa contract.
    has_seq_id = parsed.seq_id is not None
    if has_input and (has_from_msa or has_seq_id):
        parser.error("-i/--input_fpath is mutually exclusive with "
                     "--from_msa / --seq_id (unless --raw is set).")
    if not has_input and not (has_from_msa or has_seq_id):
        parser.error("must provide -i/--input_fpath OR "
                     "--from_msa + --seq_id (use --raw to pass a literal "
                     "sequence via -i).")
    if (has_from_msa or has_seq_id) and not (has_from_msa and has_seq_id):
        parser.error("--from_msa requires --seq_id (and vice versa).")
    return parsed


def _materialize_raw_input(seq: str, seq_id: str, outdir: str) -> str:
    """Normalize a raw amino-acid string and write it as a one-record
    FASTA inside outdir. Returns the path to the new file.

    Strips whitespace and uppercases. No alphabet validation: any
    character is forwarded to the projector, which handles symbol
    mapping (and may emit its own WARNINGs for non-canonical chars
    via the standard load_msa path). Empty (whitespace-only) and
    all-gap inputs still raise, since neither has anything to project.
    """
    cleaned = "".join(seq.split()).upper()
    if not cleaned:
        raise ValueError(
            "--raw input is empty (after whitespace stripping)."
        )
    if not cleaned.replace("-", ""):
        raise ValueError(
            "--raw input contains only gap characters; nothing to project."
        )
    out_fpath = os.path.join(outdir, "raw_input.fasta")
    with open(out_fpath, "w") as f:
        f.write(f">{seq_id}\n{cleaned}\n")
    return out_fpath


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

    if args.raw:
        seq_id = args.seq_id or DEFAULT_RAW_SEQ_ID
        input_fpath = _materialize_raw_input(
            args.input_fpath, seq_id, outdir,
        )
        logger.info(
            "Materialized --raw sequence (seq_id=%r) into %s",
            seq_id, input_fpath,
        )
    elif args.input_fpath is not None:
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
        align_target=args.align_target,
        workdir=os.path.join(outdir, "_align_workdir"),
        aligner_kwargs=aligner_kwargs,
        seq_metadata_path=args.seq_metadata,
    )

    if result.sequence_metadata is not None:
        md_path = os.path.join(outdir, "sequence_metadata.tsv")
        result.sequence_metadata.to_csv(md_path, sep="\t", index=False)
        logger.info("Wrote sequence metadata to %s", md_path)

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

    if args.plot:
        _emit_seq_proj_plots(result, outdir, args)

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
