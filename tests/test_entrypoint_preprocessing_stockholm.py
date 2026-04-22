"""Stockholm-input entrypoint tests for sca-preprocess.

The default input path (FASTA) is exercised by test_entrypoint_preprocessing.py;
this file focuses on the --input_format stockholm path.
"""

import os

import numpy as np
from Bio import AlignIO

from tests.conftest import DATDIR, TMPDIR, remove_dir

from mysca.run_preprocessing import parse_args, main


def _convert_fasta_to_stockholm(src_fasta: str, dst_sto: str):
    os.makedirs(os.path.dirname(os.path.abspath(dst_sto)), exist_ok=True)
    aln = AlignIO.read(src_fasta, "fasta")
    AlignIO.write(aln, dst_sto, "stockholm")


def test_preprocess_stockholm_input_matches_fasta():
    """Running sca-preprocess on a Stockholm copy of an MSA should yield the
    same preprocessing results as running it on the original FASTA."""
    src_fasta = f"{DATDIR}/msas/msa07.faa"

    fasta_outdir = f"{TMPDIR}/preprocess_fasta_input"
    sto_outdir = f"{TMPDIR}/preprocess_stockholm_input"
    sto_msa = f"{TMPDIR}/preprocess_stockholm_input_src/msa07.sto"
    for d in (fasta_outdir, sto_outdir, os.path.dirname(sto_msa)):
        if os.path.isdir(d):
            remove_dir(d)

    _convert_fasta_to_stockholm(src_fasta, sto_msa)

    shared = [
        "--gap_truncation_thresh", "0.4",
        "--sequence_gap_thresh", "0.2",
        "--reference_similarity_thresh", "0.2",
        "--sequence_similarity_thresh", "0.8",
        "--position_gap_thresh", "0.2",
        "-v", "0",
    ]
    main(parse_args(["-i", src_fasta, "-o", fasta_outdir] + shared))
    main(parse_args([
        "-i", sto_msa, "-o", sto_outdir, "--input_format", "stockholm",
    ] + shared))

    fasta_res = np.load(os.path.join(fasta_outdir, "preprocessing_results.npz"))
    sto_res = np.load(os.path.join(sto_outdir, "preprocessing_results.npz"))
    for key in ("msa", "retained_sequences", "retained_positions",
                "sequence_weights"):
        np.testing.assert_array_equal(
            fasta_res[key], sto_res[key],
            err_msg=f"Mismatch between fasta- and stockholm-input runs for {key}",
        )

    remove_dir(fasta_outdir)
    remove_dir(sto_outdir)
    remove_dir(os.path.dirname(sto_msa))
