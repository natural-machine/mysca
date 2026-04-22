"""Prealign entrypoint tests.

Gated on `mafft` (and `mmseqs` where needed) being resolvable on PATH so CI
without the binaries skips cleanly.
"""

import os
import shutil

import pytest
from Bio import AlignIO, SeqIO

from tests.conftest import DATDIR, TMPDIR, remove_dir

from mysca.run_prealign import parse_args, main
from mysca import run_preprocessing as run_preprocessing_mod


INPUT_FASTA = f"{DATDIR}/seqs/seqs07.fasta"

_MAFFT = shutil.which("mafft") is not None
_MMSEQS = shutil.which("mmseqs") is not None
needs_mafft = pytest.mark.skipif(not _MAFFT, reason="mafft not on PATH")
needs_mmseqs = pytest.mark.skipif(not _MMSEQS, reason="mmseqs not on PATH")


def _aligned_lengths(fpath):
    with open(fpath) as f:
        return {len(str(rec.seq)) for rec in SeqIO.parse(f, "fasta")}


def _record_count(fpath):
    with open(fpath) as f:
        return sum(1 for _ in SeqIO.parse(f, "fasta"))


@needs_mafft
def test_align_only():
    outdir = f"{TMPDIR}/prealign_align_only"
    if os.path.isdir(outdir):
        remove_dir(outdir)
    args = parse_args([
        "-i", INPUT_FASTA,
        "-o", outdir,
        "-v", "0",
    ])
    main(args)

    aligned = os.path.join(outdir, "aligned.fasta")
    assert os.path.isfile(aligned)
    lengths = _aligned_lengths(aligned)
    assert len(lengths) == 1, f"Non-uniform aligned lengths: {lengths}"
    assert _record_count(aligned) == _record_count(INPUT_FASTA)

    assert os.path.isfile(os.path.join(outdir, "prealign_args.json"))
    assert os.path.isfile(os.path.join(outdir, "prealign.log"))

    remove_dir(outdir)


@needs_mafft
@needs_mmseqs
def test_cluster_then_align():
    outdir = f"{TMPDIR}/prealign_cluster"
    if os.path.isdir(outdir):
        remove_dir(outdir)
    args = parse_args([
        "-i", INPUT_FASTA,
        "-o", outdir,
        "--cluster", "mmseqs2",
        "--cluster_min_seq_id", "0.5",
        "-v", "0",
    ])
    main(args)

    clustered = os.path.join(outdir, "clustered.fasta")
    aligned = os.path.join(outdir, "aligned.fasta")
    assert os.path.isfile(clustered)
    assert os.path.isfile(aligned)

    n_in = _record_count(INPUT_FASTA)
    n_clust = _record_count(clustered)
    n_aligned = _record_count(aligned)
    assert 0 < n_clust <= n_in
    assert n_aligned == n_clust

    lengths = _aligned_lengths(aligned)
    assert len(lengths) == 1, f"Non-uniform aligned lengths: {lengths}"

    remove_dir(outdir)


@needs_mafft
def test_align_stockholm_output():
    outdir = f"{TMPDIR}/prealign_sto_out"
    if os.path.isdir(outdir):
        remove_dir(outdir)
    args = parse_args([
        "-i", INPUT_FASTA,
        "-o", outdir,
        "--output_format", "stockholm",
        "-v", "0",
    ])
    main(args)

    aligned = os.path.join(outdir, "aligned.sto")
    assert os.path.isfile(aligned)
    assert not os.path.isfile(os.path.join(outdir, "aligned.fasta"))

    with open(aligned) as f:
        aln = AlignIO.read(f, "stockholm")
    assert len(aln) == _record_count(INPUT_FASTA)
    assert len({len(rec.seq) for rec in aln}) == 1

    remove_dir(outdir)


@needs_mafft
def test_end_to_end_with_preprocess():
    prealign_outdir = f"{TMPDIR}/prealign_chain"
    preprocess_outdir = f"{TMPDIR}/prealign_chain_preprocess"
    for d in (prealign_outdir, preprocess_outdir):
        if os.path.isdir(d):
            remove_dir(d)

    main(parse_args([
        "-i", INPUT_FASTA,
        "-o", prealign_outdir,
        "-v", "0",
    ]))
    aligned = os.path.join(prealign_outdir, "aligned.fasta")
    assert os.path.isfile(aligned)

    pp_args = run_preprocessing_mod.parse_args([
        "-i", aligned,
        "-o", preprocess_outdir,
        "-v", "0",
    ])
    run_preprocessing_mod.main(pp_args)

    assert os.path.isfile(
        os.path.join(preprocess_outdir, "preprocessing_results.npz")
    )

    remove_dir(prealign_outdir)
    remove_dir(preprocess_outdir)


def test_missing_binary_fails_fast():
    """Explicit --align_bin pointing at a non-existent binary must fail fast."""
    outdir = f"{TMPDIR}/prealign_missing_bin"
    if os.path.isdir(outdir):
        remove_dir(outdir)
    args = parse_args([
        "-i", INPUT_FASTA,
        "-o", outdir,
        "--align_bin", "/nonexistent/mafft",
        "-v", "0",
    ])
    with pytest.raises(FileNotFoundError):
        main(args)
    remove_dir(outdir)
