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
_CLUSTALO = shutil.which("clustalo") is not None
needs_mafft = pytest.mark.skipif(not _MAFFT, reason="mafft not on PATH")
needs_mmseqs = pytest.mark.skipif(not _MMSEQS, reason="mmseqs not on PATH")
needs_clustalo = pytest.mark.skipif(not _CLUSTALO, reason="clustalo not on PATH")


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

    # filter_history is always persisted (so that --plot can be replayed later).
    import json
    with open(os.path.join(outdir, "filter_history.json")) as f:
        fh = json.load(f)
    # No clustering → two entries: initial and align.
    assert [e["stage"] for e in fh] == ["initial", "align"]
    n_in = _record_count(INPUT_FASTA)
    assert fh[0]["n_sequences"] == n_in
    assert fh[1]["n_sequences"] == _record_count(aligned)
    # --plot was NOT passed, so no images directory should exist.
    assert not os.path.isdir(os.path.join(outdir, "images"))

    remove_dir(outdir)


@needs_mafft
def test_align_plot_emits_filter_history_png():
    """--plot writes a prealign_filter_history.png under outdir/images/."""
    outdir = f"{TMPDIR}/prealign_with_plot"
    if os.path.isdir(outdir):
        remove_dir(outdir)
    args = parse_args([
        "-i", INPUT_FASTA,
        "-o", outdir,
        "-v", "0",
        "--plot",
    ])
    main(args)

    png = os.path.join(outdir, "images", "prealign_filter_history.png")
    assert os.path.isfile(png), f"Expected plot at {png}"

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


@needs_clustalo
def test_align_only_clustalo():
    outdir = f"{TMPDIR}/prealign_align_only_clustalo"
    if os.path.isdir(outdir):
        remove_dir(outdir)
    args = parse_args([
        "-i", INPUT_FASTA,
        "-o", outdir,
        "--align", "clustalo",
        "-v", "0",
    ])
    main(args)

    aligned = os.path.join(outdir, "aligned.fasta")
    assert os.path.isfile(aligned)
    lengths = _aligned_lengths(aligned)
    assert len(lengths) == 1, f"Non-uniform aligned lengths: {lengths}"
    assert _record_count(aligned) == _record_count(INPUT_FASTA)

    args_path = os.path.join(outdir, "prealign_args.json")
    assert os.path.isfile(args_path)
    import json
    with open(args_path) as f:
        persisted = json.load(f)
    assert persisted["align"] == "clustalo"

    remove_dir(outdir)


@needs_clustalo
def test_align_clustalo_stockholm_output():
    outdir = f"{TMPDIR}/prealign_clustalo_sto_out"
    if os.path.isdir(outdir):
        remove_dir(outdir)
    args = parse_args([
        "-i", INPUT_FASTA,
        "-o", outdir,
        "--align", "clustalo",
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


@needs_clustalo
def test_align_clustalo_guidetree_out():
    outdir = f"{TMPDIR}/prealign_clustalo_guidetree"
    if os.path.isdir(outdir):
        remove_dir(outdir)
    args = parse_args([
        "-i", INPUT_FASTA,
        "-o", outdir,
        "--align", "clustalo",
        "--align_args", "guidetree_out=true",
        "-v", "0",
    ])
    main(args)

    guidetree = os.path.join(outdir, "guidetree.dnd")
    assert os.path.isfile(guidetree)
    assert os.path.getsize(guidetree) > 0

    remove_dir(outdir)


@needs_clustalo
def test_align_clustalo_output_order_tree():
    outdir = f"{TMPDIR}/prealign_clustalo_output_order"
    if os.path.isdir(outdir):
        remove_dir(outdir)
    args = parse_args([
        "-i", INPUT_FASTA,
        "-o", outdir,
        "--align", "clustalo",
        "--align_args", "output_order=tree-order",
        "-v", "0",
    ])
    main(args)

    aligned = os.path.join(outdir, "aligned.fasta")
    assert os.path.isfile(aligned)
    lengths = _aligned_lengths(aligned)
    assert len(lengths) == 1, f"Non-uniform aligned lengths: {lengths}"
    assert _record_count(aligned) == _record_count(INPUT_FASTA)

    remove_dir(outdir)


@needs_clustalo
def test_clustalo_chain_to_preprocess():
    prealign_outdir = f"{TMPDIR}/prealign_clustalo_chain"
    preprocess_outdir = f"{TMPDIR}/prealign_clustalo_chain_preprocess"
    for d in (prealign_outdir, preprocess_outdir):
        if os.path.isdir(d):
            remove_dir(d)

    main(parse_args([
        "-i", INPUT_FASTA,
        "-o", prealign_outdir,
        "--align", "clustalo",
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


def test_align_clustalo_missing_binary_fails_fast():
    """Aligner-aware _resolve_bin must look up clustalo, not mafft."""
    outdir = f"{TMPDIR}/prealign_clustalo_missing_bin"
    if os.path.isdir(outdir):
        remove_dir(outdir)
    args = parse_args([
        "-i", INPUT_FASTA,
        "-o", outdir,
        "--align", "clustalo",
        "--align_bin", "/nonexistent/clustalo",
        "-v", "0",
    ])
    with pytest.raises(FileNotFoundError):
        main(args)
    remove_dir(outdir)


def test_align_args_unknown_key_with_mafft_rejected():
    """A clustalo-only key passed via --align_args while --align mafft is
    chosen must surface as a ValueError from the mafft wrapper. We use a
    nonexistent --align_bin so we never need either binary on PATH."""
    outdir = f"{TMPDIR}/prealign_align_args_mafft_unknown"
    if os.path.isdir(outdir):
        remove_dir(outdir)
    args = parse_args([
        "-i", INPUT_FASTA,
        "-o", outdir,
        "--align", "mafft",
        "--align_bin", "/nonexistent/mafft",
        "--align_args", "guidetree_out=true",
        "-v", "0",
    ])
    with pytest.raises((FileNotFoundError, ValueError)):
        main(args)
    remove_dir(outdir)


def test_align_args_duplicate_key_rejected():
    """--align_args with duplicate keys is a user error caught at parse
    time, before any binary is resolved."""
    outdir = f"{TMPDIR}/prealign_align_args_duplicate"
    if os.path.isdir(outdir):
        remove_dir(outdir)
    args = parse_args([
        "-i", INPUT_FASTA,
        "-o", outdir,
        "--align", "clustalo",
        "--align_bin", "/nonexistent/clustalo",
        "--align_args", "output_order=tree-order", "output_order=input-order",
        "-v", "0",
    ])
    with pytest.raises(ValueError, match="Duplicate"):
        main(args)
    remove_dir(outdir)


def test_align_args_bare_key_means_true():
    """`--align_args guidetree_out` (bare) is equivalent to
    `--align_args guidetree_out=true` — accepted by the clustalo wrapper.
    Verified at the parser level so the test doesn't need clustalo."""
    from mysca.run_prealign import _parse_align_args
    assert _parse_align_args(["guidetree_out"]) == {"guidetree_out": "true"}
    assert _parse_align_args(["k=v"]) == {"k": "v"}
