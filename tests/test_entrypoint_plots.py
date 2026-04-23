"""sca-plots replay CLI tests.

Exercise the replay entrypoint on outputs from prealign / preprocessing /
sca-core. The prealign cases are gated on `mafft` being on PATH.
"""

import os
import shutil

import pytest

from tests.conftest import DATDIR, TMPDIR, remove_dir

from mysca.run_plots import parse_args, main
from mysca.run_preprocessing import (
    parse_args as prep_parse_args,
    main as prep_main,
)
from mysca.run_sca import (
    parse_args as sca_parse_args,
    main as sca_main,
)
from mysca.run_prealign import (
    parse_args as prealign_parse_args,
    main as prealign_main,
)


_MAFFT = shutil.which("mafft") is not None
needs_mafft = pytest.mark.skipif(not _MAFFT, reason="mafft not on PATH")

PREALIGN_INPUT = f"{DATDIR}/seqs/seqs07.fasta"
PREP_ARGS = f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring6a.txt"
SCA_ARGS = f"{DATDIR}/entrypoint_tests/sca_run/argstrings/argstring6a.txt"


def _get_args(fpath):
    with open(fpath) as f:
        return f.readline().split(" ")


def _run_prep_and_sca(prep_outdir, sca_outdir):
    prep_args = prep_parse_args(_get_args(PREP_ARGS))
    prep_args.msa_fpath = f"{DATDIR}/{prep_args.msa_fpath}"
    prep_args.outdir = prep_outdir
    prep_args.verbosity = 0
    prep_main(prep_args)

    sca_args = sca_parse_args(_get_args(SCA_ARGS))
    sca_args.indir = prep_outdir
    sca_args.outdir = sca_outdir
    sca_args.background = f"{DATDIR}/{sca_args.background}"
    sca_args.verbosity = 0
    sca_args.n_boot = 2
    sca_args.seed = 42
    # Force kstar >= 3 so the replayed EV/IC sweeps (axj up to 2) have
    # enough columns. The msa06 fixture is tiny (5 positions) and would
    # otherwise produce kstar=1, skipping every axj>=1 plot.
    sca_args.kstar = 3
    sca_args.n_components = 3
    sca_main(sca_args)


def test_parse_args_requires_a_stage():
    """With no stage flag, argparse should fail via SystemExit."""
    with pytest.raises(SystemExit):
        parse_args([])


@needs_mafft
def test_replay_prealign_emits_filter_history_png():
    prealign_dir = f"{TMPDIR}/plots_replay_prealign"
    if os.path.isdir(prealign_dir):
        remove_dir(prealign_dir)

    prealign_main(prealign_parse_args([
        "-i", PREALIGN_INPUT,
        "-o", prealign_dir,
        "-v", "0",
    ]))
    assert os.path.isfile(
        os.path.join(prealign_dir, "filter_history.json")
    )
    images = os.path.join(prealign_dir, "images")
    assert not os.path.isdir(images), (
        "prealign without --plot should not write images/ dir"
    )

    main(parse_args([
        "--prealign", prealign_dir,
        "-v", "0",
    ]))
    assert os.path.isfile(
        os.path.join(images, "prealign_filter_history.png")
    )

    remove_dir(prealign_dir)


def test_replay_preprocessing_emits_all_plots():
    prep_dir = f"{TMPDIR}/plots_replay_prep"
    if os.path.isdir(prep_dir):
        remove_dir(prep_dir)

    prep_args = prep_parse_args(_get_args(PREP_ARGS))
    prep_args.msa_fpath = f"{DATDIR}/{prep_args.msa_fpath}"
    prep_args.outdir = prep_dir
    prep_args.verbosity = 0
    prep_main(prep_args)

    assert not os.path.isdir(os.path.join(prep_dir, "images"))

    main(parse_args([
        "--preprocessing", prep_dir,
        "-v", "0",
    ]))
    imgdir = os.path.join(prep_dir, "images")
    for fname in (
        "filter_history.png",
        "filter_distributions.png",
        "sequence_similarity.png",
    ):
        assert os.path.isfile(os.path.join(imgdir, fname)), (
            f"Expected {fname} under {imgdir}"
        )

    remove_dir(prep_dir)


def test_replay_scacore_emits_expected_plots():
    prep_dir = f"{TMPDIR}/plots_replay_sca_prep"
    sca_dir = f"{TMPDIR}/plots_replay_sca"
    for d in (prep_dir, sca_dir):
        if os.path.isdir(d):
            remove_dir(d)

    _run_prep_and_sca(prep_dir, sca_dir)

    sca_imgdir = os.path.join(sca_dir, "images")
    shutil.rmtree(sca_imgdir, ignore_errors=True)

    # Pass --preprocessing alongside --scacore so positional-conservation
    # plots can resolve retained_positions + original MSA length.
    main(parse_args([
        "--scacore", sca_dir,
        "--preprocessing", prep_dir,
        "-v", "0",
    ]))
    for fname in (
        "dendrogram.png",
        "t_distributions.png",
        "conservation.png",
        "top_conservation.png",
        "positional_conservation.png",
        "sca_matrix.png",
        "sca_matrix_spectrum.png",
        "sca_matrix_spectrum_vs_null.png",
        "sca_matrix_important_subset.png",
    ):
        assert os.path.isfile(os.path.join(sca_imgdir, fname)), (
            f"Expected {fname} under {sca_imgdir}"
        )
    pngs = [f for f in os.listdir(sca_imgdir) if f.endswith(".png")]
    assert any(f.startswith("ev") for f in pngs), (
        f"No EV scatter plots found in {sca_imgdir}: {pngs}"
    )
    assert any(f.startswith("ic") for f in pngs), (
        f"No IC scatter plots found in {sca_imgdir}: {pngs}"
    )

    remove_dir(prep_dir)
    remove_dir(sca_dir)


def test_replay_scacore_without_preprocessing_skips_positional_conservation():
    """Without --preprocessing, top_conservation and positional_conservation
    can't be drawn (they need retained_positions + original MSA length).
    Plain conservation.png should still be produced."""
    prep_dir = f"{TMPDIR}/plots_replay_sca_noprep_prep"
    sca_dir = f"{TMPDIR}/plots_replay_sca_noprep"
    for d in (prep_dir, sca_dir):
        if os.path.isdir(d):
            remove_dir(d)

    _run_prep_and_sca(prep_dir, sca_dir)

    sca_imgdir = os.path.join(sca_dir, "images")
    shutil.rmtree(sca_imgdir, ignore_errors=True)

    main(parse_args([
        "--scacore", sca_dir,
        "-v", "0",
    ]))
    assert os.path.isfile(os.path.join(sca_imgdir, "conservation.png"))
    assert not os.path.isfile(
        os.path.join(sca_imgdir, "top_conservation.png")
    )
    assert not os.path.isfile(
        os.path.join(sca_imgdir, "positional_conservation.png")
    )

    remove_dir(prep_dir)
    remove_dir(sca_dir)


def test_imgdir_override_redirects_output():
    prep_dir = f"{TMPDIR}/plots_replay_imgdir_prep"
    sca_dir = f"{TMPDIR}/plots_replay_imgdir_sca"
    shared_imgdir = f"{TMPDIR}/plots_replay_shared_images"
    for d in (prep_dir, sca_dir, shared_imgdir):
        if os.path.isdir(d):
            remove_dir(d)

    _run_prep_and_sca(prep_dir, sca_dir)
    shutil.rmtree(os.path.join(sca_dir, "images"), ignore_errors=True)

    main(parse_args([
        "--scacore", sca_dir,
        "--imgdir", shared_imgdir,
        "-v", "0",
    ]))
    assert os.path.isfile(os.path.join(shared_imgdir, "dendrogram.png"))
    assert not os.path.isdir(os.path.join(sca_dir, "images")), (
        "With --imgdir override, the stage's default images/ dir should "
        "not be created."
    )

    remove_dir(prep_dir)
    remove_dir(sca_dir)
    remove_dir(shared_imgdir)


def test_missing_stage_dir_raises():
    with pytest.raises(FileNotFoundError):
        main(parse_args([
            "--preprocessing", f"{TMPDIR}/definitely_does_not_exist_xyz",
            "-v", "0",
        ]))
