"""Tests for the mysca.project subpackage and sca-project CLI.

Library tests exercise:
- In-sample roundtrip: project each training sequence and verify its
  per-IC residue set matches what sca-core already persisted in
  ``statsectors_msa.npz``.
- In-sample short-circuit: no external aligner is invoked when every
  input ID is already in the reference MSA.
- Aligner dispatch: unknown method raises; registered-but-not-
  implemented hmmalign raises NotImplementedError.
- Out-of-sample alignment (gated on mafft on PATH): hand-build a
  sequence that differs from a training sequence by 1–2 residues and
  confirm the result is internally consistent.

CLI test runs the entrypoint and asserts on-disk artifacts.
"""

import json
import os
import shutil

import numpy as np
import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from tests.conftest import DATDIR, TMPDIR, remove_dir

from mysca.project import (
    ALIGNERS,
    align_to_msa,
    project_sequences,
    ProjectionResult,
    SequenceProjection,
)
from mysca.project.alignment import _hmmalign
from mysca.results import PreprocessingResults, SCAResults
from mysca.run_preprocessing import (
    parse_args as prep_parse_args,
    main as prep_main,
)
from mysca.run_sca import parse_args as sca_parse_args, main as sca_main
from mysca.run_project import (
    parse_args as project_parse_args,
    main as project_main,
)


_MAFFT = shutil.which("mafft") is not None
needs_mafft = pytest.mark.skipif(not _MAFFT, reason="mafft not on PATH")


PREP_ARGS = f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring6a.txt"
SCA_ARGS = f"{DATDIR}/entrypoint_tests/sca_run/argstrings/argstring6a.txt"


def _read_argstring(fpath):
    with open(fpath) as f:
        return f.readline().split(" ")


def _run_prep_and_sca(prep_outdir, sca_outdir, *, sectors_for="all"):
    prep_args = prep_parse_args(_read_argstring(PREP_ARGS))
    prep_args.msa_fpath = f"{DATDIR}/{prep_args.msa_fpath}"
    prep_args.outdir = prep_outdir
    prep_args.verbosity = 0
    prep_main(prep_args)

    sca_args = sca_parse_args(_read_argstring(SCA_ARGS))
    sca_args.indir = prep_outdir
    sca_args.outdir = sca_outdir
    sca_args.background = f"{DATDIR}/{sca_args.background}"
    sca_args.verbosity = 0
    sca_args.n_boot = 2
    sca_args.seed = 42
    sca_args.kstar = 3
    sca_args.n_components = 3
    sca_args.sectors_for = sectors_for
    sca_main(sca_args)


@pytest.fixture
def prep_and_sca_dirs(tmp_path_factory):
    prep_dir = str(tmp_path_factory.mktemp("project_prep"))
    sca_dir = str(tmp_path_factory.mktemp("project_sca"))
    _run_prep_and_sca(prep_dir, sca_dir, sectors_for="all")
    yield prep_dir, sca_dir


def test_align_to_msa_rejects_unknown_aligner(tmp_path):
    from Bio.Align import MultipleSeqAlignment
    msa = MultipleSeqAlignment([SeqRecord(Seq("AAA"), id="a")])
    dummy_fasta = tmp_path / "x.fasta"
    dummy_fasta.write_text(">x\nAAA\n")
    with pytest.raises(ValueError, match="Unknown aligner"):
        align_to_msa(
            str(dummy_fasta), msa, str(tmp_path),
            method="not_a_real_aligner",
        )


def test_hmmalign_backend_is_registered_but_unimplemented():
    assert "hmmalign" in ALIGNERS
    with pytest.raises(NotImplementedError):
        _hmmalign()


def test_project_in_sample_matches_statsectors_msa(prep_and_sca_dirs):
    """Projecting training sequences should reproduce statsectors_msa.npz
    entry-for-entry for the top-kstar ICs.
    """
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    sca = SCAResults.load(sca_dir)
    retained_ids = list(prep.retained_sequence_ids)

    # Write the retained sequences out as a FASTA for sca-project's
    # input (in-sample short-circuit applies to every record).
    msa_obj = prep.msa_obj_orig
    ids_to_row = {rec.id: i for i, rec in enumerate(msa_obj)}
    in_fasta = os.path.join(sca_dir, "_retained_input.fasta")
    with open(in_fasta, "w") as f:
        for sid in retained_ids:
            rec = msa_obj[ids_to_row[sid]]
            f.write(f">{sid}\n{str(rec.seq).replace('-', '')}\n")

    result = project_sequences(
        in_fasta,
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
        aligner="mafft_add",  # ignored because every record is in-sample
    )
    assert isinstance(result, ProjectionResult)
    assert all(p.in_sample for p in result.projections)

    # Roundtrip check against statsectors_msa.npz for the top-kstar ICs
    # (only those were expanded per sequence per the kstar scoping).
    kstar = sca.kstar
    stats = sca.statsectors_msa
    for proj in result.projections:
        for ic in range(kstar):
            key = f"group_{ic}_{proj.seq_id}"
            if key not in stats:
                continue
            expected = np.asarray(stats[key], dtype=int)
            got = np.asarray(proj.ic_memberships[ic], dtype=int)
            assert np.array_equal(np.sort(got), np.sort(expected)), (
                f"Mismatch for {proj.seq_id} IC {ic}: "
                f"project={got.tolist()} vs statsectors={expected.tolist()}"
            )


def test_project_in_sample_does_not_invoke_aligner(prep_and_sca_dirs, tmp_path):
    """When every input ID is already in msa_obj_orig, no aligner
    callable should be invoked (even if one would crash)."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    msa_obj = prep.msa_obj_orig

    # Build a one-sequence in-sample input.
    rec = msa_obj[0]
    in_fasta = tmp_path / "one.fasta"
    in_fasta.write_text(f">{rec.id}\n{str(rec.seq).replace('-', '')}\n")

    # Register a sentinel aligner that would fail loudly if invoked.
    ALIGNERS["__sentinel_fail__"] = lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("aligner was invoked for an in-sample record")
    )
    try:
        result = project_sequences(
            str(in_fasta),
            sca_result_dir=sca_dir,
            preproc_result_dir=prep_dir,
            aligner="__sentinel_fail__",
        )
    finally:
        ALIGNERS.pop("__sentinel_fail__", None)
    assert len(result.projections) == 1
    assert result.projections[0].in_sample


@needs_mafft
def test_project_out_of_sample_roundtrip(prep_and_sca_dirs, tmp_path):
    """Project a sequence that is the same as a training sequence
    modulo a re-id. The aligned row should match the original and
    ic_memberships should match statsectors_msa for that training seq.
    """
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    sca = SCAResults.load(sca_dir)
    msa_obj = prep.msa_obj_orig

    donor_id = prep.retained_sequence_ids[0]
    donor_rec = next(r for r in msa_obj if r.id == donor_id)
    donor_raw = str(donor_rec.seq).replace("-", "")

    # Feed the same sequence under a new ID that is not in the MSA,
    # forcing the out-of-sample path.
    new_id = "out_of_sample_copy"
    assert new_id not in {r.id for r in msa_obj}
    in_fasta = tmp_path / "oos.fasta"
    in_fasta.write_text(f">{new_id}\n{donor_raw}\n")

    result = project_sequences(
        str(in_fasta),
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
        aligner="mafft_add",
    )
    [proj] = result.projections
    assert not proj.in_sample
    assert proj.raw_sequence == donor_raw

    # ic_memberships for the out-of-sample copy should match what the
    # donor has in statsectors_msa, since it's the same raw sequence.
    kstar = sca.kstar
    stats = sca.statsectors_msa
    for ic in range(kstar):
        key = f"group_{ic}_{donor_id}"
        if key not in stats:
            continue
        expected = np.sort(np.asarray(stats[key], dtype=int))
        got = np.sort(np.asarray(proj.ic_memberships[ic], dtype=int))
        assert np.array_equal(got, expected), (
            f"Out-of-sample IC {ic} membership differs from in-sample "
            f"donor {donor_id}: got={got.tolist()} "
            f"expected={expected.tolist()}"
        )


def test_field_descriptions_cover_all_init_args():
    """Same coverage invariant we enforced on PreprocessingResults /
    SCAResults applies to ProjectionResult and SequenceProjection."""
    proj = SequenceProjection(
        seq_id="x",
        raw_sequence="A",
        aligned_sequence="A",
        residue_by_processed_col=np.array([0]),
        ic_memberships=[np.array([0])],
        ic_loadings=[np.array([0.1])],
        ic_processed_cols=[np.array([0])],
        in_sample=False,
    )
    assert set(vars(proj).keys()) == set(
        SequenceProjection.FIELD_DESCRIPTIONS.keys()
    )

    result = ProjectionResult(
        projections=[proj], args={}, n_components=1,
        n_retained_positions=1, original_length=1,
    )
    assert set(vars(result).keys()) == set(
        ProjectionResult.FIELD_DESCRIPTIONS.keys()
    )

    # Both info() outputs render without raising.
    assert "SequenceProjection" in proj.info()
    assert "ProjectionResult" in result.info()


def test_sca_project_cli_writes_expected_artifacts(prep_and_sca_dirs, tmp_path):
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    msa_obj = prep.msa_obj_orig

    # Two in-sample records.
    in_fasta = tmp_path / "cli_input.fasta"
    lines = []
    for rec in msa_obj[:2]:
        lines.append(f">{rec.id}\n{str(rec.seq).replace('-', '')}")
    in_fasta.write_text("\n".join(lines) + "\n")

    out_dir = str(tmp_path / "sca_project_out")
    args = project_parse_args([
        "-i", str(in_fasta),
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out_dir,
        "-v", "0",
    ])
    project_main(args)

    for fname in ("projection.json", "projection_args.json", "projection.log"):
        assert os.path.isfile(os.path.join(out_dir, fname))
    per_seq_dir = os.path.join(out_dir, "per_sequence")
    assert os.path.isdir(per_seq_dir)
    tsvs = [f for f in os.listdir(per_seq_dir) if f.endswith(".tsv")]
    assert len(tsvs) == 2

    with open(os.path.join(out_dir, "projection.json")) as f:
        data = json.load(f)
    assert data["n_components"] >= 1
    assert len(data["projections"]) == 2
    for p in data["projections"]:
        assert p["in_sample"] is True
