"""Tests for the mysca.project subpackage and sca-project CLI.

Library tests exercise:
- In-sample roundtrip: project each training sequence and verify its
  per-IC residue set matches what sca-core already persisted in
  ``statsectors_msa.npz``.
- In-sample short-circuit: no external aligner is invoked when every
  input ID is already in the reference MSA.
- Aligner dispatch: unknown method raises.
- Out-of-sample alignment (gated on mafft / HMMER on PATH): hand-build
  a sequence that differs from a training sequence by 1–2 residues and
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
from mysca.project.projection import _gapless, _residue_indices_for_aligned
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
_HMMER = (
    shutil.which("hmmalign") is not None
    and shutil.which("hmmbuild") is not None
)
needs_mafft = pytest.mark.skipif(not _MAFFT, reason="mafft not on PATH")
needs_hmmer = pytest.mark.skipif(
    not _HMMER, reason="hmmbuild/hmmalign not on PATH",
)


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


def test_hmmalign_backend_is_registered():
    assert "hmmalign" in ALIGNERS


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


def _out_of_sample_roundtrip(prep_dir, sca_dir, tmp_path, aligner):
    """Shared body: project a sequence under a new ID (forcing the
    out-of-sample path) that is byte-identical to a training sequence;
    per-IC memberships must match statsectors_msa for the donor."""
    prep = PreprocessingResults.load(prep_dir)
    sca = SCAResults.load(sca_dir)
    msa_obj = prep.msa_obj_orig

    donor_id = prep.retained_sequence_ids[0]
    donor_rec = next(r for r in msa_obj if r.id == donor_id)
    donor_raw = str(donor_rec.seq).replace("-", "")

    new_id = f"out_of_sample_copy_{aligner}"
    assert new_id not in {r.id for r in msa_obj}
    in_fasta = tmp_path / f"oos_{aligner}.fasta"
    in_fasta.write_text(f">{new_id}\n{donor_raw}\n")

    result = project_sequences(
        str(in_fasta),
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
        aligner=aligner,
    )
    [proj] = result.projections
    assert not proj.in_sample
    assert proj.raw_sequence == donor_raw

    kstar = sca.kstar
    stats = sca.statsectors_msa
    for ic in range(kstar):
        key = f"group_{ic}_{donor_id}"
        if key not in stats:
            continue
        expected = np.sort(np.asarray(stats[key], dtype=int))
        got = np.sort(np.asarray(proj.ic_memberships[ic], dtype=int))
        assert np.array_equal(got, expected), (
            f"[{aligner}] Out-of-sample IC {ic} membership differs from "
            f"donor {donor_id}: got={got.tolist()} "
            f"expected={expected.tolist()}"
        )


@needs_mafft
def test_project_out_of_sample_roundtrip_mafft(prep_and_sca_dirs, tmp_path):
    prep_dir, sca_dir = prep_and_sca_dirs
    _out_of_sample_roundtrip(prep_dir, sca_dir, tmp_path, aligner="mafft_add")


@needs_hmmer
def test_project_out_of_sample_roundtrip_hmmer(prep_and_sca_dirs, tmp_path):
    prep_dir, sca_dir = prep_and_sca_dirs
    _out_of_sample_roundtrip(prep_dir, sca_dir, tmp_path, aligner="hmmalign")


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
        input_residue_indices=[0],
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


# ----------------------------------------------------------------------
# Raw-sequence / aligned-sequence consistency invariant.
#
# For ic_memberships[i] (raw residue indices) to dereference correctly
# into raw_sequence, raw_sequence must count the *same* residues that
# residue_by_processed_col counts — i.e. the non-gap characters of
# aligned_sequence, in the same order. The in-sample path derives
# raw = aligned.replace("-", "") directly; these tests guard that the
# out-of-sample path does the same, even when the aligner has to drop
# residues (insertions) or introduce gaps (deletions).
# ----------------------------------------------------------------------


def _assert_raw_matches_aligned_gapless(proj):
    gapless = _gapless(proj.aligned_sequence)
    assert proj.raw_sequence == gapless, (
        f"raw_sequence must equal _gapless(aligned_sequence) (strips "
        f"both '-' and '.').\n"
        f"  raw:     {proj.raw_sequence!r}\n"
        f"  aligned: {proj.aligned_sequence!r}\n"
        f"  gapless: {gapless!r}"
    )


def _choose_insertion_residue(donor_raw, alphabet="ACDE"):
    """Pick a residue to insert that's in the MSA's alphabet and
    differs from the immediate neighbors (reduces the chance the
    aligner merges it into an adjacent match)."""
    mid = len(donor_raw) // 2
    flank = set(donor_raw[max(0, mid - 1):mid + 2])
    for c in alphabet:
        if c not in flank:
            return c
    return alphabet[0]


@pytest.mark.parametrize("aligner", [
    pytest.param("mafft_add", marks=needs_mafft),
    pytest.param("hmmalign", marks=needs_hmmer),
])
def test_out_of_sample_invariant_on_insertion(prep_and_sca_dirs, tmp_path, aligner):
    """Insert one extra residue into the middle of a training sequence
    (forcing the aligner to drop it under --keeplength / match-only
    output). The resulting projection must still satisfy
    raw_sequence == aligned_sequence.replace("-", "")."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    donor_id = prep.retained_sequence_ids[0]
    donor = next(r for r in prep.msa_obj_orig if r.id == donor_id)
    donor_raw = str(donor.seq).replace("-", "")
    if len(donor_raw) < 3:
        pytest.skip("donor sequence too short for a middle insertion")

    inserted = _choose_insertion_residue(donor_raw)
    mid = len(donor_raw) // 2
    new_seq = donor_raw[:mid] + inserted + donor_raw[mid:]
    assert len(new_seq) == len(donor_raw) + 1

    new_id = f"oos_insertion_{aligner}"
    in_fasta = tmp_path / f"{new_id}.fasta"
    in_fasta.write_text(f">{new_id}\n{new_seq}\n")

    result = project_sequences(
        str(in_fasta),
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
        aligner=aligner,
    )
    [proj] = result.projections
    assert not proj.in_sample
    _assert_raw_matches_aligned_gapless(proj)


@pytest.mark.parametrize("aligner", [
    pytest.param("mafft_add", marks=needs_mafft),
    pytest.param("hmmalign", marks=needs_hmmer),
])
def test_out_of_sample_invariant_on_deletion(prep_and_sca_dirs, tmp_path, aligner):
    """Delete one residue from the middle of a training sequence. The
    aligner should introduce a gap at that column; the invariant must
    continue to hold so raw indices still dereference correctly."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    donor_id = prep.retained_sequence_ids[0]
    donor = next(r for r in prep.msa_obj_orig if r.id == donor_id)
    donor_raw = str(donor.seq).replace("-", "")
    if len(donor_raw) < 3:
        pytest.skip("donor sequence too short for a middle deletion")

    mid = len(donor_raw) // 2
    new_seq = donor_raw[:mid] + donor_raw[mid + 1:]
    assert len(new_seq) == len(donor_raw) - 1

    new_id = f"oos_deletion_{aligner}"
    in_fasta = tmp_path / f"{new_id}.fasta"
    in_fasta.write_text(f">{new_id}\n{new_seq}\n")

    result = project_sequences(
        str(in_fasta),
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
        aligner=aligner,
    )
    [proj] = result.projections
    assert not proj.in_sample
    _assert_raw_matches_aligned_gapless(proj)


def test_in_sample_invariant():
    """In-sample short-circuit must already satisfy the invariant.
    Documented here as a control against future regressions."""
    prep_dir = f"{TMPDIR}/project_invariant_in_sample_prep"
    sca_dir = f"{TMPDIR}/project_invariant_in_sample_sca"
    for d in (prep_dir, sca_dir):
        if os.path.isdir(d):
            remove_dir(d)

    prep_args = prep_parse_args(_read_argstring(PREP_ARGS))
    prep_args.msa_fpath = f"{DATDIR}/{prep_args.msa_fpath}"
    prep_args.outdir = prep_dir
    prep_args.verbosity = 0
    prep_main(prep_args)

    sca_args = sca_parse_args(_read_argstring(SCA_ARGS))
    sca_args.indir = prep_dir
    sca_args.outdir = sca_dir
    sca_args.background = f"{DATDIR}/{sca_args.background}"
    sca_args.verbosity = 0
    sca_args.n_boot = 2
    sca_args.seed = 42
    sca_args.kstar = 3
    sca_args.n_components = 3
    sca_args.sectors_for = "all"
    sca_main(sca_args)

    prep = PreprocessingResults.load(prep_dir)
    seq_id = prep.retained_sequence_ids[0]
    donor = next(r for r in prep.msa_obj_orig if r.id == seq_id)
    donor_raw = str(donor.seq).replace("-", "")

    in_fasta = f"{TMPDIR}/_invariant_in_sample.fasta"
    with open(in_fasta, "w") as f:
        f.write(f">{seq_id}\n{donor_raw}\n")

    result = project_sequences(
        in_fasta,
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
    )
    [proj] = result.projections
    assert proj.in_sample
    _assert_raw_matches_aligned_gapless(proj)

    remove_dir(prep_dir)
    remove_dir(sca_dir)
    os.remove(in_fasta)
