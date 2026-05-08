"""Tests for the mysca.project subpackage and sca-project CLI.

Library tests exercise:
- In-sample roundtrip: project each training sequence and verify its
  per-IC residue set matches what sca-core already persisted in
  ``ic_residues_per_seq.npz``.
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
from mysca.helpers import get_rawseq_indices_of_msa
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


def test_project_in_sample_matches_ic_residues_per_seq(prep_and_sca_dirs):
    """Projecting training sequences should reproduce
    ic_residues_per_seq.npz entry-for-entry for the top-kstar ICs.
    """
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    sca = SCAResults.load(sca_dir)
    retained_ids = list(prep.retained_sequence_ids)

    # Write the retained sequences out as a FASTA for sca-project's
    # input (in-sample short-circuit applies to every record).
    msa_obj = prep.msa_obj_loaded
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

    # Roundtrip check against ic_residues_per_seq for the top-kstar ICs
    # (only those are expanded per sequence under the kstar scoping).
    kstar = sca.kstar
    per_seq = sca.ic_residues_per_seq
    for proj in result.projections:
        for ic in range(kstar):
            key = f"ic_{ic}_{proj.seq_id}"
            if key not in per_seq:
                continue
            expected = np.asarray(per_seq[key], dtype=int)
            got = np.asarray(proj.ic_residues[ic], dtype=int)
            assert np.array_equal(np.sort(got), np.sort(expected)), (
                f"Mismatch for {proj.seq_id} IC {ic}: "
                f"project={got.tolist()} vs persisted={expected.tolist()}"
            )


def test_project_in_sample_up_score_matches_direct_projection(
        prep_and_sca_dirs,
):
    """ProjectionResult.up_scores for in-sample sequences must match the
    rows of SCAResults.project_sequences(prep.msa_binary3d), since the
    one-hot tensor is the same up to row-permutation by retained_sequences.
    """
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    sca = SCAResults.load(sca_dir)

    msa_obj = prep.msa_obj_loaded
    ids_to_row = {rec.id: i for i, rec in enumerate(msa_obj)}
    retained_ids = list(prep.retained_sequence_ids)
    in_fasta = os.path.join(sca_dir, "_retained_input_up.fasta")
    with open(in_fasta, "w") as f:
        for sid in retained_ids:
            rec = msa_obj[ids_to_row[sid]]
            f.write(f">{sid}\n{str(rec.seq).replace('-', '')}\n")

    result = project_sequences(
        in_fasta,
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
        aligner="mafft_add",
    )
    assert result.up_scores is not None
    assert result.up_scores.shape == (
        len(retained_ids), sca.w_ica.shape[0],
    )

    # Direct projection from the persisted msa_binary3d (the canonical
    # in-sample one-hot, ordered by retained_sequences).
    expected_all = sca.project_sequences(prep.msa_binary3d)
    retained_seqs = list(prep.retained_sequences)
    for i, sid in enumerate(retained_ids):
        proj = result.by_id(sid)
        # In-sample input order matches retained_sequence_ids order;
        # the i-th projection should match the i-th row of expected_all.
        np.testing.assert_allclose(
            proj.up_score, expected_all[i], rtol=1e-10, atol=1e-12,
        )


def test_projection_to_dataframe(prep_and_sca_dirs):
    """ProjectionResult.to_dataframe yields one row per projection with
    seq_id / aligned_sequence / raw_sequence / in_sample plus up_* cols.
    """
    pd = pytest.importorskip("pandas")
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    sca = SCAResults.load(sca_dir)

    msa_obj = prep.msa_obj_loaded
    ids_to_row = {rec.id: i for i, rec in enumerate(msa_obj)}
    retained_ids = list(prep.retained_sequence_ids)[:5]
    in_fasta = os.path.join(sca_dir, "_retained_input_df.fasta")
    with open(in_fasta, "w") as f:
        for sid in retained_ids:
            rec = msa_obj[ids_to_row[sid]]
            f.write(f">{sid}\n{str(rec.seq).replace('-', '')}\n")

    result = project_sequences(
        in_fasta,
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
        aligner="mafft_add",
    )
    df = result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(retained_ids)
    expected_cols = {"seq_id", "aligned_sequence", "raw_sequence",
                     "in_sample"}
    assert expected_cols.issubset(df.columns)
    n_comp = sca.w_ica.shape[0]
    for k in range(n_comp):
        assert f"Up_{k}" in df.columns
        assert f"gap_frac_ic_{k}" in df.columns
        assert f"n_inform_ic_{k}" in df.columns
    assert df["seq_id"].tolist() == retained_ids
    # In-sample sequences with no gaps relative to themselves should
    # have gap_frac_ic_*=0 across every IC.
    for k in range(n_comp):
        assert (df[f"gap_frac_ic_{k}"] == 0.0).all(), (
            f"In-sample retained sequences expected to have full IC {k} "
            f"coverage; got {df[f'gap_frac_ic_{k}'].tolist()}"
        )


# ---------------------------------------------------------------------- #
# ic_residues_per_seq coordinate-space contract tests.                   #
#                                                                        #
# Values are raw-sequence target-residue indices, NOT MSA-col indices.   #
# These tests pin that contract so any future refactor that confuses     #
# the two will fail loudly.                                              #
# ---------------------------------------------------------------------- #


def test_ic_residues_per_seq_values_are_target_residue_indices(
        prep_and_sca_dirs,
):
    """Bounds check: every value in ic_residues_per_seq is a valid
    0-based index into the target's raw (gap-free) sequence.

    If values were MSA col indices they'd be bounded by L_orig (the
    alignment length), which is generally larger than any individual
    target's raw-sequence length. This test catches any future drift
    where MSA-col indices accidentally land in the per-seq dict.
    """
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    sca = SCAResults.load(sca_dir)
    if not sca.ic_residues_per_seq:
        pytest.skip("ic_residues_per_seq not populated by fixture")

    raw_lengths = {
        rec.id: len(str(rec.seq).replace("-", "").replace(".", ""))
        for rec in prep.msa_obj_loaded
    }
    L_orig = prep.msa_obj_loaded.get_alignment_length()

    for key, arr in sca.ic_residues_per_seq.items():
        # Key format: "ic_{j}_{seqid}". seqid may contain underscores
        # (e.g. Pfam-style "VAV_HUMAN/788-834"); split only the leading 2.
        prefix, _j_str, seqid = key.split("_", 2)
        assert prefix == "ic", f"unexpected key format: {key}"
        L_target = raw_lengths[seqid]
        arr_np = np.asarray(arr, dtype=int)
        if arr_np.size == 0:
            continue
        assert (arr_np >= 0).all(), (
            f"{key}: contains negative indices "
            f"{arr_np[arr_np < 0].tolist()}"
        )
        assert (arr_np < L_target).all(), (
            f"{key}: max index {int(arr_np.max())} >= raw-seq length "
            f"{L_target} for {seqid!r}. If values were MSA col indices "
            f"they'd be bounded by L_orig={L_orig}; the contract is "
            f"raw-sequence target-residue indices."
        )


def test_ic_residues_per_seq_values_match_raw_seq_lookup_from_ic_positions(
        prep_and_sca_dirs,
):
    """Semantic contract: ic_residues_per_seq[ic_{j}_{seqid}] equals
    the raw-sequence residue indices recovered by looking up
    ic_positions[j] (processed-MSA cols) → original-MSA cols (via
    retained_positions) → the target sequence's raw-residue indices
    (via the MSA row, with target-gap cols dropped).

    This is the canonical reconstruction. It enforces that the file's
    contents are in *target raw-sequence* coordinates and pins the data
    flow against any future refactor that swaps in MSA-col indices.
    """
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    sca = SCAResults.load(sca_dir)
    if not sca.ic_residues_per_seq or not sca.ic_positions:
        pytest.skip("ic_residues_per_seq or ic_positions not populated")

    rawseq_idxs = get_rawseq_indices_of_msa(prep.msa_obj_loaded)
    seq_msa_idx_by_id = {
        rec.id: i for i, rec in enumerate(prep.msa_obj_loaded)
    }
    retained_positions = np.asarray(prep.retained_positions, dtype=int)

    for key, arr in sca.ic_residues_per_seq.items():
        prefix, j_str, seqid = key.split("_", 2)
        assert prefix == "ic"
        j = int(j_str)
        seq_msa_idx = seq_msa_idx_by_id[seqid]
        original_msa_cols = retained_positions[
            np.asarray(sca.ic_positions[j], dtype=int)
        ]
        expected = rawseq_idxs[seq_msa_idx, original_msa_cols]
        expected = expected[expected >= 0]  # drop target gaps
        got = np.asarray(arr, dtype=int)
        np.testing.assert_array_equal(
            np.sort(got), np.sort(expected),
            err_msg=(
                f"{key}: values do not match the raw-seq lookup of "
                f"ic_positions[{j}]. The on-disk contract has drifted."
            ),
        )


def test_project_in_sample_does_not_invoke_aligner(prep_and_sca_dirs, tmp_path):
    """When every input ID is already in msa_obj_loaded, no aligner
    callable should be invoked (even if one would crash)."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    msa_obj = prep.msa_obj_loaded

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
    per-IC residues must match ic_residues_per_seq for the donor."""
    prep = PreprocessingResults.load(prep_dir)
    sca = SCAResults.load(sca_dir)
    msa_obj = prep.msa_obj_loaded

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
    per_seq = sca.ic_residues_per_seq
    for ic in range(kstar):
        key = f"ic_{ic}_{donor_id}"
        if key not in per_seq:
            continue
        expected = np.sort(np.asarray(per_seq[key], dtype=int))
        got = np.sort(np.asarray(proj.ic_residues[ic], dtype=int))
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
        ic_residues=[np.array([0])],
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
    msa_obj = prep.msa_obj_loaded

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
        # Per-IC quality metrics ride through projection.json.
        assert "gap_fraction_per_ic" in p
        assert "informative_positions_per_ic" in p
        assert isinstance(p["gap_fraction_per_ic"], list)
        assert isinstance(p["informative_positions_per_ic"], list)
        assert len(p["gap_fraction_per_ic"]) == data["n_components"]
        assert len(p["informative_positions_per_ic"]) == data["n_components"]
        # In-sample retained sequences land at gap_frac=0 across all ICs.
        assert all(v == 0.0 for v in p["gap_fraction_per_ic"])


def test_sca_project_cli_sanitizes_seq_ids_with_slashes(
        prep_and_sca_dirs, tmp_path,
):
    """FASTA IDs with filesystem-unsafe characters (Pfam-style
    ``NAME/start-end``, JGI's ``|``-separated fields) must not create
    spurious subdirectories when the per-sequence TSV is written."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    rec = prep.msa_obj_loaded[0]
    raw = str(rec.seq).replace("-", "")
    donor_id = rec.id  # real ID so the in-sample path is taken
    # Alias the same sequence under a Pfam-style ID, then project both.
    pfam_like_id = "FAKE_HUMAN/42-100"
    assert "/" in pfam_like_id

    in_fasta = tmp_path / "slashy.fasta"
    in_fasta.write_text(
        f">{donor_id}\n{raw}\n"
        f">{pfam_like_id}\n{raw}\n"
    )
    out_dir = str(tmp_path / "slashy_out")
    args = project_parse_args([
        "-i", str(in_fasta),
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out_dir,
        "-v", "0",
    ])
    project_main(args)

    per_seq_dir = os.path.join(out_dir, "per_sequence")
    assert os.path.isdir(per_seq_dir)
    # Two sibling .tsv files — no spurious "FAKE_HUMAN" subdirectory.
    entries = sorted(os.listdir(per_seq_dir))
    tsvs = [f for f in entries if f.endswith(".tsv")]
    subdirs = [
        f for f in entries
        if os.path.isdir(os.path.join(per_seq_dir, f))
    ]
    assert len(tsvs) == 2, f"Expected 2 TSVs, got {entries}"
    assert subdirs == [], f"No subdirs expected, got {subdirs}"
    # The slash was sanitized to underscore.
    assert any("FAKE_HUMAN_42-100" in f for f in tsvs), tsvs


def test_sca_project_cli_from_msa_matches_input_fpath(
        prep_and_sca_dirs, tmp_path,
):
    """--from_msa MSA --seq_id ID is equivalent to extracting the
    record into a one-record FASTA and passing it via -i."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    msa_path = os.path.join(prep_dir, "msa_orig.fasta-aln")
    assert os.path.isfile(msa_path)
    target_id = prep.msa_obj_loaded[0].id
    raw = str(prep.msa_obj_loaded[0].seq).replace("-", "")

    out_fpath = str(tmp_path / "out_input_fpath")
    in_fasta = tmp_path / "single.fasta"
    in_fasta.write_text(f">{target_id}\n{raw}\n")
    project_main(project_parse_args([
        "-i", str(in_fasta),
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out_fpath,
        "-v", "0",
    ]))

    out_from_msa = str(tmp_path / "out_from_msa")
    project_main(project_parse_args([
        "--from_msa", msa_path,
        "--seq_id", target_id,
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out_from_msa,
        "-v", "0",
    ]))

    with open(os.path.join(out_fpath, "projection.json")) as f:
        a = json.load(f)
    with open(os.path.join(out_from_msa, "projection.json")) as f:
        b = json.load(f)
    assert len(a["projections"]) == 1
    assert len(b["projections"]) == 1
    pa, pb = a["projections"][0], b["projections"][0]
    assert pa["seq_id"] == pb["seq_id"] == target_id
    assert pa["raw_sequence"] == pb["raw_sequence"] == raw
    assert pa["in_sample"] is True
    assert pb["in_sample"] is True
    assert pa["ic_residues"] == pb["ic_residues"]


def test_sca_project_cli_from_msa_unknown_id_errors(
        prep_and_sca_dirs, tmp_path,
):
    prep_dir, sca_dir = prep_and_sca_dirs
    msa_path = os.path.join(prep_dir, "msa_orig.fasta-aln")
    args = project_parse_args([
        "--from_msa", msa_path,
        "--seq_id", "definitely_not_a_real_id_xyz",
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", str(tmp_path / "out"),
        "-v", "0",
    ])
    with pytest.raises((KeyError, ValueError)):
        project_main(args)


def test_sca_project_cli_input_args_mutually_exclusive(
        prep_and_sca_dirs, tmp_path,
):
    prep_dir, sca_dir = prep_and_sca_dirs
    msa_path = os.path.join(prep_dir, "msa_orig.fasta-aln")
    in_fasta = tmp_path / "x.fasta"
    in_fasta.write_text(">x\nACDE\n")
    with pytest.raises(SystemExit):
        project_parse_args([
            "-i", str(in_fasta),
            "--from_msa", msa_path,
            "--seq_id", "x",
            "--preprocessing", prep_dir,
            "--scacore", sca_dir,
            "-o", str(tmp_path / "out"),
        ])
    with pytest.raises(SystemExit):
        project_parse_args([
            "--preprocessing", prep_dir,
            "--scacore", sca_dir,
            "-o", str(tmp_path / "out"),
        ])
    with pytest.raises(SystemExit):
        project_parse_args([
            "--from_msa", msa_path,
            "--preprocessing", prep_dir,
            "--scacore", sca_dir,
            "-o", str(tmp_path / "out"),
        ])


# ----------------------------------------------------------------------
# --raw input path: literal AA string instead of a path / FASTA record.
# ----------------------------------------------------------------------


# An AA_STD20-only string. The prep fixture trains on a synthetic
# alphabet (msa06 uses ABCD), so the projection won't be biologically
# meaningful — but both --raw and -i FASTA paths funnel through the
# same projector, so identical outputs are still the right invariant.
_RAW_AA_STD20 = "ACDEF"


@needs_mafft
def test_sca_project_cli_raw_matches_input_fpath(
        prep_and_sca_dirs, tmp_path,
):
    """`-i <SEQ> --raw` projects the literal sequence and lands the
    same projection.json as feeding the same sequence via a one-record
    FASTA at -i. Also confirms the materialized raw_input.fasta is
    written and the default seq_id is 'raw_input'."""
    prep_dir, sca_dir = prep_and_sca_dirs
    raw = _RAW_AA_STD20

    out_fpath = str(tmp_path / "out_input_fpath")
    in_fasta = tmp_path / "single.fasta"
    in_fasta.write_text(f">raw_input\n{raw}\n")
    project_main(project_parse_args([
        "-i", str(in_fasta),
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out_fpath,
        "-v", "0",
    ]))

    out_raw = str(tmp_path / "out_raw")
    project_main(project_parse_args([
        "-i", raw, "--raw",
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out_raw,
        "-v", "0",
    ]))

    assert os.path.isfile(os.path.join(out_raw, "raw_input.fasta")), (
        "expected raw_input.fasta to be materialized inside outdir"
    )

    with open(os.path.join(out_fpath, "projection.json")) as f:
        a = json.load(f)
    with open(os.path.join(out_raw, "projection.json")) as f:
        b = json.load(f)
    assert len(a["projections"]) == len(b["projections"]) == 1
    pa, pb = a["projections"][0], b["projections"][0]
    assert pb["seq_id"] == "raw_input", (
        "default seq_id under --raw should be 'raw_input'"
    )
    # Equivalence is the invariant: both paths must produce the same
    # projection. We don't assert on the specific content (the MSA's
    # synthetic alphabet means the projector may transform the input).
    assert pa["raw_sequence"] == pb["raw_sequence"]
    assert pa["aligned_sequence"] == pb["aligned_sequence"]
    assert pa["ic_residues"] == pb["ic_residues"]
    assert pa["ic_loadings"] == pb["ic_loadings"]
    assert pa["in_sample"] == pb["in_sample"]


@needs_mafft
def test_sca_project_cli_raw_custom_seq_id(prep_and_sca_dirs, tmp_path):
    """`--raw --seq_id ID` overrides the default record ID."""
    prep_dir, sca_dir = prep_and_sca_dirs

    out = str(tmp_path / "out_raw_custom_id")
    project_main(project_parse_args([
        "-i", _RAW_AA_STD20, "--raw",
        "--seq_id", "myseq",
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out,
        "-v", "0",
    ]))
    with open(os.path.join(out, "projection.json")) as f:
        payload = json.load(f)
    assert payload["projections"][0]["seq_id"] == "myseq"


@needs_mafft
def test_sca_project_cli_raw_normalizes_whitespace_and_case(
        prep_and_sca_dirs, tmp_path,
):
    """Whitespace is stripped and lowercase is uppercased before
    validation; the materialized record matches the canonical form."""
    prep_dir, sca_dir = prep_and_sca_dirs
    raw = _RAW_AA_STD20
    half = len(raw) // 2
    noisy = f"  {raw[:half]} \n{raw[half:].lower()} "

    out = str(tmp_path / "out_raw_noisy")
    project_main(project_parse_args([
        "-i", noisy, "--raw",
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out,
        "-v", "0",
    ]))
    written = open(os.path.join(out, "raw_input.fasta")).read().splitlines()
    assert written[0] == ">raw_input"
    assert written[1] == raw


def test_sca_project_cli_raw_accepts_non_canonical_chars(
        prep_and_sca_dirs, tmp_path,
):
    """Non-canonical characters are forwarded to the projector
    untouched (the materialization step does no alphabet check)."""
    seq = "ABCDD"  # 'B' is outside AA_STD20 — must pass through
    out = str(tmp_path / "out_raw_noncanonical")
    prep_dir, sca_dir = prep_and_sca_dirs
    project_parse_args([
        "-i", seq, "--raw",
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out,
        "-v", "0",
    ])
    # We can't easily run the full projection without the right
    # alphabet alignment, so call the materialization helper directly.
    from mysca.run_project import _materialize_raw_input
    os.makedirs(out, exist_ok=True)
    fasta_path = _materialize_raw_input(seq, "raw_input", out)
    assert open(fasta_path).read().splitlines() == [">raw_input", seq]


def test_sca_project_cli_raw_rejects_all_gap(prep_and_sca_dirs, tmp_path):
    """An all-gap input raises (nothing to project). Use the long
    flag form (`--input_fpath=----`) since `-i ----` would trip
    argparse's flag detection on the leading dashes."""
    prep_dir, sca_dir = prep_and_sca_dirs
    args = project_parse_args([
        "--input_fpath=----", "--raw",
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", str(tmp_path / "out_raw_gaps"),
        "-v", "0",
    ])
    with pytest.raises(ValueError, match="only gap characters"):
        project_main(args)


def test_sca_project_cli_raw_rejects_empty(prep_and_sca_dirs, tmp_path):
    """An empty / whitespace-only input raises."""
    prep_dir, sca_dir = prep_and_sca_dirs
    args = project_parse_args([
        "-i", "   \n  ", "--raw",
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", str(tmp_path / "out_raw_empty"),
        "-v", "0",
    ])
    with pytest.raises(ValueError, match="empty"):
        project_main(args)


def test_sca_project_cli_raw_mutually_exclusive_with_from_msa(
        prep_and_sca_dirs, tmp_path,
):
    """`--raw` + `--from_msa` is rejected at parse time."""
    prep_dir, sca_dir = prep_and_sca_dirs
    msa_path = os.path.join(prep_dir, "msa_orig.fasta-aln")
    with pytest.raises(SystemExit):
        project_parse_args([
            "-i", "ACDE", "--raw",
            "--from_msa", msa_path, "--seq_id", "x",
            "--preprocessing", prep_dir,
            "--scacore", sca_dir,
            "-o", str(tmp_path / "out_raw_with_from_msa"),
        ])


def test_sca_project_cli_raw_requires_input_fpath(
        prep_and_sca_dirs, tmp_path,
):
    """`--raw` without `-i` is rejected at parse time."""
    prep_dir, sca_dir = prep_and_sca_dirs
    with pytest.raises(SystemExit):
        project_parse_args([
            "--raw",
            "--preprocessing", prep_dir,
            "--scacore", sca_dir,
            "-o", str(tmp_path / "out_raw_no_input"),
        ])


# ----------------------------------------------------------------------
# --align_target {original,processed}: choose the alignment-reference MSA.
#
# Default is "original" (today's behavior; aligns against the unfiltered
# msa_orig.fasta-aln of length L_orig). "processed" aligns against the
# post-preprocessing MSA (length L_proc, sliced from msa_obj_loaded by
# retained_sequences x retained_positions).
#
# In-sample paths short-circuit to a row read with optional column slice
# (no aligner needed). Out-of-sample paths invoke the standard aligner
# against the chosen reference.
# ----------------------------------------------------------------------


def test_sca_project_align_target_default_is_original(
        prep_and_sca_dirs, tmp_path,
):
    """Default --align_target is 'original' when the flag is omitted."""
    prep_dir, sca_dir = prep_and_sca_dirs
    in_fasta = tmp_path / "x.fasta"
    in_fasta.write_text(">x\nACDE\n")
    args = project_parse_args([
        "-i", str(in_fasta),
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", str(tmp_path / "out"),
    ])
    assert args.align_target == "original"


def test_sca_project_align_target_processed_argparse_passthrough(
        prep_and_sca_dirs, tmp_path,
):
    """`--align_target processed` is accepted at parse time."""
    prep_dir, sca_dir = prep_and_sca_dirs
    in_fasta = tmp_path / "x.fasta"
    in_fasta.write_text(">x\nACDE\n")
    args = project_parse_args([
        "-i", str(in_fasta),
        "--align_target", "processed",
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", str(tmp_path / "out"),
    ])
    assert args.align_target == "processed"


def test_sca_project_align_target_rejects_unknown_value(
        prep_and_sca_dirs, tmp_path,
):
    """Argparse choices reject any value other than original/processed."""
    prep_dir, sca_dir = prep_and_sca_dirs
    in_fasta = tmp_path / "x.fasta"
    in_fasta.write_text(">x\nACDE\n")
    with pytest.raises(SystemExit):
        project_parse_args([
            "-i", str(in_fasta),
            "--align_target", "filtered",
            "--preprocessing", prep_dir,
            "--scacore", sca_dir,
            "-o", str(tmp_path / "out"),
        ])


def _project_in_sample_to_compare(prep_dir, sca_dir, target_id, outdir,
                                   align_target):
    project_main(project_parse_args([
        "--from_msa", os.path.join(prep_dir, "msa_orig.fasta-aln"),
        "--seq_id", target_id,
        "--align_target", align_target,
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", outdir,
        "-v", "0",
    ]))
    with open(os.path.join(outdir, "projection.json")) as f:
        return json.load(f)["projections"][0]


def test_sca_project_align_target_processed_in_sample_matches_sliced_row(
        prep_and_sca_dirs, tmp_path,
):
    """Under --align_target processed, an in-sample record's
    aligned_sequence equals msa_obj_loaded[idx].seq sliced by
    retained_positions, with no aligner invocation."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    target_id = prep.msa_obj_loaded[0].id
    full_aligned = str(prep.msa_obj_loaded[0].seq)
    expected = "".join(
        full_aligned[int(c)] for c in prep.retained_positions
    )

    out = str(tmp_path / "out_proc_in_sample")
    proj = _project_in_sample_to_compare(
        prep_dir, sca_dir, target_id, out, "processed",
    )
    assert proj["aligned_sequence"] == expected
    assert proj["align_target"] == "processed"
    assert proj["in_sample"] is True
    assert proj["n_input_residues_dropped"] == 0
    assert proj["input_coverage_fraction"] == 1.0


def test_sca_project_align_target_processed_aligned_sequence_length(
        prep_and_sca_dirs, tmp_path,
):
    """len(aligned_sequence) == L_proc under processed and L_orig under
    original. Driven via the in-sample path (no mafft required). The
    msa06 fixture happens to have L_orig == L_proc; that's fine — the
    test still verifies the per-mode length contract holds."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    target_id = prep.msa_obj_loaded[0].id
    L_orig = prep.msa_obj_loaded.get_alignment_length()
    L_proc = len(prep.retained_positions)

    out_orig = str(tmp_path / "out_target_orig")
    out_proc = str(tmp_path / "out_target_proc")
    a = _project_in_sample_to_compare(
        prep_dir, sca_dir, target_id, out_orig, "original",
    )
    b = _project_in_sample_to_compare(
        prep_dir, sca_dir, target_id, out_proc, "processed",
    )
    assert len(a["aligned_sequence"]) == L_orig
    assert len(b["aligned_sequence"]) == L_proc
    assert a["align_target"] == "original"
    assert b["align_target"] == "processed"


def test_sca_project_align_target_processed_ic_residues_equivalent_in_sample(
        prep_and_sca_dirs, tmp_path,
):
    """For an in-sample record, ic_residues / ic_loadings /
    ic_processed_cols are identical under either align_target — they
    are anchored to the IC structure, not to which reference the
    aligned_sequence happens to span."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    target_id = prep.msa_obj_loaded[0].id

    a = _project_in_sample_to_compare(
        prep_dir, sca_dir, target_id,
        str(tmp_path / "out_eq_orig"), "original",
    )
    b = _project_in_sample_to_compare(
        prep_dir, sca_dir, target_id,
        str(tmp_path / "out_eq_proc"), "processed",
    )
    assert a["ic_residues"] == b["ic_residues"]
    assert a["ic_loadings"] == b["ic_loadings"]
    assert a["ic_processed_cols"] == b["ic_processed_cols"]


def test_sca_project_align_target_recorded_in_projection_json(
        prep_and_sca_dirs, tmp_path,
):
    """`align_target`, `n_input_residues_dropped`, and
    `input_coverage_fraction` appear on every projection record."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    target_id = prep.msa_obj_loaded[0].id

    out = str(tmp_path / "out_fields_present")
    proj = _project_in_sample_to_compare(
        prep_dir, sca_dir, target_id, out, "original",
    )
    for field in ("align_target", "n_input_residues_dropped",
                  "input_coverage_fraction"):
        assert field in proj, f"missing {field} in projection.json"
    assert proj["align_target"] == "original"


def test_sca_project_align_target_filtered_in_sample_record(
        prep_and_sca_dirs, tmp_path,
):
    """A record present in msa_obj_loaded but absent from
    retained_sequence_ids (filtered out at preprocessing) still
    projects cleanly under --align_target processed via the slice
    path — no aligner invocation, no error."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    retained_ids = set(prep.retained_sequence_ids.tolist())
    filtered_record = next(
        (rec for rec in prep.msa_obj_loaded if rec.id not in retained_ids),
        None,
    )
    if filtered_record is None:
        pytest.skip(
            "fixture has no filtered-out records — every loaded ID "
            "survived preprocessing"
        )

    out = str(tmp_path / "out_filtered_in_sample")
    proj = _project_in_sample_to_compare(
        prep_dir, sca_dir, filtered_record.id, out, "processed",
    )
    # Came from the loaded MSA, so seq_id is in_sample by today's
    # predicate even though preprocessing dropped it.
    assert proj["in_sample"] is True
    assert proj["align_target"] == "processed"
    assert len(proj["aligned_sequence"]) == len(prep.retained_positions)


def test_sca_project_align_target_writes_processed_reference_fasta(
        prep_and_sca_dirs, tmp_path,
):
    """When --align_target processed AND at least one record needs
    out-of-sample alignment, processed_reference.fasta-aln is
    materialized inside _align_workdir/."""
    pytest.importorskip("Bio")
    import shutil
    if shutil.which("mafft") is None:
        pytest.skip("mafft not on PATH; can't trigger out-of-sample path")
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    L_proc = len(prep.retained_positions)
    # An out-of-sample query (id absent from loaded MSA).
    in_fasta = tmp_path / "oos.fasta"
    # Pick chars from the fixture's actual alphabet so the projector
    # has something meaningful to do.
    in_fasta.write_text(">oos_query\nABCD\n")
    out = str(tmp_path / "out_oos_proc_ref")
    project_main(project_parse_args([
        "-i", str(in_fasta),
        "--align_target", "processed",
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out,
        "-v", "0",
    ]))
    proc_ref = os.path.join(out, "_align_workdir", "processed_reference.fasta-aln")
    assert os.path.isfile(proc_ref), (
        f"expected processed_reference.fasta-aln at {proc_ref}"
    )
    # File should have one record per retained sequence, each L_proc long.
    from Bio import SeqIO
    records = list(SeqIO.parse(proc_ref, "fasta"))
    assert len(records) == len(prep.retained_sequence_ids)
    for rec in records:
        assert len(rec.seq) == L_proc


def test_sca_project_coverage_fields_default_full_for_in_sample(
        prep_and_sca_dirs, tmp_path,
):
    """For an in-sample record under either mode, the input came from
    the loaded MSA itself, so input_coverage_fraction is 1.0 and
    n_input_residues_dropped is 0."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    target_id = prep.msa_obj_loaded[0].id

    for mode in ("original", "processed"):
        out = str(tmp_path / f"out_cov_{mode}")
        proj = _project_in_sample_to_compare(
            prep_dir, sca_dir, target_id, out, mode,
        )
        assert proj["input_coverage_fraction"] == 1.0
        assert proj["n_input_residues_dropped"] == 0


def test_sca_project_coverage_warning_when_input_exceeds_reference(
        prep_and_sca_dirs, tmp_path, caplog,
):
    """An out-of-sample input longer than the reference has its excess
    residues clipped, producing nonzero n_input_residues_dropped, a
    coverage_fraction < 1.0, and a logged WARNING.

    Calls ``project_sequences`` directly (rather than going through
    ``project_main``) so the entrypoint's ``configure_logging`` call —
    which sets ``mysca.propagate = False`` — doesn't suppress caplog.
    """
    import logging as _logging
    import shutil
    if shutil.which("mafft") is None:
        pytest.skip("mafft not on PATH")
    from mysca.project import project_sequences
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    L_proc = len(prep.retained_positions)
    long_seq = "ABCD" * (L_proc + 4)
    in_fasta = tmp_path / "long.fasta"
    in_fasta.write_text(f">oos_long\n{long_seq}\n")

    mysca_logger = _logging.getLogger("mysca")
    prev_propagate = mysca_logger.propagate
    mysca_logger.propagate = True
    try:
        with caplog.at_level("WARNING", logger="mysca.project.projection"):
            result = project_sequences(
                str(in_fasta),
                sca_result_dir=sca_dir,
                preproc_result_dir=prep_dir,
                aligner="mafft_add",
                align_target="processed",
                workdir=str(tmp_path / "wd"),
            )
    finally:
        mysca_logger.propagate = prev_propagate

    proj = result.projections[0]
    assert proj.n_input_residues_dropped > 0
    assert proj.input_coverage_fraction < 1.0
    assert any(
        "Low alignment coverage" in r.getMessage()
        and "oos_long" in r.getMessage()
        for r in caplog.records
    ), (
        f"expected a 'Low alignment coverage' WARNING for oos_long; "
        f"got {[(r.name, r.levelname, r.getMessage()) for r in caplog.records]}"
    )


# ----------------------------------------------------------------------
# Raw-sequence / aligned-sequence consistency invariant.
#
# For ic_residues[i] (raw residue indices) to dereference correctly
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
    donor = next(r for r in prep.msa_obj_loaded if r.id == donor_id)
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
    donor = next(r for r in prep.msa_obj_loaded if r.id == donor_id)
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
    donor = next(r for r in prep.msa_obj_loaded if r.id == seq_id)
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


# --------------------------------------------------------------------------- #
# --seq_metadata sidecar TSV in sca-project: persistence + merge into          #
# seq_projections.tsv via left-join on seq_id; missing-column error parity.   #
# --------------------------------------------------------------------------- #


def test_sca_project_seq_metadata_round_trip(prep_and_sca_dirs, tmp_path):
    """`sca-project --seq_metadata` should: (1) persist the TSV verbatim
    to <outdir>/sequence_metadata.tsv and (2) merge non-`seq_id`
    columns into seq_projections.tsv when --save_dataframe is set."""
    pd = pytest.importorskip("pandas")
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    retained = list(prep.retained_sequence_ids)[:3]

    # Build a tiny FASTA from three retained training sequences (in-sample).
    in_fasta = tmp_path / "in.fasta"
    lines = []
    msa_obj = prep.msa_obj_loaded
    by_id = {rec.id: rec for rec in msa_obj}
    for sid in retained:
        rec = by_id[sid]
        lines.append(f">{sid}\n{str(rec.seq).replace('-', '')}")
    in_fasta.write_text("\n".join(lines) + "\n")

    # Metadata covers only 2 of the 3 retained sequences. The third
    # should land with NaN in the merged TSV.
    md_path = tmp_path / "metadata.tsv"
    md_path.write_text(
        "seq_id\tkingdom\ttaxid\n"
        f"{retained[0]}\tEukaryota\t9606\n"
        f"{retained[1]}\tBacteria\t562\n"
    )

    out_dir = str(tmp_path / "out")
    args = project_parse_args([
        "-i", str(in_fasta),
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out_dir,
        "--save_dataframe",
        "--seq_metadata", str(md_path),
        "-v", "0",
    ])
    project_main(args)

    # (1) sequence_metadata.tsv is a verbatim copy.
    persisted = pd.read_csv(
        os.path.join(out_dir, "sequence_metadata.tsv"), sep="\t",
    )
    expected_md = pd.read_csv(md_path, sep="\t")
    assert list(persisted.columns) == list(expected_md.columns)
    assert persisted.equals(expected_md)

    # (2) seq_projections.tsv carries the merged columns.
    df = pd.read_csv(
        os.path.join(out_dir, "seq_projections.tsv"), sep="\t",
    )
    assert "kingdom" in df.columns
    assert "taxid" in df.columns
    by_seq = df.set_index("seq_id")
    assert by_seq.loc[retained[0], "kingdom"] == "Eukaryota"
    assert int(by_seq.loc[retained[0], "taxid"]) == 9606
    assert by_seq.loc[retained[1], "kingdom"] == "Bacteria"
    # The third retained seq has no metadata row → NaN in merged columns.
    assert pd.isna(by_seq.loc[retained[2], "kingdom"])

    # (3) projection.json should NOT carry the metadata (TSV only).
    with open(os.path.join(out_dir, "projection.json")) as f:
        proj_json = json.load(f)
    for p in proj_json["projections"]:
        assert "kingdom" not in p
        assert "taxid" not in p


def test_sca_project_seq_metadata_missing_seq_id_column_raises(
    prep_and_sca_dirs, tmp_path,
):
    """A metadata TSV without a `seq_id` column must error out with a
    pointable ValueError rather than silently merging on garbage."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    rec = prep.msa_obj_loaded[0]
    in_fasta = tmp_path / "in.fasta"
    in_fasta.write_text(f">{rec.id}\n{str(rec.seq).replace('-', '')}\n")

    bad_md = tmp_path / "no_seq_id.tsv"
    bad_md.write_text("identifier\tkingdom\nfoo\tEukaryota\n")

    args = project_parse_args([
        "-i", str(in_fasta),
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", str(tmp_path / "out"),
        "--seq_metadata", str(bad_md),
        "-v", "0",
    ])
    with pytest.raises(ValueError, match="seq_id"):
        project_main(args)


# --------------------------------------------------------------------------- #
# sca-project projection plot output: --plot/--no-plot, --seq_proj_axes,      #
# --seq_proj_color_by behavior; missing-metadata warning path.                #
# --------------------------------------------------------------------------- #


def _build_in_sample_fasta(prep, tmp_path, n=3, name="in.fasta"):
    """Helper: write the first `n` retained-MSA sequences as ungapped
    primary FASTA records — the in-sample projection short-circuit."""
    fa = tmp_path / name
    by_id = {rec.id: rec for rec in prep.msa_obj_loaded}
    seq_ids = list(prep.retained_sequence_ids)[:n]
    fa.write_text("\n".join(
        f">{sid}\n{str(by_id[sid].seq).replace('-', '')}"
        for sid in seq_ids
    ) + "\n")
    return str(fa), seq_ids


def test_sca_project_emits_seq_proj_png_by_default(
    prep_and_sca_dirs, tmp_path,
):
    """A bare `sca-project` run produces images/seq_proj_ic0v1.png."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    in_fasta, _ = _build_in_sample_fasta(prep, tmp_path)
    out_dir = str(tmp_path / "out")
    project_main(project_parse_args([
        "-i", in_fasta,
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out_dir,
        "-v", "0",
    ]))
    png = os.path.join(out_dir, "images", "seq_proj_ic0v1.png")
    assert os.path.isfile(png)


def test_sca_project_no_plot_skips_images(prep_and_sca_dirs, tmp_path):
    """`--no-plot` suppresses the entire images/ directory."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    in_fasta, _ = _build_in_sample_fasta(prep, tmp_path)
    out_dir = str(tmp_path / "out")
    project_main(project_parse_args([
        "-i", in_fasta,
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out_dir,
        "--no-plot",
        "-v", "0",
    ]))
    img_dir = os.path.join(out_dir, "images")
    assert (
        not os.path.isdir(img_dir)
        or len(os.listdir(img_dir)) == 0
    )


def test_sca_project_seq_proj_axes_renders_all_in_range_pairs(
    prep_and_sca_dirs, tmp_path,
):
    """Every in-range axis pair gets a PNG; out-of-range pairs are
    silently skipped (with a warning) without aborting the others."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    sca = SCAResults.load(sca_dir)
    n_components = sca.w_ica.shape[0]
    assert n_components >= 2  # fixture invariant

    in_fasta, _ = _build_in_sample_fasta(prep, tmp_path)
    out_dir = str(tmp_path / "out")
    # Mix in-range, in-range, and a definitely-out-of-range pair.
    project_main(project_parse_args([
        "-i", in_fasta,
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out_dir,
        "--seq_proj_axes", "0,1", "0,2",
        f"{n_components},{n_components + 1}",
        "-v", "0",
    ]))
    img_dir = os.path.join(out_dir, "images")
    assert os.path.isfile(os.path.join(img_dir, "seq_proj_ic0v1.png"))
    if n_components >= 3:
        assert os.path.isfile(
            os.path.join(img_dir, "seq_proj_ic0v2.png"),
        )
    # Out-of-range pair must NOT produce a file.
    out_of_range_png = os.path.join(
        img_dir, f"seq_proj_ic{n_components}v{n_components + 1}.png",
    )
    assert not os.path.isfile(out_of_range_png)


def test_sca_project_seq_proj_color_by_emits_named_png(
    prep_and_sca_dirs, tmp_path,
):
    """--seq_proj_color_by produces an *_by_<col>.png filename."""
    pytest.importorskip("pandas")
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    in_fasta, seq_ids = _build_in_sample_fasta(prep, tmp_path)

    md = tmp_path / "md.tsv"
    md.write_text(
        "seq_id\tkingdom\n"
        + "".join(
            f"{sid}\t{'Eukaryota' if i % 2 == 0 else 'Bacteria'}\n"
            for i, sid in enumerate(seq_ids)
        )
    )

    out_dir = str(tmp_path / "out")
    project_main(project_parse_args([
        "-i", in_fasta,
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out_dir,
        "--seq_metadata", str(md),
        "--seq_proj_color_by", "kingdom",
        "-v", "0",
    ]))
    assert os.path.isfile(
        os.path.join(out_dir, "images", "seq_proj_ic0v1_by_kingdom.png"),
    )


def test_sca_project_seq_proj_color_by_warns_without_metadata(
    prep_and_sca_dirs, tmp_path,
):
    """--seq_proj_color_by without --seq_metadata: warn and still emit
    the uncolored plot."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    in_fasta, _ = _build_in_sample_fasta(prep, tmp_path)
    out_dir = str(tmp_path / "out")
    project_main(project_parse_args([
        "-i", in_fasta,
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out_dir,
        "--seq_proj_color_by", "kingdom",
        "-v", "1",
    ]))
    # The warning is written to projection.log by configure_logging.
    with open(os.path.join(out_dir, "projection.log")) as f:
        log_contents = f.read()
    assert "--seq_metadata not supplied" in log_contents
    # Uncolored plot still produced (no _by_<col> suffix).
    assert os.path.isfile(
        os.path.join(out_dir, "images", "seq_proj_ic0v1.png"),
    )
    # And no colored variant.
    assert not os.path.isfile(
        os.path.join(out_dir, "images", "seq_proj_ic0v1_by_kingdom.png"),
    )
