"""Synthetic-fixture projection tests.

A hand-crafted MSA + query sequences + PDB residue offsets exercise
every combination of raw-input length × preprocessing column drops ×
aligner. The fixture is specified in
``tests/_data/synthetic/expected.json``; see that file and the
sibling README.md for the full index tables and design rationale.

Scope:
- preprocessing produces `retained_positions = [0, 1, 3, 4, 6, 7, 9]`
- every training sequence's in-sample projection matches the expected
  `residue_by_processed_col`
- every query's out-of-sample projection under both ``mafft_add`` and
  ``hmmalign`` matches expected `aligned_sequence`,
  `residue_by_processed_col`, and satisfies the raw/aligned invariant
- PDB structure projection respects the non-1-indexed residue
  offsets in `expected.json` for a representative training sequence
  and a query
- `project_groups_to_pdb` raises when the query-longer-than-L_orig
  PDB (length 12) is projected (post-alignment raw is length 10)
- synthetic IC groups injected into the sca_dir exercise the
  "each residue lands in the correct IC, or drops out when the
  sequence has a gap at that MSA position" guarantee.
"""

import json
import os
import shutil

import numpy as np
import pytest

from tests.conftest import DATDIR
from tests.test_structure import _write_minimal_pdb  # fixture helper

from mysca.project import project_sequences
from mysca.project.projection import _gapless, _residue_indices_for_aligned
from mysca.results import PreprocessingResults, SCAResults
from mysca.run_preprocessing import (
    parse_args as prep_parse_args,
    main as prep_main,
)
from mysca.run_sca import parse_args as sca_parse_args, main as sca_main
from mysca.structure import PDBStructure, project_pdb, project_groups_to_pdb


_MAFFT = shutil.which("mafft") is not None
_HMMER = (
    shutil.which("hmmalign") is not None
    and shutil.which("hmmbuild") is not None
)
needs_mafft = pytest.mark.skipif(not _MAFFT, reason="mafft not on PATH")
needs_hmmer = pytest.mark.skipif(
    not _HMMER, reason="hmmbuild/hmmalign not on PATH",
)


SYNTHETIC_DIR = f"{DATDIR}/synthetic"
MSA_FPATH = f"{SYNTHETIC_DIR}/synthetic_msa.fasta"
QUERIES_FPATH = f"{SYNTHETIC_DIR}/queries.fasta"
EXPECTED_FPATH = f"{SYNTHETIC_DIR}/expected.json"


@pytest.fixture(scope="module")
def expected():
    with open(EXPECTED_FPATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def prep_and_sca_dirs(tmp_path_factory, expected):
    prep_dir = str(tmp_path_factory.mktemp("synth_prep"))
    sca_dir = str(tmp_path_factory.mktemp("synth_sca"))
    pp_args = expected["preprocessing_args"]
    prep_main(prep_parse_args([
        "-i", MSA_FPATH,
        "-o", prep_dir,
        "--gap_truncation_thresh", str(pp_args["gap_truncation_thresh"]),
        "--sequence_gap_thresh", str(pp_args["sequence_gap_thresh"]),
        "--position_gap_thresh", str(pp_args["position_gap_thresh"]),
        "-v", "0",
    ]))
    sca_main(sca_parse_args([
        "-i", prep_dir,
        "-o", sca_dir,
        "--regularization", "0.03",
        "--n_boot", "2",
        "--kstar", "3",
        "--n_components", "3",
        "--sectors_for", "all",
        "--seed", "42",
        "-v", "0",
    ]))
    yield prep_dir, sca_dir


# ---------------------------------------------------------------------- #
# Preprocessing shape.                                                   #
# ---------------------------------------------------------------------- #


def test_preprocessing_shape(prep_and_sca_dirs, expected):
    prep_dir, _ = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    exp = expected["preprocessing"]

    assert prep.msa_obj_orig.get_alignment_length() == exp["L_orig"]
    assert prep.msa.shape[1] == exp["L_proc"]
    assert prep.retained_positions.tolist() == exp["retained_positions"]
    assert sorted(map(str, prep.retained_sequence_ids)) == sorted(
        exp["retained_sequence_ids"]
    )


# ---------------------------------------------------------------------- #
# Training-sequence in-sample projection.                                #
# ---------------------------------------------------------------------- #


def test_training_in_sample_mappings(prep_and_sca_dirs, expected, tmp_path):
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    training = expected["training"]

    # Write every retained sequence as its ungapped primary sequence
    # and project them in-sample (should short-circuit alignment).
    in_fasta = tmp_path / "training_in_sample.fasta"
    with open(in_fasta, "w") as f:
        for rec in prep.msa_obj_orig:
            raw = str(rec.seq).replace("-", "")
            f.write(f">{rec.id}\n{raw}\n")

    result = project_sequences(
        str(in_fasta),
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
    )
    assert all(p.in_sample for p in result.projections)

    by_id = {p.seq_id: p for p in result.projections}
    for seq_id, exp in training.items():
        assert seq_id in by_id, f"{seq_id} missing from projections"
        proj = by_id[seq_id]
        assert proj.raw_sequence == exp["raw_sequence"]
        assert proj.aligned_sequence == exp["aligned_sequence"]
        assert proj.residue_by_processed_col.tolist() == \
            exp["residue_by_processed_col"]


# ---------------------------------------------------------------------- #
# Query out-of-sample projection under each aligner.                     #
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("aligner", [
    pytest.param("mafft_add", marks=needs_mafft),
    pytest.param("hmmalign", marks=needs_hmmer),
])
def test_queries_out_of_sample_mappings(prep_and_sca_dirs, expected, aligner):
    prep_dir, sca_dir = prep_and_sca_dirs
    result = project_sequences(
        QUERIES_FPATH,
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
        aligner=aligner,
    )
    by_id = {p.seq_id: p for p in result.projections}
    for q_id, exp in expected["queries"].items():
        assert q_id in by_id, f"{q_id} missing from projections"
        proj = by_id[q_id]
        assert not proj.in_sample, (
            f"{q_id} should have taken the out-of-sample path"
        )
        assert proj.raw_sequence == exp["raw_sequence"], (
            f"{q_id} [{aligner}] raw_sequence mismatch"
        )
        assert proj.aligned_sequence == exp["aligned_sequence"], (
            f"{q_id} [{aligner}] aligned_sequence mismatch"
        )
        assert proj.residue_by_processed_col.tolist() == \
            exp["residue_by_processed_col"], (
                f"{q_id} [{aligner}] residue_by_processed_col mismatch"
            )
        # Raw/aligned invariant — same test as test_project.py but
        # exercised across every length regime here.
        assert proj.raw_sequence == _gapless(proj.aligned_sequence), (
            f"{q_id} [{aligner}] invariant violated"
        )


# ---------------------------------------------------------------------- #
# Internal consistency of residue_by_processed_col via retained_positions.
# ---------------------------------------------------------------------- #


def test_residue_by_processed_col_recomputed_from_aligned(
        prep_and_sca_dirs, expected,
):
    """Verify that residue_by_processed_col matches what you get by
    hand-indexing the aligned sequence through retained_positions.
    Catches any silent off-by-one between the two coordinate systems."""
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    retained_positions = np.asarray(prep.retained_positions, dtype=int)

    result = project_sequences(
        QUERIES_FPATH,
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
        aligner="mafft_add" if _MAFFT else "hmmalign",
    )
    for proj in result.projections:
        resi_by_orig = _residue_indices_for_aligned(proj.aligned_sequence)
        recomputed = resi_by_orig[retained_positions]
        assert np.array_equal(recomputed, proj.residue_by_processed_col)


# ---------------------------------------------------------------------- #
# Structure projection with non-1-indexed PDB residues.                  #
# ---------------------------------------------------------------------- #


def _pdb_from_fixture(fixture_id, raw_sequence, expected, tmp_path):
    """Write a minimal PDB with the residue-number range specified in
    expected.json['pdb_residue_offsets'][fixture_id]."""
    info = expected["pdb_residue_offsets"][fixture_id]
    assert info["length"] == len(raw_sequence), (
        f"PDB length {info['length']} != raw sequence length "
        f"{len(raw_sequence)} for {fixture_id}"
    )
    resi = list(range(info["start"], info["start"] + info["length"]))
    path = str(tmp_path / f"{fixture_id}.pdb")
    _write_minimal_pdb(raw_sequence, path, residue_numbers=resi)
    return path, resi


def test_structure_projection_clade_A_offset_10(
        prep_and_sca_dirs, expected, tmp_path,
):
    prep_dir, sca_dir = prep_and_sca_dirs
    seq_id = "synth_clade_A_0"
    raw = expected["training"][seq_id]["raw_sequence"]
    pdb_path, residue_numbers = _pdb_from_fixture(
        seq_id, raw, expected, tmp_path,
    )
    pdb = PDBStructure.from_file(pdb_path)
    assert pdb.sequence == raw
    assert pdb.residue_ids == residue_numbers

    proj = project_pdb(
        pdb,
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
        seq_id=seq_id,
    )
    assert proj.sequence_projection.in_sample

    # Every raw index r present in any ic_memberships must map through
    # pdb.residue_ids to the matching PDB residue number.
    for ic_idx, (raw_idxs, pdb_resids) in enumerate(zip(
        proj.sequence_projection.ic_memberships, proj.ic_pdb_residues,
    )):
        for r, pdb_resi in zip(raw_idxs.tolist(), pdb_resids):
            assert pdb_resi == residue_numbers[r], (
                f"IC {ic_idx} raw idx {r} -> PDB {pdb_resi}; "
                f"expected {residue_numbers[r]}"
            )


def test_structure_projection_clade_B_offset_50(
        prep_and_sca_dirs, expected, tmp_path,
):
    """Clade B: raw sequence is L_orig long, and raw idxs 2/5/8 are at
    columns that get dropped by preprocessing. They must not appear in
    any ic_memberships, and therefore not in ic_pdb_residues either.
    """
    prep_dir, sca_dir = prep_and_sca_dirs
    seq_id = "synth_clade_B_0"
    raw = expected["training"][seq_id]["raw_sequence"]
    pdb_path, residue_numbers = _pdb_from_fixture(
        seq_id, raw, expected, tmp_path,
    )
    pdb = PDBStructure.from_file(pdb_path)

    proj = project_pdb(
        pdb,
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
        seq_id=seq_id,
    )
    assert proj.sequence_projection.in_sample

    dropped_raw_idxs = {2, 5, 8}
    for ic_idx, raw_idxs in enumerate(proj.sequence_projection.ic_memberships):
        for r in raw_idxs.tolist():
            assert r not in dropped_raw_idxs, (
                f"IC {ic_idx} raw idx {r} is at a dropped column; "
                f"shouldn't be in ic_memberships"
            )

    for ic_idx, (raw_idxs, pdb_resids) in enumerate(zip(
        proj.sequence_projection.ic_memberships, proj.ic_pdb_residues,
    )):
        for r, pdb_resi in zip(raw_idxs.tolist(), pdb_resids):
            assert pdb_resi == residue_numbers[r]


@needs_mafft
def test_structure_projection_query_out_of_sample(
        prep_and_sca_dirs, expected, tmp_path,
):
    prep_dir, sca_dir = prep_and_sca_dirs
    q_id = "query_equal_to_orig"
    q_raw = expected["queries"][q_id]["raw_sequence"]
    pdb_path, residue_numbers = _pdb_from_fixture(
        q_id, q_raw, expected, tmp_path,
    )
    pdb = PDBStructure.from_file(pdb_path)

    proj = project_pdb(
        pdb,
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
        seq_id=q_id,
    )
    assert not proj.sequence_projection.in_sample

    for ic_idx, (raw_idxs, pdb_resids) in enumerate(zip(
        proj.sequence_projection.ic_memberships, proj.ic_pdb_residues,
    )):
        for r, pdb_resi in zip(raw_idxs.tolist(), pdb_resids):
            assert pdb_resi == residue_numbers[r]


@needs_mafft
def test_structure_projection_query_longer_than_orig_raises(
        prep_and_sca_dirs, expected, tmp_path,
):
    """A PDB built from the 12-residue input collides with the
    10-residue post-alignment raw_sequence. project_groups_to_pdb
    should refuse to map rather than silently returning wrong
    residue numbers."""
    prep_dir, sca_dir = prep_and_sca_dirs
    q_id = "query_longer_than_orig"
    input_raw = expected["queries"][q_id]["input_raw"]  # length 12
    pdb_path, _ = _pdb_from_fixture(q_id, input_raw, expected, tmp_path)
    pdb = PDBStructure.from_file(pdb_path)
    assert len(pdb.sequence) == 12

    with pytest.raises(ValueError, match="Raw-sequence length mismatch"):
        project_pdb(
            pdb,
            sca_result_dir=sca_dir,
            preproc_result_dir=prep_dir,
            seq_id=q_id,
        )


# ---------------------------------------------------------------------- #
# Gap-character handling: both `-` and `.` are recognized as gaps.       #
# ---------------------------------------------------------------------- #


def test_gap_chars_include_dot():
    """`.` is the Stockholm insert-column gap symbol. Our pipeline
    normalizes it away in practice (Biopython normalizes Stockholm `.`
    to `-` on read; _hmmalign strips `.` during insert-column
    filtering), but defend against it leaking through: both `_gapless`
    and `_residue_indices_for_aligned` must treat `.` as a gap."""
    # Hand-crafted aligned row mixing both gap conventions.
    aligned = "A.C-DE.F-G"
    assert _gapless(aligned) == "ACDEFG"
    # Residue indices: A=0 at col 0, C=1 at col 2, D=2 at col 4,
    # E=3 at col 5, F=4 at col 7, G=5 at col 9; everything else -1.
    expected = [0, -1, 1, -1, 2, 3, -1, 4, -1, 5]
    got = _residue_indices_for_aligned(aligned).tolist()
    assert got == expected, (
        f"`.` should be treated as a gap in residue indexing.\n"
        f"  aligned:  {aligned!r}\n"
        f"  expected: {expected}\n"
        f"  got:      {got}"
    )


# ---------------------------------------------------------------------- #
# Artificial IC assignment: overwrite the sca_dir's groups with hand-   #
# crafted processed-MSA positions and verify that each sequence/PDB's   #
# residues land in the intended IC (or drop out when the sequence has   #
# a gap at the IC's processed column).                                  #
# ---------------------------------------------------------------------- #


def _inject_synthetic_groups(sca_dir, groups):
    """Overwrite the group files SCAResults.load() reads with the given
    hand-crafted groups (list of lists of processed-MSA col indices).
    Also clears the paired scores files so group_scores loads as None
    (projection doesn't use scores, but mismatched lengths would be
    misleading)."""
    sector_dir = os.path.join(sca_dir, "sca_results", "msa_sectors")
    groups_dir = os.path.join(sca_dir, "groups")
    for d in (sector_dir, groups_dir):
        if os.path.isdir(d):
            for fn in os.listdir(d):
                fp = os.path.join(d, fn)
                if fn.startswith(("sector_", "group_")):
                    os.remove(fp)
    os.makedirs(sector_dir, exist_ok=True)
    os.makedirs(groups_dir, exist_ok=True)
    for i, g in enumerate(groups):
        arr = np.asarray(g, dtype=int)
        np.save(os.path.join(sector_dir, f"sector_{i}_msapos.npy"), arr)
        np.save(os.path.join(groups_dir, f"group_{i}_msapos.npy"), arr)


@pytest.fixture(scope="module")
def sca_dir_with_synthetic_groups(prep_and_sca_dirs, expected, tmp_path_factory):
    """Copy the shared sca_dir to a fresh path, overwrite its groups
    with the hand-crafted ones from expected.json."""
    _, src_sca = prep_and_sca_dirs
    dest = str(tmp_path_factory.mktemp("synth_sca_hand_groups"))
    # Copy contents of src_sca into dest (dest exists, so use copytree
    # with dirs_exist_ok).
    shutil.copytree(src_sca, dest, dirs_exist_ok=True)
    synth_groups = expected["synthetic_ic_groups"]["groups"]
    _inject_synthetic_groups(dest, synth_groups)
    return dest


def test_injected_groups_roundtrip(sca_dir_with_synthetic_groups, expected):
    """Sanity: SCAResults.load() picks up exactly the hand-crafted
    groups we wrote, in order."""
    sca = SCAResults.load(sca_dir_with_synthetic_groups)
    assert sca.groups is not None
    got = [g.tolist() for g in sca.groups]
    assert got == expected["synthetic_ic_groups"]["groups"]


def test_training_ic_memberships_against_synthetic_groups(
        prep_and_sca_dirs, sca_dir_with_synthetic_groups, expected, tmp_path,
):
    """Every training sequence, projected in-sample through the
    hand-crafted groups, should land each residue in the expected IC."""
    prep_dir, _ = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    expected_map = expected["expected_ic_memberships"]

    in_fasta = tmp_path / "training_for_ic_test.fasta"
    with open(in_fasta, "w") as f:
        for rec in prep.msa_obj_orig:
            raw = str(rec.seq).replace("-", "")
            f.write(f">{rec.id}\n{raw}\n")

    result = project_sequences(
        str(in_fasta),
        sca_result_dir=sca_dir_with_synthetic_groups,
        preproc_result_dir=prep_dir,
    )
    for proj in result.projections:
        assert proj.seq_id in expected_map
        exp = expected_map[proj.seq_id]
        got = [m.tolist() for m in proj.ic_memberships]
        assert got == exp, (
            f"{proj.seq_id} ic_memberships mismatch:\n"
            f"  expected: {exp}\n"
            f"  got:      {got}"
        )


@pytest.mark.parametrize("aligner", [
    pytest.param("mafft_add", marks=needs_mafft),
    pytest.param("hmmalign", marks=needs_hmmer),
])
def test_query_ic_memberships_against_synthetic_groups(
        prep_and_sca_dirs, sca_dir_with_synthetic_groups, expected, aligner,
):
    """Every query, projected out-of-sample, should land each residue
    in the expected IC. query_shorter_than_proc specifically produces
    an empty IC 2 because proc col 6 is a gap for that query."""
    prep_dir, _ = prep_and_sca_dirs
    expected_map = expected["expected_ic_memberships"]
    result = project_sequences(
        QUERIES_FPATH,
        sca_result_dir=sca_dir_with_synthetic_groups,
        preproc_result_dir=prep_dir,
        aligner=aligner,
    )
    for proj in result.projections:
        exp = expected_map[proj.seq_id]
        got = [m.tolist() for m in proj.ic_memberships]
        assert got == exp, (
            f"[{aligner}] {proj.seq_id} ic_memberships mismatch:\n"
            f"  expected: {exp}\n"
            f"  got:      {got}"
        )
    # Spot check the "IC 2 empty because of gap" case directly.
    short = next(
        p for p in result.projections if p.seq_id == "query_shorter_than_proc"
    )
    assert short.ic_memberships[2].size == 0, (
        f"[{aligner}] query_shorter_than_proc IC 2 should be empty "
        f"(proc col 6 is a gap for this query); got "
        f"{short.ic_memberships[2].tolist()}"
    )


def test_structure_ic_pdb_residues_against_synthetic_groups(
        prep_and_sca_dirs, sca_dir_with_synthetic_groups, expected, tmp_path,
):
    """Structure projection: for a clade B training sequence with PDB
    residues 50..59, each IC's ic_pdb_residues should be the PDB
    residue numbers corresponding to the ic_memberships raw indices."""
    prep_dir, _ = prep_and_sca_dirs
    seq_id = "synth_clade_B_0"
    raw = expected["training"][seq_id]["raw_sequence"]
    pdb_path, residue_numbers = _pdb_from_fixture(
        seq_id, raw, expected, tmp_path,
    )
    pdb = PDBStructure.from_file(pdb_path)

    proj = project_pdb(
        pdb,
        sca_result_dir=sca_dir_with_synthetic_groups,
        preproc_result_dir=prep_dir,
        seq_id=seq_id,
    )
    expected_raw = expected["expected_ic_memberships"][seq_id]
    expected_pdb = [
        [residue_numbers[r] for r in ic_members] for ic_members in expected_raw
    ]
    got_pdb = [list(xs) for xs in proj.ic_pdb_residues]
    assert got_pdb == expected_pdb, (
        f"{seq_id} ic_pdb_residues mismatch:\n"
        f"  expected: {expected_pdb}\n"
        f"  got:      {got_pdb}"
    )


@needs_mafft
def test_structure_ic_pdb_residues_query_with_gap_at_ic_position(
        prep_and_sca_dirs, sca_dir_with_synthetic_groups, expected, tmp_path,
):
    """The short query has a gap at proc col 6 (IC 2's only column),
    so its PDB has NO residue in IC 2 — ic_pdb_residues[2] must be
    empty even though ic_pdb_residues[0] and [1] have residues."""
    prep_dir, _ = prep_and_sca_dirs
    seq_id = "query_shorter_than_proc"
    raw = expected["queries"][seq_id]["raw_sequence"]
    pdb_path, residue_numbers = _pdb_from_fixture(
        seq_id, raw, expected, tmp_path,
    )
    pdb = PDBStructure.from_file(pdb_path)

    proj = project_pdb(
        pdb,
        sca_result_dir=sca_dir_with_synthetic_groups,
        preproc_result_dir=prep_dir,
        seq_id=seq_id,
    )
    exp_raw = expected["expected_ic_memberships"][seq_id]
    exp_pdb = [
        [residue_numbers[r] for r in ic_members] for ic_members in exp_raw
    ]
    got = [list(xs) for xs in proj.ic_pdb_residues]
    assert got == exp_pdb
    assert got[2] == [], (
        "query_shorter_than_proc has a gap at proc col 6; IC 2 "
        "should not contain any PDB residue."
    )
