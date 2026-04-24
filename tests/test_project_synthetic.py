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
"""

import json
import os
import shutil

import numpy as np
import pytest

from tests.conftest import DATDIR
from tests.test_structure import _write_minimal_pdb  # fixture helper

from mysca.project import project_sequences
from mysca.project.projection import _residue_indices_for_aligned
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
        assert proj.raw_sequence == proj.aligned_sequence.replace("-", ""), (
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
