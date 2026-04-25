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

from mysca.helpers import (
    get_rawseq_indices_of_msa,
    get_rawseq_positions_in_groups,
)
from mysca.project import project_sequences
from mysca.project.projection import _gapless, _residue_indices_for_aligned
from mysca.results import PreprocessingResults, SCAResults
from mysca.run_preprocessing import (
    parse_args as prep_parse_args,
    main as prep_main,
)
from mysca.run_sca import (
    parse_args as sca_parse_args,
    main as sca_main,
    log_top_ic_summary,
)
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

    # Every raw index r present in any ic_residues must map through
    # pdb.residue_ids to the matching PDB residue number.
    for ic_idx, (raw_idxs, pdb_resids) in enumerate(zip(
        proj.sequence_projection.ic_residues, proj.ic_pdb_residues,
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
    any ic_residues, and therefore not in ic_pdb_residues either.
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
    for ic_idx, raw_idxs in enumerate(proj.sequence_projection.ic_residues):
        for r in raw_idxs.tolist():
            assert r not in dropped_raw_idxs, (
                f"IC {ic_idx} raw idx {r} is at a dropped column; "
                f"shouldn't be in ic_residues"
            )

    for ic_idx, (raw_idxs, pdb_resids) in enumerate(zip(
        proj.sequence_projection.ic_residues, proj.ic_pdb_residues,
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
        proj.sequence_projection.ic_residues, proj.ic_pdb_residues,
    )):
        for r, pdb_resi in zip(raw_idxs.tolist(), pdb_resids):
            assert pdb_resi == residue_numbers[r]


@needs_mafft
def test_structure_projection_query_longer_than_orig_composes_through_input_indices(
        prep_and_sca_dirs, expected, tmp_path,
):
    """A PDB built from the 12-residue input no longer raises when
    alignment drops residues. input_residue_indices records which
    input positions survived, and project_groups_to_pdb composes
    them with pdb.residue_ids to produce correct PDB residue numbers.
    """
    prep_dir, sca_dir = prep_and_sca_dirs
    q_id = "query_longer_than_orig"
    input_raw = expected["queries"][q_id]["input_raw"]  # length 12
    pdb_path, residue_numbers = _pdb_from_fixture(
        q_id, input_raw, expected, tmp_path,
    )
    pdb = PDBStructure.from_file(pdb_path)
    assert len(pdb.sequence) == 12

    proj = project_pdb(
        pdb,
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
        seq_id=q_id,
    )
    # input_residue_indices should match expected.json.
    got = proj.sequence_projection.input_residue_indices
    assert got == expected["expected_input_residue_indices"][q_id], got

    # ic_pdb_residues composes ic_residues (raw indices) with
    # input_residue_indices and pdb.residue_ids. Verify element-wise.
    for ic_idx, (raw_members, pdb_resids) in enumerate(zip(
        proj.sequence_projection.ic_residues, proj.ic_pdb_residues,
    )):
        for r, pdb_resi in zip(raw_members.tolist(), pdb_resids):
            input_idx = got[r]
            assert pdb_resi == residue_numbers[input_idx], (
                f"IC {ic_idx} raw {r} -> input {input_idx} -> "
                f"pdb {pdb_resi}; expected {residue_numbers[input_idx]}"
            )


def test_project_groups_to_pdb_raises_when_pdb_too_short(tmp_path, expected):
    """Constructed failure: hand a project_groups_to_pdb a
    SequenceProjection whose input_residue_indices reach beyond
    pdb.residue_ids — e.g., a projection from a 12-residue input
    paired with a PDB that only carries 5 residues."""
    from mysca.project.projection import SequenceProjection
    pdb_path = str(tmp_path / "tiny.pdb")
    _write_minimal_pdb(
        "ACDEV", pdb_path, residue_numbers=[1, 2, 3, 4, 5],
    )
    pdb = PDBStructure.from_file(pdb_path)

    seq_proj = SequenceProjection(
        seq_id="x",
        raw_sequence="ACDE",
        aligned_sequence="ACDE",
        residue_by_processed_col=np.array([0, 1, 2, 3], dtype=int),
        ic_residues=[np.array([0, 3], dtype=int)],
        ic_loadings=[np.array([0.1, 0.2], dtype=float)],
        ic_processed_cols=[np.array([0, 3], dtype=int)],
        in_sample=False,
        # Deliberately reach past pdb.residue_ids (length 5).
        input_residue_indices=[0, 1, 5, 7],
    )
    with pytest.raises(
        ValueError, match="input_residue_indices references position",
    ):
        project_groups_to_pdb(seq_proj, pdb)


# ---------------------------------------------------------------------- #
# input_residue_indices recovery                                         #
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("aligner", [
    pytest.param("mafft_add", marks=needs_mafft),
    pytest.param("hmmalign", marks=needs_hmmer),
])
def test_query_input_residue_indices(prep_and_sca_dirs, expected, aligner):
    """For each query, input_residue_indices should record exactly
    which positions in the ORIGINAL input sequence survived the
    column-preserving alignment. In-sample / same-length cases are
    the identity; query_longer_than_orig drops positions 6 and 7
    (N and S) and the helper should recover
    [0,1,2,3,4,5,8,9,10,11]."""
    prep_dir, sca_dir = prep_and_sca_dirs
    result = project_sequences(
        QUERIES_FPATH,
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
        aligner=aligner,
    )
    exp_map = expected["expected_input_residue_indices"]
    for proj in result.projections:
        assert proj.input_residue_indices == exp_map[proj.seq_id], (
            f"{proj.seq_id} [{aligner}]: "
            f"got {proj.input_residue_indices}; "
            f"expected {exp_map[proj.seq_id]}"
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
    """Overwrite the IC-position files SCAResults.load() reads with the
    given hand-crafted groups (list of lists of processed-MSA col
    indices). Also clears the paired scores files so group_scores loads
    as None (projection doesn't use scores, but mismatched lengths
    would be misleading)."""
    sector_dir = os.path.join(sca_dir, "sca_results", "msa_sectors")
    ic_pos_dir = os.path.join(sca_dir, "ic_positions")
    for d in (sector_dir, ic_pos_dir):
        if os.path.isdir(d):
            for fn in os.listdir(d):
                fp = os.path.join(d, fn)
                if fn.startswith(("sector_", "ic_")):
                    os.remove(fp)
    os.makedirs(sector_dir, exist_ok=True)
    os.makedirs(ic_pos_dir, exist_ok=True)
    for i, g in enumerate(groups):
        arr = np.asarray(g, dtype=int)
        # Internal load source (still under msa_sectors/ in Phase A).
        np.save(os.path.join(sector_dir, f"sector_{i}_msapos.npy"), arr)
        # New top-level mirror (Phase A).
        np.save(os.path.join(ic_pos_dir, f"ic_{i}_msaproc.npy"), arr)


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


def test_synthetic_ic_residues_per_seq_construction_matches_ground_truth(
        prep_and_sca_dirs, expected,
):
    """Independent ground-truth contract test for the per-target IC
    residue projection that backs `ic_residues_per_seq.npz`.

    Replicates the exact chain `run_sca.py` uses to populate
    `ic_residues_per_seq` (raw-sequence index lookup → retained-positions
    slice → `get_rawseq_positions_in_groups`) against the hand-crafted
    synthetic groups, and asserts the output matches the
    *independently hand-specified* `expected_ic_residues` table in
    `expected.json`.

    The sibling tests `test_ic_residues_per_seq_values_are_target_residue_indices`
    and `test_ic_residues_per_seq_values_match_raw_seq_lookup_from_ic_positions`
    in test_project.py are *consistency* tests: they verify that the
    file's contents are reachable via the producer chain. This test
    closes the loop with an independent ground truth — it would catch
    off-by-one or gap-handling bugs in `get_rawseq_positions_in_groups`
    or its callers that the consistency tests would silently mirror.
    """
    prep_dir, _ = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)

    rawseq_idxs = get_rawseq_indices_of_msa(prep.msa_obj_orig)
    retained_sequences = np.asarray(prep.retained_sequences, dtype=int)
    retained_positions = np.asarray(prep.retained_positions, dtype=int)
    rawseq_idxs = rawseq_idxs[retained_sequences, :][:, retained_positions]

    synthetic_groups = expected["synthetic_ic_groups"]["groups"]
    per_seq_per_ic = get_rawseq_positions_in_groups(
        rawseq_idxs, synthetic_groups,
    )

    expected_map = expected["expected_ic_residues"]
    msa_ids_by_retained_idx = [
        prep.msa_obj_orig[int(s)].id for s in retained_sequences
    ]
    asserted_at_least_one = False
    for retained_idx, seqid in enumerate(msa_ids_by_retained_idx):
        if seqid not in expected_map:
            continue
        for ic, exp_residues in enumerate(expected_map[seqid]):
            got = sorted(per_seq_per_ic[retained_idx][ic])
            exp = sorted(exp_residues)
            assert got == exp, (
                f"Synthetic ic_residues_per_seq construction for {seqid} IC "
                f"{ic}: got {got}, expected {exp} (independent "
                f"ground truth from expected.json). The producer "
                f"chain (rawseq_idxs → retained_positions → "
                f"get_rawseq_positions_in_groups) appears to have "
                f"drifted."
            )
            asserted_at_least_one = True
    assert asserted_at_least_one, (
        "Test made zero assertions; expected_ic_residues and the "
        "retained-sequence list don't intersect (fixture broken?)."
    )


def test_injected_groups_roundtrip(sca_dir_with_synthetic_groups, expected):
    """Sanity: SCAResults.load() picks up exactly the hand-crafted
    groups we wrote, in order."""
    sca = SCAResults.load(sca_dir_with_synthetic_groups)
    assert sca.ic_positions is not None
    got = [g.tolist() for g in sca.ic_positions]
    assert got == expected["synthetic_ic_groups"]["groups"]


def test_training_ic_residues_against_synthetic_groups(
        prep_and_sca_dirs, sca_dir_with_synthetic_groups, expected, tmp_path,
):
    """Every training sequence, projected in-sample through the
    hand-crafted groups, should land each residue in the expected IC."""
    prep_dir, _ = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    expected_map = expected["expected_ic_residues"]

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
        got = [m.tolist() for m in proj.ic_residues]
        assert got == exp, (
            f"{proj.seq_id} ic_residues mismatch:\n"
            f"  expected: {exp}\n"
            f"  got:      {got}"
        )


@pytest.mark.parametrize("aligner", [
    pytest.param("mafft_add", marks=needs_mafft),
    pytest.param("hmmalign", marks=needs_hmmer),
])
def test_query_ic_residues_against_synthetic_groups(
        prep_and_sca_dirs, sca_dir_with_synthetic_groups, expected, aligner,
):
    """Every query, projected out-of-sample, should land each residue
    in the expected IC. query_shorter_than_proc specifically produces
    an empty IC 2 because proc col 6 is a gap for that query."""
    prep_dir, _ = prep_and_sca_dirs
    expected_map = expected["expected_ic_residues"]
    result = project_sequences(
        QUERIES_FPATH,
        sca_result_dir=sca_dir_with_synthetic_groups,
        preproc_result_dir=prep_dir,
        aligner=aligner,
    )
    for proj in result.projections:
        exp = expected_map[proj.seq_id]
        got = [m.tolist() for m in proj.ic_residues]
        assert got == exp, (
            f"[{aligner}] {proj.seq_id} ic_residues mismatch:\n"
            f"  expected: {exp}\n"
            f"  got:      {got}"
        )
    # Spot check the "IC 2 empty because of gap" case directly.
    short = next(
        p for p in result.projections if p.seq_id == "query_shorter_than_proc"
    )
    assert short.ic_residues[2].size == 0, (
        f"[{aligner}] query_shorter_than_proc IC 2 should be empty "
        f"(proc col 6 is a gap for this query); got "
        f"{short.ic_residues[2].tolist()}"
    )


def test_structure_ic_pdb_residues_against_synthetic_groups(
        prep_and_sca_dirs, sca_dir_with_synthetic_groups, expected, tmp_path,
):
    """Structure projection: for a clade B training sequence with PDB
    residues 50..59, each IC's ic_pdb_residues should be the PDB
    residue numbers corresponding to the ic_residues raw indices."""
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
    expected_raw = expected["expected_ic_residues"][seq_id]
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
    exp_raw = expected["expected_ic_residues"][seq_id]
    exp_pdb = [
        [residue_numbers[r] for r in ic_members] for ic_members in exp_raw
    ]
    got = [list(xs) for xs in proj.ic_pdb_residues]
    assert got == exp_pdb
    assert got[2] == [], (
        "query_shorter_than_proc has a gap at proc col 6; IC 2 "
        "should not contain any PDB residue."
    )


# ---------------------------------------------------------------------- #
# log_top_ic_summary against the synthetic fixture + hand-crafted groups.
# Asserts the log's processed → unprocessed → reference-pos → reference-
# res chain on both a clade-A reference (where the reference has gaps at
# the dropped columns — a chance for the "-" rendering to misalign) and
# a clade-B reference (no gaps in the reference, straightforward case).
# ---------------------------------------------------------------------- #


@pytest.fixture
def capture_run_sca_logs():
    import logging
    target = logging.getLogger("mysca.run_sca")
    records = []

    class _Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collector(level=logging.DEBUG)
    old_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        target.removeHandler(handler)
        target.setLevel(old_level)


def test_log_top_ic_summary_synthetic_clade_B_reference(
        prep_and_sca_dirs, expected, capture_run_sca_logs,
):
    """Clade-B reference (raw=ACTDEVFGWI, no gaps) means every
    unprocessed col dereferences directly to a residue. Use the hand-
    crafted groups (which are known to land each clade-B residue at a
    deterministic raw index) so the reference lines are fully
    predictable."""
    prep_dir, _ = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    synth_groups = [
        np.asarray(g, dtype=int)
        for g in expected["synthetic_ic_groups"]["groups"]
    ]

    log_top_ic_summary(
        synth_groups,
        kstar=2,
        evals_sca=np.array([2.5, 1.7, 0.4]),
        retained_positions=prep.retained_positions,
        msa_obj_orig=prep.msa_obj_orig,
        reference_id="synth_clade_B_0",
        n_logged_comps=10,
    )
    msgs = [r.getMessage() for r in capture_run_sca_logs]

    header = next(m for m in msgs if m.startswith("Top "))
    assert "reference=synth_clade_B_0" in header
    assert "3/3 ICs" in header

    # Expected from expected.json's index tables for synth_clade_B_0:
    #   proc [0,1,4] -> unproc [0,1,6]  -> ref_pos [0,1,6]  -> res [A,C,F]
    #   proc [2,3,5] -> unproc [3,4,7]  -> ref_pos [3,4,7]  -> res [D,E,G]
    #   proc [6]     -> unproc [9]      -> ref_pos [9]      -> res [I]
    expected_chain = [
        ("[0, 1, 4]", "[0, 1, 6]", "[0, 1, 6]", "[A, C, F]"),
        ("[2, 3, 5]", "[3, 4, 7]", "[3, 4, 7]", "[D, E, G]"),
        ("[6]",       "[9]",       "[9]",       "[I]"),
    ]
    proc_lines = [m for m in msgs if "processed:" in m and "unprocessed" not in m]
    unproc_lines = [m for m in msgs if "unprocessed:" in m]
    ref_pos_lines = [m for m in msgs if "reference pos:" in m]
    ref_res_lines = [m for m in msgs if "reference res:" in m]
    assert len(proc_lines) == len(unproc_lines) == 3
    assert len(ref_pos_lines) == len(ref_res_lines) == 3

    for ic, (p, u, rp, rr) in enumerate(expected_chain):
        assert proc_lines[ic].endswith(p), f"IC {ic} processed: {proc_lines[ic]!r}"
        assert unproc_lines[ic].endswith(u), f"IC {ic} unproc: {unproc_lines[ic]!r}"
        assert ref_pos_lines[ic].endswith(rp), f"IC {ic} ref pos: {ref_pos_lines[ic]!r}"
        assert ref_res_lines[ic].endswith(rr), f"IC {ic} ref res: {ref_res_lines[ic]!r}"


def test_log_top_ic_summary_synthetic_clade_A_reference(
        prep_and_sca_dirs, expected, capture_run_sca_logs,
):
    """Clade-A reference (aligned=AC-DE-FG-I) has gaps at the dropped
    original columns — but those columns don't appear in any
    ``unprocessed`` list (because they're not in retained_positions),
    so the reference lines here are gap-free. The interesting thing is
    that the reference's raw indices compact over the real residues
    only."""
    prep_dir, _ = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    synth_groups = [
        np.asarray(g, dtype=int)
        for g in expected["synthetic_ic_groups"]["groups"]
    ]

    log_top_ic_summary(
        synth_groups,
        kstar=3,
        evals_sca=np.array([1.0, 0.6, 0.2]),
        retained_positions=prep.retained_positions,
        msa_obj_orig=prep.msa_obj_orig,
        reference_id="synth_clade_A_0",
        n_logged_comps=10,
    )
    msgs = [r.getMessage() for r in capture_run_sca_logs]

    header = next(m for m in msgs if m.startswith("Top "))
    assert "reference=synth_clade_A_0" in header

    # Clade A reference has raw sequence ACDEFGI (7 residues). Its
    # raw-residue index per ORIGINAL MSA column:
    #   col 0='A' → 0;  col 1='C' → 1;  col 2='-' → gap;
    #   col 3='D' → 2;  col 4='E' → 3;  col 5='-' → gap;
    #   col 6='F' → 4;  col 7='G' → 5;  col 8='-' → gap;
    #   col 9='I' → 6.
    # Since retained_positions excludes cols 2,5,8, no ref gaps appear.
    expected_chain = [
        # proc [0,1,4] → unproc [0,1,6] → ref_pos [0,1,4] → res [A,C,F]
        ("[0, 1, 4]", "[0, 1, 6]", "[0, 1, 4]", "[A, C, F]"),
        # proc [2,3,5] → unproc [3,4,7] → ref_pos [2,3,5] → res [D,E,G]
        ("[2, 3, 5]", "[3, 4, 7]", "[2, 3, 5]", "[D, E, G]"),
        # proc [6] → unproc [9] → ref_pos [6] → res [I]
        ("[6]",       "[9]",       "[6]",       "[I]"),
    ]
    proc_lines = [m for m in msgs if "processed:" in m and "unprocessed" not in m]
    unproc_lines = [m for m in msgs if "unprocessed:" in m]
    ref_pos_lines = [m for m in msgs if "reference pos:" in m]
    ref_res_lines = [m for m in msgs if "reference res:" in m]
    for ic, (p, u, rp, rr) in enumerate(expected_chain):
        assert proc_lines[ic].endswith(p)
        assert unproc_lines[ic].endswith(u)
        assert ref_pos_lines[ic].endswith(rp)
        assert ref_res_lines[ic].endswith(rr)


def test_log_top_ic_summary_synthetic_log_alignment(
        prep_and_sca_dirs, expected, capture_run_sca_logs,
):
    """Alignment sanity: the four indented labels must start at the
    same column, and all four data lists must also start at the same
    column. Otherwise the log is hard to eyeball."""
    prep_dir, _ = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    synth_groups = [
        np.asarray(g, dtype=int)
        for g in expected["synthetic_ic_groups"]["groups"]
    ]

    log_top_ic_summary(
        synth_groups,
        kstar=2,
        evals_sca=np.array([2.5, 1.7, 0.4]),
        retained_positions=prep.retained_positions,
        msa_obj_orig=prep.msa_obj_orig,
        reference_id="synth_clade_B_0",
        n_logged_comps=1,  # just IC 0 is enough for the alignment check
    )
    msgs = [r.getMessage() for r in capture_run_sca_logs]
    indented = [m for m in msgs if m.startswith("    ")]
    # Expect processed, unprocessed, reference pos, reference res for IC 0.
    assert len(indented) == 4

    # Every indented line should have the data-list '[' at the same column.
    bracket_cols = [m.index("[") for m in indented]
    assert len(set(bracket_cols)) == 1, (
        f"Label column misaligned across indented lines: "
        f"{list(zip(bracket_cols, indented))}"
    )
