"""Tests for the mysca.structure subpackage and sca-structure CLI.

Loader tests use the real Soil3 fixture. Project-roundtrip tests
generate a minimal but well-formatted PDB on the fly whose sequence
matches a retained MSA sequence, so the in-sample short-circuit path
can reproduce the sector assignments already persisted in
``statsectors_msa.npz``.
"""

import json
import os
import shutil
from typing import Iterable

import numpy as np
import pytest

from tests.conftest import DATDIR, TMPDIR, remove_dir

from mysca.results import PreprocessingResults, SCAResults
from mysca.structure import (
    PDBStructure,
    SequencePdbMap,
    PdbEntry,
    project_pdb,
    project_groups_to_pdb,
    struct2seq,
)
from mysca.io import load_pdb_structure
from mysca.run_preprocessing import (
    parse_args as prep_parse_args,
    main as prep_main,
)
from mysca.run_sca import parse_args as sca_parse_args, main as sca_main
from mysca.run_structure import (
    parse_args as structure_parse_args,
    main as structure_main,
)


_MAFFT = shutil.which("mafft") is not None
needs_mafft = pytest.mark.skipif(not _MAFFT, reason="mafft not on PATH")


SOIL3_PDB = f"{DATDIR}/structs/Soil3.scaffold_414071996_c1_8.pdb"
SOIL3_FASTA = f"{DATDIR}/seqs/Soil3.scaffold_414071996_c1_8.fasta"

PREP_ARGS = f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring7a.txt"
SCA_ARGS = f"{DATDIR}/entrypoint_tests/sca_run/argstrings/argstring7a.txt"


# Three-letter codes for the 20 standard amino acids.
_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _write_minimal_pdb(
    seq: str, path: str, *, chain: str = "A",
    residue_numbers: Iterable[int] | None = None,
) -> None:
    """Write a PDB with one CA atom per residue (sufficient for PPBuilder).

    Formatting follows the fixed-width PDB ATOM record spec so Biopython
    parses chain / residue name / residue number correctly:

        cols 1-6:   'ATOM  '
        cols 7-11:  atom serial (right-justified)
        cols 13-16: atom name (left-justified within columns 13-16,
                    with the element starting at col 14 for 1-char
                    elements: ' CA ')
        col 17:     altLoc (blank)
        cols 18-20: residue name (3-letter)
        col 22:     chain ID
        cols 23-26: residue sequence number
        cols 31-38: x coord (8.3f)
        cols 39-46: y coord
        cols 47-54: z coord
        cols 55-60: occupancy (6.2f)
        cols 61-66: temperature factor
        cols 77-78: element symbol
    """
    if residue_numbers is None:
        residue_numbers = range(1, len(seq) + 1)
    residue_numbers = list(residue_numbers)
    assert len(residue_numbers) == len(seq)
    with open(path, "w") as f:
        for i, (aa, resn) in enumerate(zip(seq, residue_numbers), start=1):
            three = _ONE_TO_THREE[aa]
            x = float(i)
            y = 0.0
            z = 0.0
            # Hand-rolled to get the fixed widths right.
            line = (
                f"ATOM  {i:>5d} "            # cols 1-12 ('ATOM  ' + serial + space)
                f" CA "                       # cols 13-16 atom name
                f" "                          # col 17 altLoc
                f"{three} "                   # cols 18-21 (resname + space)
                f"{chain}"                    # col 22 chain
                f"{resn:>4d}"                 # cols 23-26 residue number
                f"    "                       # cols 27-30 (iCode + 3 spaces)
                f"{x:8.3f}{y:8.3f}{z:8.3f}"   # cols 31-54 coords
                f"  1.00 20.00          "     # cols 55-76
                f" C  "                       # cols 77-80 element
                "\n"
            )
            f.write(line)
        f.write("END\n")


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


@pytest.fixture(scope="module")
def prep_and_sca_dirs(tmp_path_factory):
    prep_dir = str(tmp_path_factory.mktemp("structure_prep"))
    sca_dir = str(tmp_path_factory.mktemp("structure_sca"))
    _run_prep_and_sca(prep_dir, sca_dir, sectors_for="all")
    yield prep_dir, sca_dir


# ----------------------------------------------------------------------
# PDBStructure loader tests (use the real Soil3 fixture).
# ----------------------------------------------------------------------


def test_pdb_structure_from_file_returns_sequence_and_residue_ids():
    pdb = PDBStructure.from_file(SOIL3_PDB)
    assert pdb.structure_id == "Soil3.scaffold_414071996_c1_8"
    assert len(pdb.sequence) > 0
    assert len(pdb.residue_ids) == len(pdb.sequence)
    assert all(isinstance(r, int) for r in pdb.residue_ids)

    # struct2seq (back-compat helper) agrees on the primary sequence.
    seq2 = struct2seq(pdb.structure)
    assert pdb.sequence == seq2

    # Chain resolution: Soil3 has one chain.
    assert pdb.chain_id  # non-empty

    # Residue IDs are monotonically non-decreasing for a single chain.
    for a, b in zip(pdb.residue_ids, pdb.residue_ids[1:]):
        assert b >= a


def test_pdb_structure_chain_not_found_raises(tmp_path):
    pdb_path = str(tmp_path / "tiny.pdb")
    _write_minimal_pdb("ACDE", pdb_path, chain="B")
    with pytest.raises(KeyError, match="Chain"):
        PDBStructure.from_file(pdb_path, chain="Z")


def test_pdb_structure_missing_file():
    with pytest.raises(FileNotFoundError):
        PDBStructure.from_file("/nonexistent/x.pdb")


# ----------------------------------------------------------------------
# SequencePdbMap tests.
# ----------------------------------------------------------------------


def test_sequence_pdb_map_from_tsv(tmp_path):
    pdb_rel = tmp_path / "rel.pdb"
    pdb_rel.write_text("(stub)")
    tsv = tmp_path / "map.tsv"
    tsv.write_text(
        "# comment\n"
        "seq_a\trel.pdb\tA\n"
        "\n"
        f"seq_b\t{pdb_rel.resolve()}\n"
    )
    m = SequencePdbMap.from_tsv(str(tsv))
    assert len(m) == 2
    assert "seq_a" in m and "seq_b" in m
    assert m["seq_a"].chain == "A"
    # Relative path resolved against the TSV's directory.
    assert os.path.isabs(m["seq_a"].pdb_path)
    assert m["seq_b"].chain is None


def test_sequence_pdb_map_rejects_bad_rows(tmp_path):
    tsv = tmp_path / "bad.tsv"
    tsv.write_text("just_one_col\n")
    with pytest.raises(ValueError, match="expected 2 or 3"):
        SequencePdbMap.from_tsv(str(tsv))


def test_sifts_lookup_is_registered_but_unimplemented():
    with pytest.raises(NotImplementedError):
        SequencePdbMap.from_sifts_for_uniprot_ids(["Q9Y6K5"])


# ----------------------------------------------------------------------
# Project-PDB end-to-end with in-sample short-circuit.
# ----------------------------------------------------------------------


def test_project_pdb_in_sample_matches_statsectors(prep_and_sca_dirs, tmp_path):
    """Build a synthetic PDB whose sequence matches a retained training
    sequence (in-sample path). IC-group residues from the projection
    must match what statsectors_msa.npz already has for that sequence,
    and project_groups_to_pdb must push those through to PDB residue
    numbers 1..L_raw.
    """
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    sca = SCAResults.load(sca_dir)

    seq_id = prep.retained_sequence_ids[0]
    donor = next(r for r in prep.msa_obj_orig if r.id == seq_id)
    raw = str(donor.seq).replace("-", "")

    # Residue numbers 10..10+L-1 so the PDB-resi→raw-resi offset is
    # non-trivial (not just 1:1 with the 0-based raw index + 1).
    residue_numbers = list(range(10, 10 + len(raw)))
    pdb_path = str(tmp_path / f"{seq_id}.pdb")
    _write_minimal_pdb(raw, pdb_path, residue_numbers=residue_numbers)

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

    # Compare membership to statsectors_msa.npz (top-kstar only — the
    # only groups expanded per sequence per the kstar scoping).
    kstar = sca.kstar
    stats = sca.statsectors_msa
    for ic in range(kstar):
        key = f"group_{ic}_{seq_id}"
        if key not in stats:
            continue
        expected_raw = np.sort(np.asarray(stats[key], dtype=int))
        got_raw = np.sort(
            np.asarray(proj.sequence_projection.ic_memberships[ic], dtype=int)
        )
        assert np.array_equal(got_raw, expected_raw), (
            f"IC {ic} raw-residue mismatch: got={got_raw.tolist()}, "
            f"expected={expected_raw.tolist()}"
        )
        # PDB-residue projection: each raw index r should map to
        # residue_numbers[r] (i.e. r + 10).
        expected_pdb = [residue_numbers[r] for r in got_raw.tolist()]
        got_pdb = sorted(proj.ic_pdb_residues[ic])
        assert got_pdb == expected_pdb


def test_project_groups_to_pdb_length_guard(prep_and_sca_dirs, tmp_path):
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)
    seq_id = prep.retained_sequence_ids[0]
    donor = next(r for r in prep.msa_obj_orig if r.id == seq_id)
    raw = str(donor.seq).replace("-", "")

    pdb_path = str(tmp_path / "ok.pdb")
    _write_minimal_pdb(raw, pdb_path)
    pdb = PDBStructure.from_file(pdb_path)

    proj = project_pdb(
        pdb,
        sca_result_dir=sca_dir,
        preproc_result_dir=prep_dir,
        seq_id=seq_id,
    )
    # Construct a mismatched PDB (different length).
    short_pdb_path = str(tmp_path / "short.pdb")
    _write_minimal_pdb(raw[:-1], short_pdb_path)
    short_pdb = PDBStructure.from_file(short_pdb_path)

    with pytest.raises(ValueError, match="Raw-sequence length mismatch"):
        project_groups_to_pdb(proj.sequence_projection, short_pdb)


# ----------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------


def test_sca_structure_cli_single_pdb(prep_and_sca_dirs, tmp_path):
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)

    seq_id = prep.retained_sequence_ids[0]
    donor = next(r for r in prep.msa_obj_orig if r.id == seq_id)
    raw = str(donor.seq).replace("-", "")

    pdb_path = str(tmp_path / f"{seq_id}.pdb")
    _write_minimal_pdb(raw, pdb_path)

    out_dir = str(tmp_path / "structure_out")
    args = structure_parse_args([
        "-s", pdb_path,
        "--seq_id", seq_id,
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out_dir,
        "-v", "0",
    ])
    structure_main(args)

    assert os.path.isfile(os.path.join(out_dir, "structure_projection.json"))
    assert os.path.isfile(os.path.join(out_dir, "structure_args.json"))
    assert os.path.isfile(os.path.join(out_dir, "structure.log"))
    per_dir = os.path.join(out_dir, "per_structure")
    assert os.path.isdir(per_dir)
    tsvs = [f for f in os.listdir(per_dir) if f.endswith(".tsv")]
    assert len(tsvs) == 1

    with open(os.path.join(out_dir, "structure_projection.json")) as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["structure_id"]  # non-empty
    # pdb_path must be recorded so sca-pymol can re-load the structure.
    assert "pdb_path" in data[0]
    assert os.path.isfile(data[0]["pdb_path"])
    assert os.path.abspath(data[0]["pdb_path"]) == os.path.abspath(pdb_path)


def test_sca_structure_cli_seq_map(prep_and_sca_dirs, tmp_path):
    prep_dir, sca_dir = prep_and_sca_dirs
    prep = PreprocessingResults.load(prep_dir)

    # Build two PDBs for two retained sequences, listed in a TSV.
    chosen_ids = list(prep.retained_sequence_ids[:2])
    tsv_lines = []
    for sid in chosen_ids:
        donor = next(r for r in prep.msa_obj_orig if r.id == sid)
        raw = str(donor.seq).replace("-", "")
        pdb_path = tmp_path / f"{sid}.pdb"
        _write_minimal_pdb(raw, str(pdb_path))
        tsv_lines.append(f"{sid}\t{pdb_path.resolve()}\n")
    tsv = tmp_path / "map.tsv"
    tsv.write_text("".join(tsv_lines))

    out_dir = str(tmp_path / "structure_seqmap_out")
    args = structure_parse_args([
        "--seq_map", str(tsv),
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", out_dir,
        "-v", "0",
    ])
    structure_main(args)
    with open(os.path.join(out_dir, "structure_projection.json")) as f:
        data = json.load(f)
    assert len(data) == 2
    assert sorted(d["structure_id"] for d in data) == sorted(chosen_ids)
    for entry in data:
        assert "pdb_path" in entry
        assert os.path.isfile(entry["pdb_path"])


def test_cli_rejects_both_or_neither_input(tmp_path):
    with pytest.raises(SystemExit):
        structure_parse_args([
            "--preprocessing", str(tmp_path),
            "--scacore", str(tmp_path),
            "-o", str(tmp_path / "o"),
        ])
    with pytest.raises(SystemExit):
        structure_parse_args([
            "-s", "x.pdb", "--seq_map", "y.tsv",
            "--preprocessing", str(tmp_path),
            "--scacore", str(tmp_path),
            "-o", str(tmp_path / "o"),
        ])
