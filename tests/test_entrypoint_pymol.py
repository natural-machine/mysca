"""sca-pymol entrypoint tests.

The live-PyMOL integration test is gated on ``pymol`` being importable
(the default test env does not install ``pymol-open-source``). The
pymol-free tests cover argparse validation and the features-module
loader — both paths a user can hit without any rendering.
"""

import json
import os
from unittest.mock import MagicMock

import pytest

from tests.conftest import DATDIR

from mysca.run_pymol import (
    parse_args,
    _load_features_module,
    _load_projections,
    _split_features_names,
)


try:
    import pymol  # noqa: F401
    _HAS_PYMOL = True
except ImportError:
    _HAS_PYMOL = False
needs_pymol = pytest.mark.skipif(not _HAS_PYMOL, reason="pymol not importable")


FEATURES_FIXTURE = f"{DATDIR}/pymol_features/good.py"


# ---------------------------------------------------------------------- #
# Argparse validation (no pymol required).                               #
# ---------------------------------------------------------------------- #


def test_parse_args_requires_structure_and_outdir():
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["--structure", "x"])
    with pytest.raises(SystemExit):
        parse_args(["-o", "y"])


def test_parse_args_features_without_py_errors():
    """--features without --features_py must error at parse time."""
    with pytest.raises(SystemExit):
        parse_args([
            "--structure", "x",
            "-o", "y",
            "--features", "show_mo",
        ])


def test_parse_args_features_with_py_accepted():
    args = parse_args([
        "--structure", "x",
        "-o", "y",
        "--features_py", FEATURES_FIXTURE,
        "--features", "feature_a,feature_b",
    ])
    assert args.features_py == FEATURES_FIXTURE
    assert args.features == "feature_a,feature_b"


def test_parse_args_features_py_without_names_ok():
    """--features_py alone is allowed (no functions invoked)."""
    args = parse_args([
        "--structure", "x",
        "-o", "y",
        "--features_py", FEATURES_FIXTURE,
    ])
    assert args.features is None


def test_split_features_names_strips_and_filters_empties():
    assert _split_features_names(None) == []
    assert _split_features_names("") == []
    assert _split_features_names("a,b, c , ,d") == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------- #
# Features module loader.                                                #
# ---------------------------------------------------------------------- #


def test_load_features_module_happy_path():
    fns = _load_features_module(FEATURES_FIXTURE, ["feature_a", "feature_b"])
    assert len(fns) == 2
    assert all(callable(fn) for fn in fns)


def test_load_features_module_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Features file not found"):
        _load_features_module(str(tmp_path / "nope.py"), ["anything"])


def test_load_features_module_missing_name():
    with pytest.raises(ValueError, match="has no attribute"):
        _load_features_module(FEATURES_FIXTURE, ["does_not_exist"])


def test_load_features_module_non_callable():
    with pytest.raises(TypeError, match="is not callable"):
        _load_features_module(FEATURES_FIXTURE, ["NOT_CALLABLE"])


def test_load_features_module_runs_with_mock_cmd():
    """Invoke a resolved feature fn with a MagicMock stand-in for
    PyMOL's cmd; confirm it exercises cmd.select / cmd.show."""
    [fn] = _load_features_module(FEATURES_FIXTURE, ["feature_a"])
    mock_cmd = MagicMock()
    fn("my_struct", mock_cmd, color=None, context={"scaffold": "my_struct"})
    mock_cmd.select.assert_called_once()
    mock_cmd.show.assert_called_once()
    # First positional arg of select should be the selection name.
    (name, selection), _kwargs = mock_cmd.select.call_args
    assert name == "feature_a_sel"
    assert "my_struct" in selection


# ---------------------------------------------------------------------- #
# Projection loader.                                                     #
# ---------------------------------------------------------------------- #


def test_load_projections_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="structure_projection.json"):
        _load_projections(str(tmp_path))


def test_load_projections_returns_list(tmp_path):
    payload = [{"structure_id": "X", "chain_id": "A", "pdb_path": "/a.pdb",
                "sequence_projection": {"ic_loadings": []},
                "ic_pdb_residues": []}]
    (tmp_path / "structure_projection.json").write_text(json.dumps(payload))
    got = _load_projections(str(tmp_path))
    assert got == payload


# ---------------------------------------------------------------------- #
# Live-PyMOL integration test. Runs only when both pymol and mafft are
# available in the env — gated by the existing _HAS_MAFFT machinery in
# tests/test_structure.py (imported transitively via
# tests/test_project_synthetic.py's prep_and_sca_dirs fixture).
# ---------------------------------------------------------------------- #


@needs_pymol
def test_sca_pymol_cli_smoke(tmp_path):
    """Drive sca-structure to produce a real structure_projection.json,
    then run sca-pymol on it with --groups 0 and assert the expected
    PNG lands under outdir. No pixel diff."""
    import shutil
    _MAFFT = shutil.which("mafft") is not None
    if not _MAFFT:
        pytest.skip("mafft not on PATH")

    # Reuse the synthetic prep+sca fixture plumbing.
    from tests.test_project_synthetic import _run_prep_and_sca, SYNTHETIC_DIR  # noqa: E501
    from tests.test_structure import _write_minimal_pdb
    from mysca.results import PreprocessingResults
    from mysca.run_structure import (
        parse_args as structure_parse_args,
        main as structure_main,
    )
    from mysca.run_pymol import main as pymol_main

    prep_dir = str(tmp_path / "prep")
    sca_dir = str(tmp_path / "sca")
    _run_prep_and_sca(prep_dir, sca_dir)

    prep = PreprocessingResults.load(prep_dir)
    seq_id = prep.retained_sequence_ids[0]
    donor = next(r for r in prep.msa_obj_orig if r.id == seq_id)
    raw = str(donor.seq).replace("-", "")
    pdb_path = str(tmp_path / f"{seq_id}.pdb")
    _write_minimal_pdb(raw, pdb_path, residue_numbers=list(range(100, 100 + len(raw))))

    struct_out = str(tmp_path / "structure_out")
    structure_main(structure_parse_args([
        "-s", pdb_path,
        "--seq_id", seq_id,
        "--preprocessing", prep_dir,
        "--scacore", sca_dir,
        "-o", struct_out,
        "-v", "0",
    ]))

    pymol_out = str(tmp_path / "pymol_out")
    pymol_main(parse_args([
        "--structure", struct_out,
        "--groups", "0",
        "-o", pymol_out,
        "-v", "0",
    ]))

    produced = [f for f in os.listdir(pymol_out) if f.endswith(".png")]
    assert any(f.endswith("_group0.png") for f in produced), (
        f"expected *_group0.png under {pymol_out}; got {produced}"
    )
    with open(os.path.join(pymol_out, "pymol.log")) as f:
        log = f.read()
    assert "Done!" in log
