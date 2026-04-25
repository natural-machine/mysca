"""sca-pymol entrypoint tests.

The live-PyMOL integration test is gated on ``pymol`` being importable
(the default test env does not install ``pymol-open-source``). The
pymol-free tests cover argparse validation and the features-module
loader — both paths a user can hit without any rendering.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import DATDIR

from mysca.run_pymol import (
    parse_args,
    _apply_group_coloring,
    _load_features_module,
    _load_projections,
    _render_frame,
    _split_features_names,
    _write_animation,
    _write_views,
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


def test_parse_args_accepts_spin_ray_dpi():
    args = parse_args([
        "--structure", "x", "-o", "y",
        "--spin_axis", "z", "--spin_degrees", "180",
        "--ray", "none", "--dpi", "150",
    ])
    assert args.spin_axis == "z"
    assert args.spin_degrees == 180.0
    assert args.ray == "none"
    assert args.dpi == 150


def test_parse_args_defaults_preserve_today():
    """Defaults for the new flags preserve pre-Phase-2 behavior:
    Y-axis 360° spin, ray-trace every frame, 300 dpi."""
    args = parse_args(["--structure", "x", "-o", "y"])
    assert args.spin_axis == "y"
    assert args.spin_degrees == 360.0
    assert args.ray == "all"
    assert args.dpi == 300


def test_parse_args_rejects_bad_spin_axis():
    with pytest.raises(SystemExit):
        parse_args(["--structure", "x", "-o", "y", "--spin_axis", "w"])


def test_parse_args_rejects_bad_ray():
    with pytest.raises(SystemExit):
        parse_args(["--structure", "x", "-o", "y", "--ray", "maybe"])


def test_parse_args_accepts_format_choices():
    for fmt in ("gif", "mp4", "both"):
        args = parse_args([
            "--structure", "x", "-o", "y", "--format", fmt,
        ])
        assert args.format == fmt


def test_parse_args_format_defaults_to_gif():
    args = parse_args(["--structure", "x", "-o", "y"])
    assert args.format == "gif"


def test_parse_args_rejects_bad_format():
    with pytest.raises(SystemExit):
        parse_args(["--structure", "x", "-o", "y", "--format", "webm"])


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
# Render helpers (pymol-free via MagicMock).                             #
# ---------------------------------------------------------------------- #


def test_apply_group_coloring_empty_returns_none():
    cmd = MagicMock()
    got = _apply_group_coloring(
        cmd, "sel", pdb_resids=[], scores=[],
        sector_color="0xff0000", sector_style="spheres",
    )
    assert got is None
    cmd.select.assert_not_called()
    cmd.show.assert_not_called()


def test_apply_group_coloring_colors_and_shows():
    cmd = MagicMock()
    got = _apply_group_coloring(
        cmd, "sel", pdb_resids=[10, 12, 14], scores=None,
        sector_color="0xff0000", sector_style="spheres",
    )
    assert got == "sel"
    cmd.select.assert_called_once()
    (name, selection), _ = cmd.select.call_args
    assert name == "sel"
    assert selection == "resi 10+12+14"
    cmd.show.assert_called_once_with("spheres", "sel")
    cmd.color.assert_called_once_with("0xff0000", "sel")
    cmd.set.assert_not_called()


def test_apply_group_coloring_applies_per_residue_alpha():
    cmd = MagicMock()
    got = _apply_group_coloring(
        cmd, "sel", pdb_resids=[10, 20, 30], scores=[0.1, 0.5, 0.9],
        sector_color="0xff0000", sector_style="spheres",
    )
    assert got == "sel"
    # One cmd.set per residue for the transparency mapping.
    assert cmd.set.call_count == 3
    for call in cmd.set.call_args_list:
        (attr, _alpha, sel), _ = call
        assert attr == "sphere_transparency"
        assert sel.startswith("sel and resi ")


def test_apply_group_coloring_handles_degenerate_scores():
    """When all scores are equal, the alpha-mapping formula would
    divide by zero; the helper must short-circuit to alpha=1."""
    cmd = MagicMock()
    _apply_group_coloring(
        cmd, "sel", pdb_resids=[1, 2], scores=[0.5, 0.5],
        sector_color="0xff0000", sector_style="spheres",
    )
    # Each call: (attr, 1 - 1.0, "...") == (attr, 0.0, "...").
    for call in cmd.set.call_args_list:
        (_attr, transparency, _sel), _ = call
        assert transparency == 0.0


def test_write_views_emits_four_rotated_pngs(tmp_path):
    cmd = MagicMock()
    _write_views(cmd, str(tmp_path), "X_group0", ref_scaffold=None)
    assert os.path.isdir(tmp_path / "views")
    # Four PNGs and four rotations of struct.
    assert cmd.png.call_count == 4
    assert cmd.rotate.call_count == 4
    # Each cmd.png ends in the expected view{i}.png.
    for i, call in enumerate(cmd.png.call_args_list):
        (path,), _ = call
        assert path.endswith(f"X_group0_view{i}.png")


def test_write_views_rotates_ref_struct_too(tmp_path):
    cmd = MagicMock()
    _write_views(cmd, str(tmp_path), "X", ref_scaffold="ref_struct")
    # Four rotations for the target + four for the ref = 8.
    assert cmd.rotate.call_count == 8


# ---------------------------------------------------------------------- #
# _render_frame: unified per-frame renderer (pymol-free via MagicMock).  #
# ---------------------------------------------------------------------- #


def _minimal_projection(n_groups=2):
    """Fixture projection with two IC groups carrying disjoint residue
    lists and matching loadings, enough to drive _render_frame without a
    full sca-structure run."""
    return {
        "structure_id": "X",
        "chain_id": "A",
        "pdb_path": "/fake/X.pdb",
        "ic_pdb_residues": [[10, 12], [20, 22, 24]][:n_groups],
        "sequence_projection": {
            "ic_loadings": [[0.1, 0.9], [0.2, 0.5, 0.8]][:n_groups],
        },
    }


def _render_frame_kwargs(cmd, tmp_path, **overrides):
    base = dict(
        cmd=cmd,
        scaffold="X",
        projection=_minimal_projection(),
        group_idxs=[0],
        sector_colors=["0xff0000", "0x00ff00"],
        sector_style="spheres",
        struct_color="gray70",
        ref_scaffold=None,
        feature_fns=[],
        views=False,
        animate=False,
        nframes=4,
        duration=0.4,
        basename="X_group0",
        group_idx_for_features=0,
        outdir=str(tmp_path),
    )
    base.update(overrides)
    return base


def test_render_frame_writes_png_and_cleans_selections(tmp_path):
    cmd = MagicMock()
    _render_frame(**_render_frame_kwargs(cmd, tmp_path))
    cmd.png.assert_called_once()
    (path,), kw = cmd.png.call_args
    assert path == f"{tmp_path}/X_group0.png"
    assert kw.get("dpi") == 300
    # Selection created for group 0 and cleaned up at end.
    cmd.select.assert_called_once()
    assert any(
        c.args[0] == "group_selection0" for c in cmd.delete.call_args_list
    )


def test_render_frame_multigroup_creates_distinct_selections(tmp_path):
    cmd = MagicMock()
    _render_frame(**_render_frame_kwargs(
        cmd, tmp_path,
        group_idxs=[0, 1],
        basename="X_groups_0,1",
        group_idx_for_features=None,
    ))
    # One select call per group with distinct selection names.
    names = [c.args[0] for c in cmd.select.call_args_list]
    assert names == ["group_selection0", "group_selection1"]
    # Both selections deleted at cleanup.
    deleted = [c.args[0] for c in cmd.delete.call_args_list]
    assert "group_selection0" in deleted and "group_selection1" in deleted


def test_render_frame_forwards_group_idx_to_feature_fns(tmp_path):
    cmd = MagicMock()
    seen = {}

    def feature_fn(scaffold, _cmd, *, color=None, context=None):
        seen["scaffold"] = scaffold
        seen["group_idx"] = context["group_idx"]

    _render_frame(**_render_frame_kwargs(
        cmd, tmp_path, feature_fns=[feature_fn], group_idx_for_features=7,
    ))
    assert seen == {"scaffold": "X", "group_idx": 7}


def test_render_frame_empty_group_still_renders(tmp_path):
    """An IC group with zero PDB residues should not raise; the frame is
    still written (the struct-only image) and no selection is created."""
    cmd = MagicMock()
    proj = _minimal_projection()
    proj["ic_pdb_residues"] = [[]]
    proj["sequence_projection"]["ic_loadings"] = [[]]
    _render_frame(**_render_frame_kwargs(
        cmd, tmp_path, projection=proj,
    ))
    cmd.select.assert_not_called()
    cmd.png.assert_called_once()
    # No selection to clean up.
    assert cmd.delete.call_count == 0


def test_render_frame_views_flag_writes_views(tmp_path):
    cmd = MagicMock()
    _render_frame(**_render_frame_kwargs(cmd, tmp_path, views=True))
    assert os.path.isdir(tmp_path / "views")
    # Main PNG + 4 view PNGs.
    assert cmd.png.call_count == 5


def test_render_frame_animate_invokes_write_animation(tmp_path):
    """When animate=True, _write_animation is invoked with the frame's
    basename (not a hardcoded group-index path)."""
    cmd = MagicMock()
    with patch("mysca.run_pymol._write_animation") as mock_anim:
        _render_frame(**_render_frame_kwargs(
            cmd, tmp_path, animate=True, basename="X_groups_0,1",
            group_idxs=[0, 1], group_idx_for_features=None,
        ))
    mock_anim.assert_called_once()
    _args, kw = mock_anim.call_args
    passed = list(_args) + list(kw.values())
    assert "X_groups_0,1" in passed
    assert 4 in passed  # nframes
    assert 0.4 in passed  # duration


def test_render_frame_forwards_dpi_to_main_png(tmp_path):
    cmd = MagicMock()
    _render_frame(**_render_frame_kwargs(cmd, tmp_path, dpi=150))
    (_path,), kw = cmd.png.call_args
    assert kw["dpi"] == 150


def test_render_frame_forwards_animation_kwargs(tmp_path):
    """--spin_axis/--spin_degrees/--ray/--dpi are threaded into
    _write_animation by _render_frame."""
    cmd = MagicMock()
    with patch("mysca.run_pymol._write_animation") as mock_anim:
        _render_frame(**_render_frame_kwargs(
            cmd, tmp_path, animate=True,
            spin_axis="z", spin_degrees=90.0, ray="none", dpi=150,
        ))
    kw = mock_anim.call_args.kwargs
    assert kw["spin_axis"] == "z"
    assert kw["spin_degrees"] == 90.0
    assert kw["ray"] == "none"
    assert kw["dpi"] == 150


# ---------------------------------------------------------------------- #
# _write_animation: spin / ray / dpi behavior (pymol-free via MagicMock
# + patched _load_animation_deps).                                       #
# ---------------------------------------------------------------------- #


def _mock_anim_deps():
    """Patch _load_animation_deps to return MagicMock stand-ins for
    imageio + PIL.Image, so _write_animation runs without either dep."""
    return patch(
        "mysca.run_pymol._load_animation_deps",
        return_value=(MagicMock(), MagicMock()),
    )


def test_write_animation_defaults_preserve_today(tmp_path):
    """Pre-Phase-2 behavior: spin_axis='y', spin_degrees=360,
    ray=1 every frame, dpi=300."""
    cmd = MagicMock()
    with _mock_anim_deps():
        _write_animation(cmd, str(tmp_path), "X", nframes=4, duration=0.4)
    assert [c.args for c in cmd.turn.call_args_list] == [("y", 90.0)] * 4
    for c in cmd.png.call_args_list:
        assert c.kwargs["ray"] == 1
        assert c.kwargs["dpi"] == 300


def test_write_animation_spin_axis_and_degrees(tmp_path):
    cmd = MagicMock()
    with _mock_anim_deps():
        _write_animation(
            cmd, str(tmp_path), "X", nframes=4, duration=0.4,
            spin_axis="x", spin_degrees=180.0,
        )
    assert [c.args for c in cmd.turn.call_args_list] == [("x", 45.0)] * 4


def test_write_animation_ray_none(tmp_path):
    cmd = MagicMock()
    with _mock_anim_deps():
        _write_animation(
            cmd, str(tmp_path), "X", nframes=3, duration=0.3, ray="none",
        )
    assert [c.kwargs["ray"] for c in cmd.png.call_args_list] == [0, 0, 0]


def test_write_animation_ray_first(tmp_path):
    cmd = MagicMock()
    with _mock_anim_deps():
        _write_animation(
            cmd, str(tmp_path), "X", nframes=3, duration=0.3, ray="first",
        )
    assert [c.kwargs["ray"] for c in cmd.png.call_args_list] == [1, 0, 0]


def test_write_animation_ray_all(tmp_path):
    cmd = MagicMock()
    with _mock_anim_deps():
        _write_animation(
            cmd, str(tmp_path), "X", nframes=3, duration=0.3, ray="all",
        )
    assert [c.kwargs["ray"] for c in cmd.png.call_args_list] == [1, 1, 1]


def test_write_animation_dpi_forwards(tmp_path):
    cmd = MagicMock()
    with _mock_anim_deps():
        _write_animation(
            cmd, str(tmp_path), "X", nframes=2, duration=0.2, dpi=150,
        )
    for c in cmd.png.call_args_list:
        assert c.kwargs["dpi"] == 150


# ---------------------------------------------------------------------- #
# _write_animation format: gif / mp4 / both                              #
# ---------------------------------------------------------------------- #


def test_write_animation_format_gif_default(tmp_path):
    """Default format writes exactly one .gif via imageio.mimsave."""
    cmd = MagicMock()
    mock_imageio, mock_Image = MagicMock(), MagicMock()
    with patch("mysca.run_pymol._load_animation_deps",
               return_value=(mock_imageio, mock_Image)):
        _write_animation(cmd, str(tmp_path), "X", nframes=2, duration=0.2)
    assert mock_imageio.mimsave.call_count == 1
    outfile = mock_imageio.mimsave.call_args.args[0]
    assert outfile.endswith("X.gif")
    # GIF path passes `duration`, not `fps`.
    assert "duration" in mock_imageio.mimsave.call_args.kwargs
    assert "fps" not in mock_imageio.mimsave.call_args.kwargs


def test_write_animation_format_mp4(tmp_path):
    """--format mp4: writes exactly one .mp4 via imageio.mimsave with
    fps=nframes/duration. _require_ffmpeg is invoked first."""
    cmd = MagicMock()
    mock_imageio, mock_Image = MagicMock(), MagicMock()
    with patch("mysca.run_pymol._load_animation_deps",
               return_value=(mock_imageio, mock_Image)), \
         patch("mysca.run_pymol._require_ffmpeg") as mock_req:
        _write_animation(
            cmd, str(tmp_path), "X", nframes=4, duration=0.4, format="mp4",
        )
    mock_req.assert_called_once()
    assert mock_imageio.mimsave.call_count == 1
    outfile = mock_imageio.mimsave.call_args.args[0]
    assert outfile.endswith("X.mp4")
    assert mock_imageio.mimsave.call_args.kwargs["fps"] == 10.0


def test_write_animation_format_both(tmp_path):
    """--format both: writes .gif AND .mp4. ffmpeg required."""
    cmd = MagicMock()
    mock_imageio, mock_Image = MagicMock(), MagicMock()
    with patch("mysca.run_pymol._load_animation_deps",
               return_value=(mock_imageio, mock_Image)), \
         patch("mysca.run_pymol._require_ffmpeg") as mock_req:
        _write_animation(
            cmd, str(tmp_path), "X", nframes=2, duration=0.2, format="both",
        )
    mock_req.assert_called_once()
    assert mock_imageio.mimsave.call_count == 2
    outfiles = [c.args[0] for c in mock_imageio.mimsave.call_args_list]
    assert any(p.endswith("X.gif") for p in outfiles)
    assert any(p.endswith("X.mp4") for p in outfiles)


def test_write_animation_format_gif_skips_ffmpeg_check(tmp_path):
    """--format gif must NOT invoke _require_ffmpeg (the whole point of
    keeping gif the default is that imageio-ffmpeg is optional)."""
    cmd = MagicMock()
    mock_imageio, mock_Image = MagicMock(), MagicMock()
    with patch("mysca.run_pymol._load_animation_deps",
               return_value=(mock_imageio, mock_Image)), \
         patch("mysca.run_pymol._require_ffmpeg") as mock_req:
        _write_animation(
            cmd, str(tmp_path), "X", nframes=2, duration=0.2, format="gif",
        )
    mock_req.assert_not_called()


def test_require_ffmpeg_raises_with_hint(monkeypatch):
    """_require_ffmpeg raises ImportError with an install hint when
    imageio_ffmpeg is not importable."""
    from mysca.run_pymol import _require_ffmpeg
    import sys
    real_mod = sys.modules.pop("imageio_ffmpeg", None)
    # Block the import so the call is guaranteed to fail regardless of
    # whether the package is installed on this machine.
    import builtins
    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "imageio_ffmpeg":
            raise ImportError("blocked for test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    try:
        with pytest.raises(ImportError, match="imageio-ffmpeg"):
            _require_ffmpeg()
    finally:
        if real_mod is not None:
            sys.modules["imageio_ffmpeg"] = real_mod


def test_render_frame_forwards_format_to_write_animation(tmp_path):
    cmd = MagicMock()
    with patch("mysca.run_pymol._write_animation") as mock_anim:
        _render_frame(**_render_frame_kwargs(
            cmd, tmp_path, animate=True, format="mp4",
        ))
    assert mock_anim.call_args.kwargs["format"] == "mp4"


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

    # Reuse the msa07 argstrings + _write_minimal_pdb helper that
    # test_structure.py already uses, so the smoke test tracks the
    # same fixture path as the structure tests.
    from tests.conftest import DATDIR
    from tests.test_structure import _write_minimal_pdb
    from mysca.results import PreprocessingResults
    from mysca.run_preprocessing import (
        parse_args as prep_parse_args,
        main as prep_main,
    )
    from mysca.run_sca import (
        parse_args as sca_parse_args,
        main as sca_main,
    )
    from mysca.run_structure import (
        parse_args as structure_parse_args,
        main as structure_main,
    )
    from mysca.run_pymol import main as pymol_main

    def _argstring(path):
        with open(path) as f:
            return f.readline().split(" ")

    prep_args = prep_parse_args(_argstring(
        f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring7a.txt"
    ))
    prep_args.msa_fpath = f"{DATDIR}/{prep_args.msa_fpath}"
    prep_dir = str(tmp_path / "prep")
    prep_args.outdir = prep_dir
    prep_args.verbosity = 0
    prep_main(prep_args)

    sca_args = sca_parse_args(_argstring(
        f"{DATDIR}/entrypoint_tests/sca_run/argstrings/argstring7a.txt"
    ))
    sca_args.indir = prep_dir
    sca_dir = str(tmp_path / "sca")
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
    raw = str(donor.seq).replace("-", "")
    pdb_path = str(tmp_path / "target.pdb")
    _write_minimal_pdb(
        raw, pdb_path,
        residue_numbers=list(range(100, 100 + len(raw))),
    )

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
    # pymol.log is created even at -v 0 (no INFO records emitted, so
    # it may be empty). The PNG assertion above already proves the
    # CLI finished.
    assert os.path.isfile(os.path.join(pymol_out, "pymol.log"))
