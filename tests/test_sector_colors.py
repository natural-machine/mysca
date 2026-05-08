"""Tests for ``mysca.constants.resolve_sector_colors`` and the
``--sector_colors`` CLI surface on sca-core, sca-pymol, and sca-plots.
"""

import json
import os

import pytest

from mysca.constants import SECTOR_COLORS, resolve_sector_colors
from mysca.run_sca import parse_args as sca_parse_args
from mysca.run_pymol import parse_args as pymol_parse_args
from mysca.run_plots import parse_args as plots_parse_args


def test_default_returns_builtin_palette_copy():
    palette = resolve_sector_colors("default")
    assert palette == SECTOR_COLORS
    assert palette is not SECTOR_COLORS  # don't hand out the module-level list

    palette[0] = "#000000"
    assert SECTOR_COLORS[0] != "#000000"


def test_none_returns_none():
    assert resolve_sector_colors("none") is None


def test_none_python_returns_default():
    assert resolve_sector_colors(None) == SECTOR_COLORS


def test_comma_list_parses_and_normalizes_to_hex():
    palette = resolve_sector_colors("#ff0000, #00ff00 ,red")
    assert palette == ["#ff0000", "#00ff00", "#ff0000"]


def test_comma_list_rejects_unknown_token():
    with pytest.raises(ValueError, match="unrecognised color tokens"):
        resolve_sector_colors("#ff0000,notacolor")


def test_empty_comma_list_rejected():
    with pytest.raises(ValueError, match="empty list"):
        resolve_sector_colors(", , ,")


def test_text_file_palette(tmp_path):
    fpath = tmp_path / "palette.txt"
    fpath.write_text(
        "#ff0000\n"
        "  blue  \n"
        "\n"
        "#00ff00\n"
    )
    palette = resolve_sector_colors(str(fpath))
    assert palette == ["#ff0000", "#0000ff", "#00ff00"]


def test_text_file_invalid_token_rejected(tmp_path):
    fpath = tmp_path / "palette.txt"
    fpath.write_text("#ff0000\n# header comment\n#00ff00\n")
    with pytest.raises(ValueError, match="unrecognised color tokens"):
        resolve_sector_colors(str(fpath))


def test_empty_file_rejected(tmp_path):
    fpath = tmp_path / "empty.txt"
    fpath.write_text("\n\n   \n")
    with pytest.raises(ValueError, match="no color entries"):
        resolve_sector_colors(str(fpath))


def test_json_file_palette(tmp_path):
    fpath = tmp_path / "palette.json"
    fpath.write_text(json.dumps(["#ff0000", "blue", "#00ff00"]))
    palette = resolve_sector_colors(str(fpath))
    assert palette == ["#ff0000", "#0000ff", "#00ff00"]


def test_json_file_must_be_flat_string_array(tmp_path):
    fpath = tmp_path / "palette.json"
    fpath.write_text(json.dumps([["#ff0000", "blue"]]))
    with pytest.raises(ValueError, match="flat array of color strings"):
        resolve_sector_colors(str(fpath))


def test_named_listed_cmap_uses_cmap_colors():
    # tab10 is a ListedColormap with 10 entries; we should get all of them
    # back as hex strings.
    palette = resolve_sector_colors("tab10")
    assert len(palette) == 10
    assert all(c.startswith("#") and len(c) == 7 for c in palette)


def test_named_continuous_cmap_sampled_to_default_length():
    palette = resolve_sector_colors("viridis")
    assert len(palette) == len(SECTOR_COLORS)
    assert all(c.startswith("#") for c in palette)
    assert palette[0] != palette[-1]


def test_unknown_cmap_name_raises():
    with pytest.raises(ValueError, match="not a path"):
        resolve_sector_colors("definitely_not_a_cmap")


def test_sca_core_sector_colors_default_and_override():
    args = sca_parse_args(["-i", "x", "-o", "y"])
    assert args.sector_colors == "default"

    args = sca_parse_args([
        "-i", "x", "-o", "y",
        "--sector_colors", "#ff0000,#00ff00",
    ])
    assert args.sector_colors == "#ff0000,#00ff00"


def test_sca_core_dropped_old_sector_cmap_flag():
    with pytest.raises(SystemExit):
        sca_parse_args([
            "-i", "x", "-o", "y", "--sector_cmap", "default",
        ])


def test_sca_pymol_sector_colors_default_and_override():
    args = pymol_parse_args(["--structure", "x", "-o", "y"])
    assert args.sector_colors == "default"

    args = pymol_parse_args([
        "--structure", "x", "-o", "y",
        "--sector_colors", "tab10",
    ])
    assert args.sector_colors == "tab10"


def test_sca_pymol_rejects_sector_colors_none():
    with pytest.raises(SystemExit):
        pymol_parse_args([
            "--structure", "x", "-o", "y", "--sector_colors", "none",
        ])


def test_sca_plots_sector_colors_default_and_override():
    args = plots_parse_args(["--scacore", "x"])
    assert args.sector_colors == "default"

    args = plots_parse_args([
        "--scacore", "x", "--sector_colors", "Set1",
    ])
    assert args.sector_colors == "Set1"
