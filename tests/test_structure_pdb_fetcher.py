"""Unit tests for ``mysca.structure.fetcher.download_pdb_file``.

Never hits the network. ``urllib.request.urlopen`` is patched to
return canned PDB bodies or HTTP errors. Mirrors the patterns in
``tests/test_structure_sifts.py``.
"""

import io
import os
import urllib.error
from unittest.mock import patch

import pytest

from mysca.structure.fetcher import (
    DEFAULT_PDB_CACHE_DIR,
    USER_AGENT,
    _URL_TEMPLATES,
    download_pdb_file,
)


# Minimal valid PDB body (HEADER + END is enough — we never re-parse it
# with Bio.PDB in these tests; we just check round-tripping bytes).
_PDB_BODY = b"HEADER    DEMO\nEND\n"


def _fake_urlopen(body=_PDB_BODY):
    """Context-manager-compatible stand-in for ``urllib.request.urlopen``
    that yields ``body`` from ``.read()``."""
    class _Response:
        def read(self):
            return body
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
    return _Response()


def _http_error(code):
    return urllib.error.HTTPError(
        url="x", code=code, msg="", hdrs=None, fp=io.BytesIO(),
    )


def test_default_cache_dir_constant():
    """Documents the on-disk default; runners assume this matches the
    CLI default in run_structure.py."""
    assert DEFAULT_PDB_CACHE_DIR == ".pdb_cache"


def test_user_agent_is_set():
    """We send a User-Agent — RCSB has been known to reject anonymous
    Python urllib clients."""
    assert USER_AGENT.startswith("mysca/")


def test_url_templates_cover_expected_pairs():
    """Lock the supported (source, form) matrix."""
    assert set(_URL_TEMPLATES) == {
        ("rcsb",  "asym"),
        ("rcsb",  "assembly1"),
        ("rcsb",  "assembly2"),
        ("pdbe",  "asym"),
    }


def test_download_pdb_writes_file_to_dest_dir(tmp_path):
    with patch(
        "mysca.structure.fetcher.urllib.request.urlopen",
        return_value=_fake_urlopen(),
    ) as mock_open:
        path = download_pdb_file("1SHF", dest_dir=str(tmp_path))
    assert path == str(tmp_path / "1shf.pdb")
    assert os.path.isfile(path)
    with open(path, "rb") as f:
        assert f.read() == _PDB_BODY
    mock_open.assert_called_once()


def test_download_pdb_uses_cache(tmp_path):
    with patch(
        "mysca.structure.fetcher.urllib.request.urlopen",
        return_value=_fake_urlopen(),
    ) as mock_open:
        download_pdb_file("1SHF", dest_dir=str(tmp_path))
        download_pdb_file("1SHF", dest_dir=str(tmp_path))  # cache hit
    mock_open.assert_called_once()


def test_download_pdb_force_refresh_bypasses_cache(tmp_path):
    with patch(
        "mysca.structure.fetcher.urllib.request.urlopen",
        return_value=_fake_urlopen(),
    ) as mock_open:
        download_pdb_file("1SHF", dest_dir=str(tmp_path))
        download_pdb_file(
            "1SHF", dest_dir=str(tmp_path), force_refresh=True,
        )
    assert mock_open.call_count == 2


def test_download_pdb_normalizes_filename_lowercase(tmp_path):
    """Pass mixed-case ID; on-disk filename must be lowercase so the
    case-insensitive mapping.py lookup finds it."""
    with patch(
        "mysca.structure.fetcher.urllib.request.urlopen",
        return_value=_fake_urlopen(),
    ):
        path = download_pdb_file("1Q16", dest_dir=str(tmp_path))
    assert os.path.basename(path) == "1q16.pdb"


def test_download_pdb_creates_dest_dir(tmp_path):
    dest = tmp_path / "fresh" / "deeply" / "nested"
    assert not dest.exists()
    with patch(
        "mysca.structure.fetcher.urllib.request.urlopen",
        return_value=_fake_urlopen(),
    ):
        path = download_pdb_file("1abc", dest_dir=str(dest))
    assert dest.is_dir()
    assert os.path.isfile(path)


def test_download_pdb_unknown_source_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        download_pdb_file("1abc", dest_dir=str(tmp_path), source="elsewhere")


def test_download_pdb_unknown_form_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        download_pdb_file("1abc", dest_dir=str(tmp_path), form="trimer")


def test_download_pdb_404_raises(tmp_path):
    """SIFTS swallows 404 (no-match-as-empty); the fetcher must NOT —
    a missing PDB at the source is a hard error."""
    with patch(
        "mysca.structure.fetcher.urllib.request.urlopen",
        side_effect=_http_error(404),
    ):
        with pytest.raises(urllib.error.HTTPError):
            download_pdb_file("9zzz", dest_dir=str(tmp_path))


@pytest.mark.parametrize("source,form,expected_url", [
    ("rcsb",  "asym",       "https://files.rcsb.org/download/1ABC.pdb"),
    ("rcsb",  "assembly1",  "https://files.rcsb.org/download/1ABC.pdb1"),
    ("rcsb",  "assembly2",  "https://files.rcsb.org/download/1ABC.pdb2"),
    ("pdbe",  "asym",       "https://www.ebi.ac.uk/pdbe/entry-files/download/pdb1abc.ent"),
])
def test_download_pdb_url_for_each_source_form_combination(
        tmp_path, source, form, expected_url,
):
    with patch(
        "mysca.structure.fetcher.urllib.request.urlopen",
        return_value=_fake_urlopen(),
    ) as mock_open:
        download_pdb_file(
            "1abc", dest_dir=str(tmp_path), source=source, form=form,
        )
    request = mock_open.call_args[0][0]
    assert request.full_url == expected_url
    # User-Agent header is set on every request.
    assert request.get_header("User-agent", "").startswith("mysca/")


def test_download_pdb_assembly_form_uses_pdb1_extension_but_saves_as_pdb(tmp_path):
    """`assembly1` fetches `.pdb1` from RCSB but the on-disk filename
    is still normalized to `{id}.pdb` for stable downstream lookup."""
    with patch(
        "mysca.structure.fetcher.urllib.request.urlopen",
        return_value=_fake_urlopen(),
    ) as mock_open:
        path = download_pdb_file(
            "1abc", dest_dir=str(tmp_path), form="assembly1",
        )
    assert mock_open.call_args[0][0].full_url.endswith(".pdb1")
    assert os.path.basename(path) == "1abc.pdb"


def test_download_pdb_pdbe_source_uses_ent_url_but_saves_as_pdb(tmp_path):
    with patch(
        "mysca.structure.fetcher.urllib.request.urlopen",
        return_value=_fake_urlopen(),
    ) as mock_open:
        path = download_pdb_file(
            "1abc", dest_dir=str(tmp_path), source="pdbe",
        )
    assert mock_open.call_args[0][0].full_url.endswith("pdb1abc.ent")
    assert os.path.basename(path) == "1abc.pdb"
