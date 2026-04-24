"""Unit tests for mysca.structure.sifts and SIFTS-driven SequencePdbMap.

Never hits the network. ``urllib.request.urlopen`` is patched to
return canned JSON responses or HTTP errors; responses are cached
to a per-test tmp directory so runs don't touch the user's home.
"""

import io
import json
import os
import urllib.error
from unittest.mock import patch

import pytest

from mysca.structure.sifts import (
    fetch_best_structures_for_uniprot,
    best_structure_entry,
    SIFTS_BEST_STRUCTURES_URL,
)
from mysca.structure.mapping import SequencePdbMap, PdbEntry


def _fake_urlopen(payload):
    """Return a context-manager-compatible stand-in for
    ``urllib.request.urlopen`` that yields the JSON-encoded ``payload``
    on ``.read()``."""
    class _Response:
        def __init__(self, body):
            self._body = body
        def read(self):
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
    body = json.dumps(payload).encode("utf-8")
    return _Response(body)


def _http_error(code):
    return urllib.error.HTTPError(
        url="x", code=code, msg="", hdrs=None, fp=io.BytesIO(),
    )


# ---------------------------------------------------------------------- #
# fetch_best_structures_for_uniprot                                      #
# ---------------------------------------------------------------------- #


def test_fetch_best_structures_returns_list_under_uniprot_key(tmp_path):
    payload = {
        "P00742": [
            {"pdb_id": "1c4v", "chain_id": "A", "coverage": 0.9, "resolution": 1.9},
            {"pdb_id": "1fjs", "chain_id": "B", "coverage": 0.87, "resolution": 2.2},
        ],
    }
    with patch(
        "mysca.structure.sifts.urllib.request.urlopen",
        return_value=_fake_urlopen(payload),
    ) as mock_open:
        got = fetch_best_structures_for_uniprot(
            "P00742", cache_dir=str(tmp_path),
        )
    assert got == payload["P00742"]
    mock_open.assert_called_once()
    called_url = mock_open.call_args[0][0]
    assert called_url == SIFTS_BEST_STRUCTURES_URL.format(uniprot_id="P00742")


def test_fetch_uses_cache_on_second_call(tmp_path):
    payload = {"P0": [{"pdb_id": "1abc", "chain_id": "A"}]}
    with patch(
        "mysca.structure.sifts.urllib.request.urlopen",
        return_value=_fake_urlopen(payload),
    ) as mock_open:
        fetch_best_structures_for_uniprot("P0", cache_dir=str(tmp_path))
        fetch_best_structures_for_uniprot("P0", cache_dir=str(tmp_path))
    # Second call should have hit the cache, not the network.
    assert mock_open.call_count == 1
    cache_file = tmp_path / "P0.json"
    assert cache_file.is_file()
    assert json.loads(cache_file.read_text()) == payload["P0"]


def test_fetch_force_refresh_bypasses_cache(tmp_path):
    payload_a = {"P0": [{"pdb_id": "1aaa", "chain_id": "A"}]}
    payload_b = {"P0": [{"pdb_id": "1bbb", "chain_id": "B"}]}
    # First call populates cache.
    with patch(
        "mysca.structure.sifts.urllib.request.urlopen",
        return_value=_fake_urlopen(payload_a),
    ):
        first = fetch_best_structures_for_uniprot(
            "P0", cache_dir=str(tmp_path),
        )
    assert first[0]["pdb_id"] == "1aaa"

    with patch(
        "mysca.structure.sifts.urllib.request.urlopen",
        return_value=_fake_urlopen(payload_b),
    ) as mock_open:
        refreshed = fetch_best_structures_for_uniprot(
            "P0", cache_dir=str(tmp_path), force_refresh=True,
        )
    assert mock_open.call_count == 1
    assert refreshed[0]["pdb_id"] == "1bbb"


def test_fetch_404_returns_empty_list(tmp_path):
    with patch(
        "mysca.structure.sifts.urllib.request.urlopen",
        side_effect=_http_error(404),
    ):
        got = fetch_best_structures_for_uniprot(
            "NOPE999", cache_dir=str(tmp_path),
        )
    assert got == []


def test_fetch_non_404_propagates(tmp_path):
    with patch(
        "mysca.structure.sifts.urllib.request.urlopen",
        side_effect=_http_error(500),
    ):
        with pytest.raises(urllib.error.HTTPError):
            fetch_best_structures_for_uniprot(
                "P0", cache_dir=str(tmp_path),
            )


def test_best_structure_entry_returns_first_or_none(tmp_path):
    payload = {"P0": [
        {"pdb_id": "1aaa", "chain_id": "A"},
        {"pdb_id": "2bbb", "chain_id": "X"},
    ]}
    with patch(
        "mysca.structure.sifts.urllib.request.urlopen",
        return_value=_fake_urlopen(payload),
    ):
        top = best_structure_entry("P0", cache_dir=str(tmp_path))
    assert top == {"pdb_id": "1aaa", "chain_id": "A"}

    with patch(
        "mysca.structure.sifts.urllib.request.urlopen",
        side_effect=_http_error(404),
    ):
        top_missing = best_structure_entry("NONE", cache_dir=str(tmp_path))
    assert top_missing is None


# ---------------------------------------------------------------------- #
# SequencePdbMap.from_sifts_for_uniprot_ids                              #
# ---------------------------------------------------------------------- #


def _touch(path, contents="placeholder"):
    with open(path, "w") as f:
        f.write(contents)


def test_from_sifts_happy_path(tmp_path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    _touch(pdb_dir / "1aaa.pdb")
    _touch(pdb_dir / "2bbb.pdb")

    cache_dir = tmp_path / "cache"
    payloads = {
        "P0": {"P0": [{"pdb_id": "1aaa", "chain_id": "A"}]},
        "P1": {"P1": [{"pdb_id": "2bbb", "chain_id": "B"}]},
    }

    def _side_effect(url, timeout=None):
        uniprot = url.rsplit("/", 1)[-1]
        return _fake_urlopen(payloads[uniprot])

    with patch(
        "mysca.structure.sifts.urllib.request.urlopen",
        side_effect=_side_effect,
    ):
        m = SequencePdbMap.from_sifts_for_uniprot_ids(
            ["P0", "P1"],
            pdb_dir=str(pdb_dir),
            cache_dir=str(cache_dir),
        )

    assert set(m.keys()) == {"P0", "P1"}
    assert m["P0"] == PdbEntry(
        pdb_path=str(pdb_dir / "1aaa.pdb"), chain="A",
    )
    assert m["P1"].chain == "B"
    assert m["P1"].pdb_path == str(pdb_dir / "2bbb.pdb")


def test_from_sifts_strict_raises_on_missing_pdb(tmp_path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()  # empty — no 1aaa.pdb in it
    cache_dir = tmp_path / "cache"
    payload = {"P0": [{"pdb_id": "1aaa", "chain_id": "A"}]}

    with patch(
        "mysca.structure.sifts.urllib.request.urlopen",
        return_value=_fake_urlopen({"P0": payload["P0"]}),
    ):
        with pytest.raises(FileNotFoundError, match="1aaa"):
            SequencePdbMap.from_sifts_for_uniprot_ids(
                ["P0"], pdb_dir=str(pdb_dir), cache_dir=str(cache_dir),
                strict=True,
            )


def test_from_sifts_non_strict_skips_missing(tmp_path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    _touch(pdb_dir / "1aaa.pdb")  # only the first id's file exists
    cache_dir = tmp_path / "cache"
    payloads = {
        "P0": {"P0": [{"pdb_id": "1aaa", "chain_id": "A"}]},
        "P1": {"P1": [{"pdb_id": "2bbb", "chain_id": "B"}]},
    }

    def _side_effect(url, timeout=None):
        uniprot = url.rsplit("/", 1)[-1]
        return _fake_urlopen(payloads[uniprot])

    with patch(
        "mysca.structure.sifts.urllib.request.urlopen",
        side_effect=_side_effect,
    ):
        m = SequencePdbMap.from_sifts_for_uniprot_ids(
            ["P0", "P1"],
            pdb_dir=str(pdb_dir),
            cache_dir=str(cache_dir),
            strict=False,
        )
    assert list(m.keys()) == ["P0"]  # P1 skipped


def test_from_sifts_skips_uniprots_without_matches(tmp_path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    cache_dir = tmp_path / "cache"

    def _side_effect(url, timeout=None):
        # Always 404.
        raise _http_error(404)

    with patch(
        "mysca.structure.sifts.urllib.request.urlopen",
        side_effect=_side_effect,
    ):
        m = SequencePdbMap.from_sifts_for_uniprot_ids(
            ["P_MISSING"],
            pdb_dir=str(pdb_dir),
            cache_dir=str(cache_dir),
        )
    assert len(m) == 0


def test_from_sifts_requires_pdb_dir_to_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="pdb_dir not found"):
        SequencePdbMap.from_sifts_for_uniprot_ids(
            ["P0"],
            pdb_dir=str(tmp_path / "does_not_exist"),
            cache_dir=str(tmp_path / "cache"),
        )


def test_from_sifts_finds_uppercase_pdb_filename(tmp_path):
    """SIFTS returns lowercase pdb_ids (`1shf`), but RCSB downloads
    are typically uppercase (`1SHF.pdb`). The lookup must try both.

    On case-insensitive filesystems (default macOS HFS+/APFS)
    ``os.path.isfile("1shf.pdb")`` already returns True even when
    only ``1SHF.pdb`` exists, so we can't distinguish which path
    the code returned — but the SIFTS-mode lookup must still succeed."""
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    # Only the uppercase file exists on disk.
    _touch(pdb_dir / "1SHF.pdb")
    cache_dir = tmp_path / "cache"
    payload = {"P06241": [{"pdb_id": "1shf", "chain_id": "A"}]}

    with patch(
        "mysca.structure.sifts.urllib.request.urlopen",
        return_value=_fake_urlopen(payload),
    ):
        m = SequencePdbMap.from_sifts_for_uniprot_ids(
            ["P06241"], pdb_dir=str(pdb_dir), cache_dir=str(cache_dir),
        )
    assert "P06241" in m
    assert os.path.isfile(m["P06241"].pdb_path)
    assert m["P06241"].chain == "A"
    # pdb_path basename is the lowercase or uppercase form — both
    # resolve to the same file on case-insensitive filesystems.
    assert os.path.basename(m["P06241"].pdb_path).lower() == "1shf.pdb"


def test_from_sifts_strict_error_names_both_case_variants(tmp_path):
    """When neither the lower- nor upper-cased filename exists, the
    error message must mention both so the user can choose which to
    download."""
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    cache_dir = tmp_path / "cache"
    payload = {"P0": [{"pdb_id": "1abc", "chain_id": "A"}]}

    with patch(
        "mysca.structure.sifts.urllib.request.urlopen",
        return_value=_fake_urlopen(payload),
    ):
        with pytest.raises(FileNotFoundError) as excinfo:
            SequencePdbMap.from_sifts_for_uniprot_ids(
                ["P0"], pdb_dir=str(pdb_dir), cache_dir=str(cache_dir),
                strict=True,
            )
    msg = str(excinfo.value)
    assert "1abc.pdb" in msg
    assert "1ABC.pdb" in msg
