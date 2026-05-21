"""PDB acquisition + parsing helpers for the Streamlit app.

Three responsibilities:

  1. Extract a UniProt accession from an SCA reference ID (handles the
     ``db|accession|entry_name`` format Pfam-style MSAs carry).
  2. Fetch an AlphaFold predicted structure for a UniProt accession,
     caching to disk under ``<bundle>/.pdb_cache/`` so repeat loads are
     network-free.
  3. Parse PDB text (path-or-string) into a :class:`PDBStructure` —
     ``PDBStructure.from_file`` only accepts a path, so we materialize
     uploaded / fetched bytes to a temp file first.

The on-disk RCSB fetcher already lives in :mod:`mysca.structure.fetcher`
but AlphaFold has a different URL convention (``alphafold.ebi.ac.uk``)
and is keyed by UniProt accession, not PDB ID — so it lives here.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from mysca import __version__ as _MYSCA_VERSION
from mysca.structure.pdb import PDBStructure

logger = logging.getLogger("mysca.sca_app.pdb_io")

ALPHAFOLD_API_TEMPLATE = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
USER_AGENT = f"mysca/{_MYSCA_VERSION} (+sca-app)"

_PIPE_RE = re.compile(r"^(?:sp|tr)\|([A-Z0-9]+)(?:\.\d+)?\|")
_BARE_RE = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[A-Z][0-9][A-Z0-9]{3}[0-9](?:[A-Z0-9]{4,5})?$")


def extract_uniprot_accession(reference_id: str) -> str | None:
    """Return the UniProt accession embedded in an SCA reference ID.

    Handles ``sp|P12345|NAME_HUMAN``, ``tr|H1AD96|H1AD96_PHOPY``, and a
    bare accession (``P12345``). Returns ``None`` if no UniProt-looking
    accession can be extracted — callers fall back to manual entry.
    """
    if not reference_id:
        return None
    m = _PIPE_RE.match(reference_id)
    if m:
        return m.group(1)
    bare = reference_id.split("|")[0].split("/")[0].strip()
    if _BARE_RE.match(bare):
        return bare
    return None


def fetch_alphafold_pdb(
    uniprot_acc: str,
    *,
    cache_dir: str | os.PathLike,
    force_refresh: bool = False,
    timeout: float = 30.0,
) -> Path:
    """Download the latest AlphaFold predicted structure for ``uniprot_acc``.

    Resolves the actual file URL via the AlphaFold prediction API
    (``alphafold.ebi.ac.uk/api/prediction/{acc}``) — accessions have
    different ``latestVersion`` values (e.g. H1AD96 is at v6, many
    others at v4), so hardcoding a version causes spurious 404s.

    Caches to ``cache_dir/<basename>`` (the filename AlphaFold uses,
    e.g. ``AF-H1AD96-F1-model_v6.pdb``); a cache hit avoids the network
    entirely.

    Raises:
        urllib.error.HTTPError: AlphaFold has no model for the
            accession (API 404), or the file URL returns non-200.
        urllib.error.URLError: network-level failure.
        ValueError: the API response was empty or missing ``pdbUrl``.
    """
    acc = uniprot_acc.strip().upper()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    api_url = ALPHAFOLD_API_TEMPLATE.format(acc=acc)
    logger.info("Querying AlphaFold API for %s: %s", acc, api_url)
    req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        api_body = resp.read()
    payload = json.loads(api_body)
    if not payload:
        raise ValueError(
            f"AlphaFold API returned no predictions for {acc!r}."
        )
    pdb_url = payload[0].get("pdbUrl")
    if not pdb_url:
        raise ValueError(
            f"AlphaFold API entry for {acc!r} has no `pdbUrl` field."
        )

    dest = cache_dir / Path(pdb_url).name
    if dest.is_file() and not force_refresh:
        return dest
    logger.info("Fetching AlphaFold PDB for %s: %s", acc, pdb_url)
    req = urllib.request.Request(pdb_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    dest.write_bytes(body)
    return dest


def parse_pdb_text(
    pdb_text: str,
    *,
    chain: str | None = None,
    structure_id: str | None = None,
) -> PDBStructure:
    """Parse PDB text (path or raw content) into a :class:`PDBStructure`.

    Writes a temp file when given raw content so we can reuse the
    existing ``PDBStructure.from_file`` codepath without forking the
    parser. Caller is responsible for keeping the returned struct's
    ``pdb_path`` alive only as long as needed — the underlying temp
    file is deleted on process exit (Streamlit reruns keep the
    PDBStructure in session_state so this is fine in practice).
    """
    p = Path(pdb_text)
    if "\n" not in pdb_text and p.is_file():
        return PDBStructure.from_file(
            str(p), chain=chain, structure_id=structure_id,
        )
    fd, tmp_path = tempfile.mkstemp(prefix="sca_app_", suffix=".pdb")
    with os.fdopen(fd, "w") as fh:
        fh.write(pdb_text)
    return PDBStructure.from_file(
        tmp_path, chain=chain, structure_id=structure_id or "uploaded",
    )


def list_chains(pdb: PDBStructure) -> list[str]:
    """List chain IDs present in the loaded PDB structure."""
    model = next(iter(pdb.structure))
    return [c.id for c in model if c.id and c.id != " "]
