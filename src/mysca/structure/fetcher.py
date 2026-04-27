"""Download PDB files on demand.

Mirror of ``mysca.structure.sifts``: stdlib ``urllib`` only,
aggressive on-disk caching under ``dest_dir``, conservative network
etiquette. Tests patch ``urllib.request.urlopen``.

The fetcher is opt-in (``sca-structure --fetch``). On a cache hit it
returns the existing path without any network call. The on-disk
filename is always normalized to ``{pdb_id_lower}.pdb`` regardless of
``source`` so the existing case-insensitive lookup in
``mapping.SequencePdbMap.from_sifts_for_uniprot_ids`` finds the file.

Currently supported (source, form) pairs:

- ``("rcsb", "asym")``    -> ``files.rcsb.org/download/{ID}.pdb``
- ``("rcsb", "assembly1")``-> ``files.rcsb.org/download/{ID}.pdb1``
- ``("rcsb", "assembly2")``-> ``files.rcsb.org/download/{ID}.pdb2``
- ``("pdbe", "asym")``    -> ``ebi.ac.uk/pdbe/entry-files/download/pdb{id}.ent``

mmCIF support is intentionally out of scope: ``mysca.structure.pdb``
parses ``.pdb`` only via Bio.PDB.PDBParser. Adding mmCIF dispatch is a
separate workstream.
"""

import logging
import os
import urllib.error
import urllib.request

from mysca import __version__ as _MYSCA_VERSION

logger = logging.getLogger("mysca.structure.fetcher")


DEFAULT_PDB_CACHE_DIR = ".pdb_cache"
USER_AGENT = f"mysca/{_MYSCA_VERSION} (+https://github.com/natural-machine/mysca)"

# URL templates keyed by (source, form). Each must format with at
# least {id_lower} or {id_upper}; the concrete templates pick whichever
# their endpoint expects.
_URL_TEMPLATES: dict[tuple[str, str], str] = {
    ("rcsb",  "asym"):       "https://files.rcsb.org/download/{id_upper}.pdb",
    ("rcsb",  "assembly1"):  "https://files.rcsb.org/download/{id_upper}.pdb1",
    ("rcsb",  "assembly2"):  "https://files.rcsb.org/download/{id_upper}.pdb2",
    ("pdbe",  "asym"):       "https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{id_lower}.ent",
}


def _resolve_url(pdb_id: str, source: str, form: str) -> str:
    template = _URL_TEMPLATES.get((source, form))
    if template is None:
        raise ValueError(
            f"Unsupported (source, form) for download_pdb_file: "
            f"({source!r}, {form!r}). Supported: "
            f"{sorted(_URL_TEMPLATES.keys())}"
        )
    return template.format(id_lower=pdb_id.lower(), id_upper=pdb_id.upper())


def _dest_path(pdb_id: str, dest_dir: str) -> str:
    """Cache filename is always lowercase ``{id}.pdb``, even for the
    PDBe ``.ent`` source — keeps mapping.py's case-insensitive lookup
    working without per-source filename rules."""
    return os.path.join(dest_dir, f"{pdb_id.lower()}.pdb")


def download_pdb_file(
        pdb_id: str,
        *,
        dest_dir: str,
        source: str = "rcsb",
        form: str = "asym",
        force_refresh: bool = False,
        timeout: float = 30.0,
) -> str:
    """Download ``pdb_id`` from ``source``/``form`` into ``dest_dir``.

    Cache hit (existing ``{dest_dir}/{pdb_id_lower}.pdb`` and
    ``force_refresh=False``) avoids the network entirely. On miss,
    fetches the URL, writes the body to disk, and returns the path.

    Args:
        pdb_id: PDB accession (case-insensitive).
        dest_dir: directory to read the cache from / write the fetched
            file into. Created on first use.
        source: ``"rcsb"`` (default) or ``"pdbe"``.
        form: structure form. ``"asym"`` (default; asymmetric unit),
            ``"assembly1"`` / ``"assembly2"`` (biological assembly,
            RCSB only).
        force_refresh: bypass the on-disk cache and re-download.
        timeout: HTTP timeout in seconds.

    Returns:
        Absolute or relative path to the saved file under ``dest_dir``.

    Raises:
        ValueError: unknown (source, form) combination.
        urllib.error.HTTPError: 404 or other HTTP failures (NOT
            swallowed — a missing PDB ID at the source is a hard error
            here, unlike SIFTS's 404-as-no-match convention).
        urllib.error.URLError: network-level failures.
    """
    url = _resolve_url(pdb_id, source, form)
    cache_path = _dest_path(pdb_id, dest_dir)

    if not force_refresh and os.path.isfile(cache_path):
        logger.debug("PDB cache hit for %s: %s", pdb_id, cache_path)
        return cache_path

    os.makedirs(dest_dir, exist_ok=True)
    logger.info("Fetching PDB %s from %s/%s: %s", pdb_id, source, form, url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    with open(cache_path, "wb") as f:
        f.write(body)
    return cache_path
