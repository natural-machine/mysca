"""SIFTS best-structures lookup for UniProt IDs.

Thin wrapper around EBI's PDBe ``mappings/best_structures/{uniprot_id}``
endpoint. Given a UniProt accession, returns the list of PDB entries
ranked by SIFTS best-structure criteria (resolution, coverage, etc.)
so downstream code can resolve a "best-available" structure without
the caller curating a TSV by hand.

Responses are cached under ``cache_dir`` (defaults to
``~/.mysca/sifts_cache/``) as ``<uniprot_id>.json``. Cache hits avoid
the network call entirely — critical for tests, CI, and repeat runs
over the same set of UniProt IDs.

No new Python dep: uses ``urllib.request`` from the stdlib. For tests
that need to simulate responses, patch ``urllib.request.urlopen``.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger("mysca.structure.sifts")


SIFTS_BEST_STRUCTURES_URL = (
    "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot_id}"
)
DEFAULT_CACHE_DIR = os.path.expanduser("~/.mysca/sifts_cache")


def _cache_dir(cache_dir: Optional[str]) -> str:
    return cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR


def _cache_path(uniprot_id: str, cache_dir: Optional[str]) -> str:
    return os.path.join(_cache_dir(cache_dir), f"{uniprot_id}.json")


def fetch_best_structures_for_uniprot(
        uniprot_id: str,
        *,
        cache_dir: Optional[str] = None,
        timeout: float = 10.0,
        force_refresh: bool = False,
) -> list[dict]:
    """Return the SIFTS ``best_structures`` list for one UniProt ID.

    Args:
        uniprot_id: UniProt accession (e.g. ``"P00742"``).
        cache_dir: directory for the JSON cache. Default
            ``~/.mysca/sifts_cache/``. Created on first use.
        timeout: HTTP timeout in seconds.
        force_refresh: bypass the cache and re-fetch.

    Returns:
        The list under the UniProt key in EBI's JSON response, in
        SIFTS rank order (best-first). Empty list when the endpoint
        returns 404 (the UniProt ID has no known PDB structures).

    Raises:
        urllib.error.URLError: network-level failures.
        json.JSONDecodeError: the endpoint returned malformed JSON.
    """
    cache_path = _cache_path(uniprot_id, cache_dir)
    if not force_refresh and os.path.isfile(cache_path):
        logger.debug("SIFTS cache hit for %s: %s", uniprot_id, cache_path)
        with open(cache_path) as f:
            cached = json.load(f)
        return cached

    url = SIFTS_BEST_STRUCTURES_URL.format(uniprot_id=uniprot_id)
    logger.info("Fetching SIFTS best_structures for %s: %s", uniprot_id, url)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            logger.warning(
                "SIFTS returned 404 for %s; treating as no-match.",
                uniprot_id,
            )
            payload = {}
        else:
            raise

    entries = payload.get(uniprot_id, [])
    os.makedirs(_cache_dir(cache_dir), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(entries, f)
    return entries


def best_structure_entry(
        uniprot_id: str,
        *,
        cache_dir: Optional[str] = None,
        **fetch_kwargs,
) -> Optional[dict]:
    """Convenience: return the top-ranked SIFTS entry for one UniProt
    ID, or ``None`` if there are no matches."""
    entries = fetch_best_structures_for_uniprot(
        uniprot_id, cache_dir=cache_dir, **fetch_kwargs,
    )
    return entries[0] if entries else None
