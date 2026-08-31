"""Fetch an AlphaFold structure + PAE for a family member.

Optional convenience for ``sca-examine``: pick a sequence from the family
(random, or a caller-chosen ``seq_id``), resolve its UniProt accession, and
download the AlphaFold model (PDB) and predicted aligned error (PAE JSON). A
user who already has a structure they trust should pass it directly with
``sca-examine --structure``; this module only covers the automated-lookup path.

The AlphaFold DB covers essentially all UniProt accessions, but a given one can
occasionally be missing (obsolete entry, fragment), so we draw a few sequences
at random and use the first that resolves.

API: https://alphafold.ebi.ac.uk/api/prediction/{accession} -> JSON list; each
entry exposes ``pdbUrl`` and ``paeDocUrl``. See
https://alphafold.ebi.ac.uk/api-docs.
"""

import json
import logging
import os
import urllib.error
import urllib.request

import numpy as np

from mysca import __version__ as _MYSCA_VERSION

logger = logging.getLogger("mysca.examine.alphafold")

ALPHAFOLD_PREDICTION_URL = "https://alphafold.ebi.ac.uk/api/prediction/{}"
_USER_AGENT = f"mysca/{_MYSCA_VERSION} (+https://alphafold.ebi.ac.uk/api-docs)"


def extract_accession(seq_id):
    """``'A0A6J1W0I0_9SAUR/25-273'`` -> ``'A0A6J1W0I0'``.

    Strips an alignment range after ``/`` and the organism mnemonic after the
    first ``_`` (UniProt entry-name ``ACCESSION_ORGANISM`` convention)."""
    token = str(seq_id).split("/", 1)[0]
    return token.split("_", 1)[0]


def alphafold_prediction(accession, timeout=30):
    """AlphaFold prediction list for an accession, or None if absent (HTTP 404
    or empty list)."""
    url = ALPHAFOLD_PREDICTION_URL.format(accession)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    return data or None


def _download(url, dest, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as fh:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
    return dest


def fetch_alphafold_structure(seq_ids, out_dir, *, rng=None, seq_id=None,
                              max_tries=5):
    """Download an AlphaFold PDB + PAE for a family member into ``out_dir``.

    When ``seq_id`` is given, only that sequence is tried. Otherwise sequences
    are drawn at random (seeded by ``rng``) until one resolves in AlphaFold.

    Returns a metadata dict (seq_id, accession, entryId, file paths). Raises
    RuntimeError if nothing resolves within ``max_tries``.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = rng or np.random.default_rng(0)
    seq_ids = np.asarray(seq_ids).astype(str)
    n = len(seq_ids)

    if seq_id is not None:
        candidates = [str(seq_id)]
    else:
        # Draw without replacement up to max_tries distinct sequences.
        order = rng.permutation(n)
        candidates = [str(seq_ids[i]) for i in order[:max_tries]]

    tried = []
    for attempt, sid in enumerate(candidates, start=1):
        accession = extract_accession(sid)
        if accession in tried:
            continue
        tried.append(accession)
        logger.info("  [%d/%d] %s -> accession %s",
                    attempt, len(candidates), sid, accession)
        try:
            pred = alphafold_prediction(accession)
        except urllib.error.URLError as e:
            logger.warning("      network error: %s; trying another.", e)
            continue
        if not pred:
            logger.info("      not in AlphaFold DB; trying another.")
            continue

        entry = pred[0]
        entry_id = entry.get("entryId", f"AF-{accession}-F1")
        pdb_url = entry.get("pdbUrl")
        pae_url = entry.get("paeDocUrl")
        if not pdb_url or not pae_url:
            logger.info("      entry lacks pdbUrl/paeDocUrl; trying another.")
            continue

        pdb_path = os.path.join(out_dir, f"{entry_id}.pdb")
        pae_path = os.path.join(out_dir, f"{entry_id}_predicted_aligned_error.json")
        logger.info("      downloading %s (v%s)...",
                    entry_id, entry.get("latestVersion"))
        _download(pdb_url, pdb_path)
        _download(pae_url, pae_path)

        meta = {
            "seq_id": sid,
            "accession": accession,
            "entryId": entry_id,
            "uniprotAccession": entry.get("uniprotAccession"),
            "latestVersion": entry.get("latestVersion"),
            "pdbUrl": pdb_url,
            "paeDocUrl": pae_url,
            "pdb_path": pdb_path,
            "pae_path": pae_path,
            "n_attempts": attempt,
        }
        with open(os.path.join(out_dir, "structure_source.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("      wrote %s + PAE", pdb_path)
        return meta

    raise RuntimeError(
        f"No AlphaFold structure found after {len(tried)} attempt(s) "
        f"(accessions tried: {', '.join(tried)}).")


def load_pae_json(path):
    """Load an AlphaFold PAE JSON into a square float array of predicted
    aligned error, or None if the file has no recognisable PAE field."""
    data = json.load(open(path))
    if isinstance(data, list):
        data = data[0] if data else {}
    if isinstance(data, dict) and "predicted_aligned_error" in data:
        return np.asarray(data["predicted_aligned_error"], dtype=np.float64)
    # Legacy AlphaFold PAE format: [{"distance": [[...]]}] or {"distance": ...}
    if isinstance(data, dict) and "distance" in data:
        return np.asarray(data["distance"], dtype=np.float64)
    logger.warning("No 'predicted_aligned_error' field in %s; ignoring PAE.", path)
    return None
