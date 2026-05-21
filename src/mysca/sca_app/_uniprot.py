"""Batch-fetch UniProt metadata for a list of accessions.

The IC scatter wants real biology to color by — taxonomic ranks
(domain → genus), protein name, EC class, keywords. UniProt's REST
API exposes all of that keyed on accession, and we can extract a
UniProt accession from any ``sp|…|`` or ``tr|…|`` seq_id (100% hit
rate on the test bundle's Pfam-style IDs).

Public surface:

  - :func:`extract_uniprot_accessions` — pull accession out of every seq_id.
  - :func:`fetch_uniprot_metadata` — batched HTTP fetch into a tidy
    DataFrame, with on-disk TSV caching so reruns are network-free.
  - :func:`parse_lineage` — split UniProt's ``Name (rank), Name (rank), …``
    lineage string into a ``{rank: name}`` dict.
  - :func:`enrich_dataframe` — left-join derived ``uniprot_*`` columns
    (domain/kingdom/phylum/class/order/family/genus, organism, protein
    name, keywords, EC class) onto a per-sequence DataFrame.

Failure modes are wide:
  - HTTPError / URLError → propagated; callers convert to a user message.
  - Accessions UniProt no longer knows about → no row in the response;
    they end up with NaN in the joined columns. Other rows are unaffected.
"""

from __future__ import annotations

import io
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

from mysca import __version__ as _MYSCA_VERSION

logger = logging.getLogger("mysca.sca_app.uniprot")

UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
USER_AGENT = f"mysca/{_MYSCA_VERSION} (+sca-app)"
DEFAULT_FIELDS = (
    "accession,organism_name,organism_id,lineage,"
    "protein_name,keyword,ec"
)
DEFAULT_BATCH_SIZE = 100  # UniProt rejects queries past ~3500 URL chars
ACCESSION_KEY = "accession"

# Trailing ``_PHOPY``-style 5-letter mnemonic in a seq_id.
_PIPE_ACC_RE = re.compile(r"^(?:sp|tr)\|([A-Z0-9]+)(?:\.\d+)?\|", re.IGNORECASE)
_BARE_ACC_RE = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9](?:[A-Z0-9]{4,5})?$")
_LINEAGE_SEGMENT_RE = re.compile(r"^(.+?) \(([^)]+)\)$")

# Ranks we surface as standalone columns. Order matters — the page lists
# columns in this order so the broad-to-narrow phylogeny reads naturally.
PHYLOGENY_RANKS = (
    "domain", "kingdom", "phylum", "class",
    "order", "family", "genus",
)

EC_CLASS_NAMES = {
    "1": "1 oxidoreductase",
    "2": "2 transferase",
    "3": "3 hydrolase",
    "4": "4 lyase",
    "5": "5 isomerase",
    "6": "6 ligase",
    "7": "7 translocase",
}


def extract_uniprot_accession(seq_id: str) -> str | None:
    """Pull a UniProt accession out of a seq_id, or ``None``."""
    if not seq_id:
        return None
    m = _PIPE_ACC_RE.match(seq_id)
    if m:
        return m.group(1).upper()
    bare = str(seq_id).split("|")[0].split("/")[0].strip()
    if _BARE_ACC_RE.match(bare):
        return bare
    return None


def extract_uniprot_accessions(seq_ids: Iterable[str]) -> dict[str, str | None]:
    """``{seq_id: accession_or_None}`` for every input id."""
    return {sid: extract_uniprot_accession(sid) for sid in seq_ids}


def parse_lineage(lineage_str: str | None) -> dict[str, str]:
    """``"Eukaryota (domain), Metazoa (kingdom), ..."`` → ``{rank: name}``.

    First-seen wins when a rank label repeats (UniProt sometimes emits
    multiple ``(clade)`` segments).
    """
    if not lineage_str or not isinstance(lineage_str, str):
        return {}
    out: dict[str, str] = {}
    for segment in lineage_str.split(", "):
        m = _LINEAGE_SEGMENT_RE.match(segment.strip())
        if not m:
            continue
        name, rank = m.group(1).strip(), m.group(2).strip()
        if rank in out or rank == "no rank":
            continue
        out[rank] = name
    return out


def _ec_class(ec_str: str | None) -> str | None:
    if not isinstance(ec_str, str) or not ec_str.strip():
        return None
    # UniProt may list multiple EC numbers separated by ``;`` or ``/``.
    first = re.split(r"[;,/]", ec_str.strip(), maxsplit=1)[0].strip()
    head = first.split(".", 1)[0]
    return EC_CLASS_NAMES.get(head)


def _http_get_tsv(query: str, fields: str, timeout: float) -> str:
    params = urllib.parse.urlencode({
        "query": query,
        "format": "tsv",
        "fields": fields,
        "compressed": "false",
    })
    url = f"{UNIPROT_STREAM_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _fetch_chunk(
    accessions: list[str],
    *,
    fields: str,
    timeout: float,
) -> pd.DataFrame:
    if not accessions:
        return pd.DataFrame()
    query = " OR ".join(f"accession:{a}" for a in accessions)
    text = _http_get_tsv(query, fields, timeout)
    return pd.read_csv(io.StringIO(text), sep="\t", dtype=str).fillna("")


def fetch_uniprot_metadata(
    accessions: Iterable[str],
    *,
    cache_path: str | Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: float = 30.0,
    force_refresh: bool = False,
    fields: str = DEFAULT_FIELDS,
) -> pd.DataFrame:
    """Return a DataFrame of raw UniProt fields for ``accessions``.

    Columns are UniProt's verbose header names — ``Entry``, ``Organism``,
    ``Taxonomic lineage``, ``Protein names``, ``Keywords``, ``EC number``,
    ``Organism (ID)`` — renamed so ``Entry`` becomes ``accession`` for
    join clarity. Missing accessions (e.g. obsolete/merged in UniProt)
    silently drop out.

    Parameters
    ----------
    cache_path
        TSV file persisting prior results. On a cache hit only the
        missing accessions are fetched, and the cache is rewritten with
        the union. ``None`` skips caching.
    batch_size
        Accessions per request. ~200 keeps the URL well under typical
        HTTP limits; larger is faster but riskier.
    force_refresh
        Re-fetch everything regardless of cache state.
    """
    wanted = sorted({a for a in accessions if a})
    cache_df = pd.DataFrame()
    if cache_path is not None and not force_refresh:
        p = Path(cache_path)
        if p.is_file():
            try:
                cache_df = pd.read_csv(p, sep="\t", dtype=str).fillna("")
            except Exception:  # noqa: BLE001 — corrupt cache → refetch
                cache_df = pd.DataFrame()

    have = (
        set(cache_df[ACCESSION_KEY].astype(str))
        if ACCESSION_KEY in cache_df.columns else set()
    )
    needed = [a for a in wanted if a not in have]

    fetched_chunks: list[pd.DataFrame] = []
    for i in range(0, len(needed), batch_size):
        chunk = needed[i:i + batch_size]
        logger.info(
            "Fetching UniProt batch %d (%d accessions)",
            i // batch_size + 1, len(chunk),
        )
        df = _fetch_chunk(chunk, fields=fields, timeout=timeout)
        if not df.empty:
            df = df.rename(columns={"Entry": ACCESSION_KEY})
            fetched_chunks.append(df)

    if fetched_chunks:
        new_df = pd.concat(fetched_chunks, ignore_index=True)
        if not cache_df.empty:
            cache_df = pd.concat(
                [cache_df, new_df], ignore_index=True,
            ).drop_duplicates(ACCESSION_KEY, keep="last")
        else:
            cache_df = new_df
        if cache_path is not None:
            p = Path(cache_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            cache_df.to_csv(p, sep="\t", index=False)

    if cache_df.empty:
        return cache_df
    return cache_df[cache_df[ACCESSION_KEY].isin(set(wanted))].reset_index(drop=True)


def derive_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Project raw UniProt fields onto the ``uniprot_*`` columns the
    IC page uses. Returns a NEW DataFrame keyed by ``accession``."""
    if raw.empty:
        return pd.DataFrame(columns=[ACCESSION_KEY])

    lineage_col = "Taxonomic lineage"
    lineage_dicts = raw[lineage_col].apply(parse_lineage) \
        if lineage_col in raw.columns else None

    out = pd.DataFrame({ACCESSION_KEY: raw[ACCESSION_KEY]})
    if lineage_dicts is not None:
        for rank in PHYLOGENY_RANKS:
            out[f"uniprot_{rank}"] = lineage_dicts.apply(lambda d: d.get(rank))
    out["uniprot_organism"] = raw.get("Organism", pd.Series(dtype=str))
    out["uniprot_protein_name"] = raw.get("Protein names", pd.Series(dtype=str))
    out["uniprot_keywords"] = raw.get("Keywords", pd.Series(dtype=str))
    if "EC number" in raw.columns:
        out["uniprot_ec_class"] = raw["EC number"].apply(_ec_class)
    # Replace empty strings with NaN so downstream consumers can
    # ``dropna`` cleanly.
    return out.replace({"": None})


def enrich_dataframe(
    df: pd.DataFrame,
    *,
    cache_path: str | Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: float = 30.0,
    force_refresh: bool = False,
    seq_id_col: str = "seq_id",
) -> pd.DataFrame:
    """Left-join ``uniprot_*`` columns onto ``df`` using accessions
    extracted from ``df[seq_id_col]``. Idempotent: existing
    ``uniprot_*`` columns are dropped before the join, so calling
    twice doesn't multiply columns.
    """
    if seq_id_col not in df.columns:
        raise KeyError(
            f"DataFrame has no `{seq_id_col}` column to extract accessions from."
        )
    accessions = df[seq_id_col].astype(str).map(extract_uniprot_accession)
    raw = fetch_uniprot_metadata(
        accessions.dropna().tolist(),
        cache_path=cache_path,
        batch_size=batch_size,
        timeout=timeout,
        force_refresh=force_refresh,
    )
    enriched = derive_columns(raw)

    df_out = df.drop(
        columns=[c for c in df.columns if c.startswith("uniprot_")],
        errors="ignore",
    )
    df_out = df_out.assign(_uniprot_accession=accessions)
    df_out = df_out.merge(
        enriched,
        how="left",
        left_on="_uniprot_accession",
        right_on=ACCESSION_KEY,
    )
    df_out = df_out.drop(columns=[ACCESSION_KEY, "_uniprot_accession"],
                         errors="ignore")
    return df_out


def has_cached_metadata(cache_path: str | Path) -> bool:
    p = Path(cache_path)
    return p.is_file() and p.stat().st_size > 0
