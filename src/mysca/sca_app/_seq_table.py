"""Per-retained-sequence DataFrame for IC-projection plots.

The IC scatter colors points by biological metadata. Resolution order
for color columns:

  1. Columns from ``sequence_metadata.tsv`` (taxonomic ranks, function
     annotations, …) — merged in by
     :meth:`SCAResults.to_dataframe`. These take priority.
  2. ``organism`` parsed from the FASTA description (when it carries
     UniProt-style ``OS=…`` / ``[Genus species]`` tokens).
  3. ``species_code`` — the UniProt mnemonic (5-letter species code)
     extracted from the trailing token of ``seq_id``. Always derivable
     for Pfam/UniProt-formatted IDs.
  4. ``seq_length`` — ungapped residue count of the retained sequence.
     Useful for spotting partial-vs-full-length protein clusters.

Non-biological columns that the SCA pipeline produces alongside Uᵖ —
sequence weights and per-IC coverage — are intentionally NOT in the
color list. They're metric-y diagnostics, not separators.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from mysca.results import PreprocessingResults, SCAResults
from mysca.sca_app._results_io import ResultsBundle
from mysca.sca_app._uniprot import (
    enrich_dataframe as enrich_with_uniprot,
    has_cached_metadata,
)

UP_PREFIX = "Up_"
NEVER_COLOR_BY = (
    "seq_id", "aligned_sequence", "raw_sequence",
    "in_sample", "sequence_weight",
)
UNIPROT_CACHE_SUBDIR = ".uniprot_cache"
UNIPROT_CACHE_FILE = "metadata.tsv"

# Trailing UniProt-style mnemonic, e.g. "..._H1AD96_PHOPY" → "PHOPY".
_MNEMONIC_RE = re.compile(r"_([A-Z0-9]{2,6})$")
# Description tokens that carry organism info.
_OS_RE = re.compile(r"OS=([^=]+?)(?:\s+(?:OX|GN|PE|SV)=|$)")
_BRACKETS_RE = re.compile(r"\[([^\]]+)\]\s*$")


def _species_code(seq_id: str) -> str | None:
    s = str(seq_id)
    last = s.split("|")[-1]
    m = _MNEMONIC_RE.search(last)
    return m.group(1) if m else None


def _organism_from_description(desc: str | None) -> str | None:
    if not desc:
        return None
    m = _OS_RE.search(desc)
    if m:
        return m.group(1).strip()
    m = _BRACKETS_RE.search(desc)
    if m:
        return m.group(1).strip()
    return None


def _ungapped_length(aligned: str) -> int:
    return sum(1 for c in str(aligned) if c not in ".-")


def uniprot_cache_path(bundle: ResultsBundle):
    """Where ``_uniprot.fetch_uniprot_metadata`` persists its cache for this bundle."""
    return bundle.root / UNIPROT_CACHE_SUBDIR / UNIPROT_CACHE_FILE


def build_sequence_table(
    bundle: ResultsBundle, *, enrich_uniprot: bool = True,
) -> pd.DataFrame:
    """Return the per-sequence DataFrame for IC-projection plots.

    Resolution order:

    1. Prefer the on-disk ``seq_projections.tsv`` written by sca-core
       with ``--save_dataframe``. Already carries merged metadata when
       ``--seq_metadata`` was passed.
    2. Otherwise compute Uᵖ live via
       :meth:`SCAResults.to_dataframe(prep)`.

    In both cases, we annotate with the derived biological columns
    ``species_code`` (always), ``organism`` (when descriptions are
    populated), and ``seq_length`` (when ``aligned_sequence`` is
    present).

    When ``enrich_uniprot`` is True (default) and a UniProt cache TSV
    exists alongside the bundle, the ``uniprot_*`` columns
    (domain/kingdom/phylum/class/order/family/genus, organism, protein
    name, keywords, EC class) are left-joined onto the table. The
    cache is populated only when the page explicitly calls
    ``_uniprot.enrich_dataframe`` with a network-allowed flag — this
    function never hits the network on its own.
    """
    sca = SCAResults.load(bundle.root)
    prep: PreprocessingResults | None = None
    if bundle.seq_projections_tsv is not None:
        df = pd.read_csv(bundle.seq_projections_tsv, sep="\t")
        if bundle.preprocessing_dir is not None:
            prep = PreprocessingResults.load(bundle.preprocessing_dir)
    else:
        if bundle.preprocessing_dir is None:
            raise RuntimeError(
                "No `seq_projections.tsv` in the bundle and no "
                "preprocessing dir to compute Uᵖ from."
            )
        prep = PreprocessingResults.load(bundle.preprocessing_dir)
        df = sca.to_dataframe(prep)

    df["species_code"] = df["seq_id"].astype(str).map(_species_code)

    if "aligned_sequence" in df.columns:
        df["seq_length"] = df["aligned_sequence"].map(_ungapped_length)

    # Description-derived organism, if descriptions were persisted.
    if prep is not None and prep.retained_sequence_descriptions is not None:
        org_map = {
            str(sid): _organism_from_description(desc)
            for sid, desc in zip(
                prep.retained_sequence_ids,
                prep.retained_sequence_descriptions,
            )
        }
        organisms = df["seq_id"].astype(str).map(org_map)
        if organisms.notna().any():
            df["organism"] = organisms

    # Merge UniProt enrichment from on-disk cache if it exists. We never
    # hit the network here — the page kicks off the fetch explicitly.
    cache = uniprot_cache_path(bundle)
    if enrich_uniprot and has_cached_metadata(cache):
        try:
            df = enrich_with_uniprot(df, cache_path=cache)
        except Exception:  # noqa: BLE001 — corrupt cache → fall through
            pass

    return df


def list_up_columns(df: pd.DataFrame) -> list[str]:
    """Return ``Up_<i>`` columns sorted by index."""
    cols = [c for c in df.columns if c.startswith(UP_PREFIX)]
    return sorted(cols, key=lambda c: int(c[len(UP_PREFIX):]))


def list_color_columns(df: pd.DataFrame) -> list[str]:
    """Columns suitable for coloring the scatter — biological signal only.

    Excludes Uᵖ, sequence IDs, and the pipeline-diagnostic columns
    (``sequence_weight``, ``coverage_ic_*``). Metadata columns and the
    derived organism / species / length columns survive. The returned
    list is **ordered**: derived biological signals first, then any
    user-supplied metadata columns, so the scatter's color picker
    surfaces useful defaults early.
    """
    skip = set(NEVER_COLOR_BY) | set(list_up_columns(df))
    coverage_cols = {c for c in df.columns if c.startswith("coverage_ic_")}
    skip |= coverage_cols
    candidates = [c for c in df.columns if c not in skip]

    # Surface biological priors first (UniProt phylogeny → other UniProt
    # fields → derived → metadata in user order).
    priority = [
        "uniprot_kingdom", "uniprot_phylum", "uniprot_class",
        "uniprot_order", "uniprot_family", "uniprot_genus",
        "uniprot_ec_class", "uniprot_organism", "uniprot_keywords",
        "uniprot_protein_name", "uniprot_domain",
        "organism", "species_code", "seq_length",
    ]
    out: list[str] = []
    for p in priority:
        if p in candidates:
            out.append(p)
    for c in candidates:
        if c not in out:
            out.append(c)
    return out


def is_numeric_column(df: pd.DataFrame, col: str) -> bool:
    """True iff ``df[col]`` is numerically typed."""
    return bool(pd.api.types.is_numeric_dtype(df[col]))


def has_user_metadata(df: pd.DataFrame) -> bool:
    """True iff the bundle contributed metadata beyond the basic
    derived columns. Counts UniProt enrichment as "metadata"."""
    derived = {"species_code", "organism", "seq_length"}
    for c in list_color_columns(df):
        if c not in derived:
            return True
    return False


def has_uniprot_enrichment(df: pd.DataFrame) -> bool:
    """True iff any ``uniprot_*`` column is present and non-empty."""
    for c in df.columns:
        if c.startswith("uniprot_") and df[c].notna().any():
            return True
    return False


# Two-step color picker maps {category → [(actual_col, display_label), ...]}.
# Order in the inner list controls the "level" selectbox order.
_PHYLOGENY_RANKS_ORDERED = ("kingdom", "phylum", "class", "order", "family", "genus")
_FUNCTION_SPEC = (
    ("uniprot_ec_class", "EC class (UniProt)"),
    ("uniprot_keywords", "keywords (UniProt)"),
    ("uniprot_protein_name", "protein name (UniProt)"),
)
_SEQUENCE_SPEC = (
    ("uniprot_organism", "organism (UniProt)"),
    ("organism", "organism (FASTA description)"),
    ("species_code", "species mnemonic"),
    ("seq_length", "sequence length"),
    ("uniprot_domain", "domain (UniProt)"),
)


def categorize_color_columns(
    df: pd.DataFrame,
) -> dict[str, list[tuple[str, str]]]:
    """Group color-eligible columns into UI categories for a two-step picker.

    Returns ``{category_label: [(actual_col, display_label), ...]}``.
    Categories without any usable column are omitted entirely.

    Categories:
      * **Phylogeny** — UniProt taxonomic ranks (kingdom → genus).
      * **Function** — UniProt EC class, keywords, protein name.
      * **Sequence-derived** — organism / species mnemonic / length / domain.
      * **Custom metadata** — any other surviving column (uploaded TSV,
        ``sequence_metadata.tsv``, etc.).
    """
    cats: dict[str, list[tuple[str, str]]] = {}

    phylo = []
    for rank in _PHYLOGENY_RANKS_ORDERED:
        col = f"uniprot_{rank}"
        if col in df.columns and df[col].notna().any():
            phylo.append((col, f"{rank} (UniProt)"))
    if phylo:
        cats["Phylogeny"] = phylo

    func = [(c, lbl) for c, lbl in _FUNCTION_SPEC
            if c in df.columns and df[c].notna().any()]
    if func:
        cats["Function"] = func

    seq = [(c, lbl) for c, lbl in _SEQUENCE_SPEC
           if c in df.columns and df[c].notna().any()]
    if seq:
        cats["Sequence-derived"] = seq

    accounted = {c for items in cats.values() for c, _ in items}
    skip = set(NEVER_COLOR_BY) | set(list_up_columns(df)) | accounted
    skip |= {c for c in df.columns if c.startswith("coverage_ic_")}
    custom = []
    for c in df.columns:
        if c in skip:
            continue
        if df[c].notna().any():
            custom.append((c, c))
    if custom:
        cats["Custom metadata"] = custom

    return cats


def merge_user_metadata(df: pd.DataFrame, user_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join ``user_df`` onto ``df`` by ``seq_id``.

    The uploaded DataFrame must have a ``seq_id`` column; all other
    columns become available as color options. Columns colliding with
    existing names get a ``_user`` suffix so the original data is
    preserved.
    """
    if "seq_id" not in user_df.columns:
        raise ValueError("Uploaded metadata must include a `seq_id` column.")
    return df.merge(
        user_df, on="seq_id", how="left", suffixes=("", "_user"),
    )
