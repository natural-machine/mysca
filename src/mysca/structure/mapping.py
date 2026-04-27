"""Map MSA sequence IDs to PDB structures.

Two lookup sources:

- **User-supplied TSV** (primary): ``SequencePdbMap.from_tsv(path)``
  reads a 2- or 3-column TSV:

        seq_id<TAB>pdb_path[<TAB>chain]

  ``chain`` is optional; if omitted the first chain is used.

- **SIFTS on-demand**: ``SequencePdbMap.from_sifts_for_uniprot_ids``
  fetches EBI PDBe ``best_structures`` for each UniProt accession
  (cached locally under ``./.sifts_cache/``) and builds a map
  pointing at pre-downloaded PDB files in ``pdb_dir``.
"""

import logging
import os
import urllib.error
from dataclasses import dataclass
from typing import Iterable, Optional

from mysca.structure.fetcher import download_pdb_file
from mysca.structure.sifts import best_structure_entry

logger = logging.getLogger("mysca.structure.mapping")


@dataclass(frozen=True)
class PdbEntry:
    pdb_path: str
    chain: Optional[str] = None


class SequencePdbMap:
    """Mapping from MSA sequence ID to a ``PdbEntry``."""

    def __init__(self, mapping: dict[str, PdbEntry]):
        self._map = dict(mapping)

    def __contains__(self, seq_id: str) -> bool:
        return seq_id in self._map

    def __getitem__(self, seq_id: str) -> PdbEntry:
        return self._map[seq_id]

    def __len__(self) -> int:
        return len(self._map)

    def __iter__(self):
        return iter(self._map)

    def items(self):
        return self._map.items()

    def keys(self):
        return self._map.keys()

    def get(self, seq_id: str, default=None):
        return self._map.get(seq_id, default)

    @classmethod
    def from_tsv(cls, path: str) -> "SequencePdbMap":
        """Read a TSV of ``seq_id\\tpdb_path[\\tchain]``.

        Blank lines and lines starting with ``#`` are ignored. Relative
        ``pdb_path`` entries are resolved relative to the TSV's
        directory.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SequencePdbMap TSV not found: {path}")
        base = os.path.dirname(os.path.abspath(path))
        mapping: dict[str, PdbEntry] = {}
        with open(path) as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.rstrip("\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) not in (2, 3):
                    raise ValueError(
                        f"{path}:{lineno} expected 2 or 3 tab-separated "
                        f"fields, got {len(parts)}: {line!r}"
                    )
                seq_id = parts[0].strip()
                pdb_path = parts[1].strip()
                chain = parts[2].strip() if len(parts) == 3 else None
                if not os.path.isabs(pdb_path):
                    pdb_path = os.path.normpath(os.path.join(base, pdb_path))
                if seq_id in mapping:
                    raise ValueError(
                        f"{path}:{lineno} duplicate seq_id {seq_id!r}"
                    )
                mapping[seq_id] = PdbEntry(pdb_path=pdb_path, chain=chain)
        return cls(mapping)

    @classmethod
    def from_sifts_for_uniprot_ids(
        cls,
        uniprot_ids: Iterable[str],
        *,
        pdb_dir: str,
        cache_dir: Optional[str] = None,
        pdb_suffix: str = ".pdb",
        strict: bool = True,
        timeout: float = 10.0,
        force_refresh: bool = False,
        fetch: bool = False,
        pdb_source: str = "rcsb",
        pdb_form: str = "asym",
        force_refetch: bool = False,
    ) -> "SequencePdbMap":
        """Resolve UniProt IDs to ``PdbEntry`` via EBI SIFTS.

        For each UniProt accession, queries EBI's
        ``mappings/best_structures/{id}`` endpoint and takes the
        top-ranked PDB entry. The resulting ``pdb_path`` is
        ``{pdb_dir}/{pdb_id}{pdb_suffix}``. Callers are responsible
        for pre-downloading the structure files into ``pdb_dir``
        (SIFTS does not ship PDB files) — unless ``fetch=True``, in
        which case missing PDBs are downloaded on demand into
        ``pdb_dir``.

        Args:
            uniprot_ids: iterable of UniProt accessions. The accession
                itself becomes the key of the returned map.
            pdb_dir: directory containing the downloaded PDB files.
                Created when ``fetch=True`` and missing.
            cache_dir: local cache for SIFTS JSON responses
                (default ``./.sifts_cache/``).
            pdb_suffix: filename suffix for the on-disk PDBs (e.g.
                ``".pdb"`` or ``".cif"``). Default ``.pdb``.
            strict: when True (default), raise ``FileNotFoundError``
                if a resolved PDB file is missing from ``pdb_dir``.
                When False, log a warning and skip the entry.
            timeout: HTTP timeout (seconds) per UniProt query and per
                PDB fetch.
            force_refresh: bypass the local SIFTS cache and re-query
                SIFTS.
            fetch: when True, download missing PDB files into
                ``pdb_dir`` via ``mysca.structure.fetcher.download_pdb_file``
                before falling through to the strict / non-strict
                missing-file path. Off by default.
            pdb_source: ``"rcsb"`` (default) or ``"pdbe"``. Only used
                when ``fetch=True``.
            pdb_form: ``"asym"`` (default), ``"assembly1"``, or
                ``"assembly2"``. Only used when ``fetch=True``.
            force_refetch: bypass the on-disk PDB cache and re-download.
                Only used when ``fetch=True``.

        Returns:
            A ``SequencePdbMap`` keyed by UniProt accession.

        Raises:
            FileNotFoundError: when ``strict=True`` and a resolved
                PDB is missing from ``pdb_dir`` (and ``fetch`` is off
                or the fetch attempt failed).
        """
        if not os.path.isdir(pdb_dir):
            if fetch:
                os.makedirs(pdb_dir, exist_ok=True)
            else:
                raise FileNotFoundError(
                    f"pdb_dir not found: {pdb_dir}. Pre-download PDB files "
                    "(e.g. from RCSB) into this directory before calling "
                    "from_sifts_for_uniprot_ids, or pass fetch=True."
                )

        mapping: dict[str, PdbEntry] = {}
        for uniprot_id in uniprot_ids:
            entry = best_structure_entry(
                uniprot_id,
                cache_dir=cache_dir,
                timeout=timeout,
                force_refresh=force_refresh,
            )
            if entry is None:
                logger.info(
                    "SIFTS has no best_structures for %s; skipping.",
                    uniprot_id,
                )
                continue
            pdb_id = entry.get("pdb_id")
            chain = entry.get("chain_id")
            if not pdb_id:
                logger.warning(
                    "SIFTS entry for %s is missing pdb_id: %r", uniprot_id, entry,
                )
                continue
            # SIFTS returns lowercase PDB IDs; filenames in the wild
            # are often uppercase (e.g. 1SHF.pdb from RCSB). Try both
            # cases so users don't have to rename their files.
            lower_path = os.path.join(
                pdb_dir, f"{pdb_id.lower()}{pdb_suffix}",
            )
            upper_path = os.path.join(
                pdb_dir, f"{pdb_id.upper()}{pdb_suffix}",
            )
            if os.path.isfile(lower_path):
                pdb_path = lower_path
            elif os.path.isfile(upper_path):
                pdb_path = upper_path
            elif fetch:
                fetch_error: Optional[Exception] = None
                try:
                    pdb_path = download_pdb_file(
                        pdb_id,
                        dest_dir=pdb_dir,
                        source=pdb_source,
                        form=pdb_form,
                        force_refresh=force_refetch,
                        timeout=timeout,
                    )
                    logger.info(
                        "Fetched %s from %s/%s -> %s",
                        pdb_id, pdb_source, pdb_form, pdb_path,
                    )
                except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                    fetch_error = exc
                    logger.warning(
                        "Fetch failed for %s (%s/%s): %s",
                        pdb_id, pdb_source, pdb_form, exc,
                    )
                if fetch_error is not None:
                    msg = (
                        f"SIFTS resolved {uniprot_id} to PDB {pdb_id} "
                        f"(chain {chain}), and the on-demand fetch from "
                        f"{pdb_source}/{pdb_form} failed ({fetch_error}). "
                        f"Download it manually into {pdb_dir} or adjust "
                        "--pdb_source / --pdb_form."
                    )
                    if strict:
                        raise FileNotFoundError(msg)
                    logger.warning("%s (skipping)", msg)
                    continue
            else:
                msg = (
                    f"SIFTS resolved {uniprot_id} to PDB {pdb_id} "
                    f"(chain {chain}), but neither "
                    f"{os.path.basename(lower_path)} nor "
                    f"{os.path.basename(upper_path)} exists under "
                    f"{pdb_dir}. Download it from RCSB or adjust "
                    "--pdb_suffix."
                )
                if strict:
                    raise FileNotFoundError(msg)
                logger.warning("%s (skipping)", msg)
                continue
            mapping[uniprot_id] = PdbEntry(pdb_path=pdb_path, chain=chain)
        return cls(mapping)
