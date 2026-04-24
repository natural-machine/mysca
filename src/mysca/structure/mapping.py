"""Map MSA sequence IDs to PDB structures.

Two lookup sources:

- **User-supplied TSV** (primary): ``SequencePdbMap.from_tsv(path)``
  reads a 2- or 3-column TSV:

        seq_id<TAB>pdb_path[<TAB>chain]

  ``chain`` is optional; if omitted the first chain is used.

- **SIFTS on-demand**: ``SequencePdbMap.from_sifts_for_uniprot_ids``
  fetches EBI PDBe ``best_structures`` for each UniProt accession
  (cached locally under ``~/.mysca/sifts_cache/``) and builds a map
  pointing at pre-downloaded PDB files in ``pdb_dir``.
"""

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Optional

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
    ) -> "SequencePdbMap":
        """Resolve UniProt IDs to ``PdbEntry`` via EBI SIFTS.

        For each UniProt accession, queries EBI's
        ``mappings/best_structures/{id}`` endpoint and takes the
        top-ranked PDB entry. The resulting ``pdb_path`` is
        ``{pdb_dir}/{pdb_id}{pdb_suffix}``; callers are responsible
        for pre-downloading the structure files into ``pdb_dir``
        (SIFTS does not ship PDB files).

        Args:
            uniprot_ids: iterable of UniProt accessions. The accession
                itself becomes the key of the returned map.
            pdb_dir: directory containing the downloaded PDB files.
            cache_dir: local cache for SIFTS JSON responses
                (default ``~/.mysca/sifts_cache/``).
            pdb_suffix: filename suffix for the on-disk PDBs (e.g.
                ``".pdb"`` or ``".cif"``). Default ``.pdb``.
            strict: when True (default), raise ``FileNotFoundError``
                if a resolved PDB file is missing from ``pdb_dir``.
                When False, log a warning and skip the entry.
            timeout: HTTP timeout (seconds) per UniProt query.
            force_refresh: bypass the local cache and re-query SIFTS.

        Returns:
            A ``SequencePdbMap`` keyed by UniProt accession.

        Raises:
            FileNotFoundError: when ``strict=True`` and a resolved
                PDB is missing from ``pdb_dir``.
        """
        if not os.path.isdir(pdb_dir):
            raise FileNotFoundError(
                f"pdb_dir not found: {pdb_dir}. Pre-download PDB files "
                "(e.g. from RCSB) into this directory before calling "
                "from_sifts_for_uniprot_ids."
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
            pdb_path = os.path.join(
                pdb_dir, f"{pdb_id.lower()}{pdb_suffix}",
            )
            if not os.path.isfile(pdb_path):
                msg = (
                    f"SIFTS resolved {uniprot_id} to PDB {pdb_id} "
                    f"(chain {chain}), but the file is missing: "
                    f"{pdb_path}. Download it from RCSB or adjust "
                    "--pdb_suffix."
                )
                if strict:
                    raise FileNotFoundError(msg)
                logger.warning("%s (skipping)", msg)
                continue
            mapping[uniprot_id] = PdbEntry(pdb_path=pdb_path, chain=chain)
        return cls(mapping)
