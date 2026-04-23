"""Map MSA sequence IDs to PDB structures.

Two lookup sources, configurable:

- **User-supplied TSV** (primary): ``SequencePdbMap.from_tsv(path)``
  reads a 2- or 3-column TSV:

        seq_id<TAB>pdb_path[<TAB>chain]

  ``chain`` is optional; if omitted the first chain is used.

- **SIFTS on-demand**: ``SequencePdbMap.from_sifts_for_uniprot_ids``
  is registered as a method but currently raises NotImplementedError.
  Following the same pattern as ``hmmalign`` in ``mysca.project``,
  this keeps the interface shape visible without adding a network
  dependency to the first commit.
"""

import os
from dataclasses import dataclass
from typing import Iterable, Optional


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
        cache_dir: Optional[str] = None,
    ) -> "SequencePdbMap":
        """Resolve UniProt IDs to PDB entries via EBI SIFTS.

        Not yet implemented. Registered as a classmethod so the shape
        is visible for future work; to be filled in once we add a
        mockable HTTP client.
        """
        raise NotImplementedError(
            "SIFTS lookup is not yet implemented. For now, build a "
            "SequencePdbMap via SequencePdbMap.from_tsv()."
        )
