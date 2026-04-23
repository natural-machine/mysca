"""Load PDB structures and expose their primary sequence for projection.

The core type is ``PDBStructure``: a thin wrapper around a single
chain's polypeptide sequence plus the list of PDB residue IDs aligned
to that sequence. This is the bridge between SCA's residue-index
coordinate system (0-based into an ungapped amino-acid string) and the
biologist-facing PDB residue numbers.

``struct2seq`` is retained at module scope as a small back-compat
helper for callers that previously imported it from
``mysca.structures`` (plural).
"""

import logging
import os
from typing import Optional

from Bio.Align import MultipleSeqAlignment
from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Structure import Structure
from Bio.SeqUtils import seq1

logger = logging.getLogger("mysca.structure.pdb")


def struct2seq(structure: Structure) -> str:
    """Return the primary amino-acid sequence of a Bio.PDB Structure.

    Uses Biopython's ``PPBuilder`` and returns the sequence of the
    first polypeptide. Retained here so callers previously importing
    from ``mysca.structures`` keep working after the migration.
    """
    ppb = PPBuilder()
    for pp in ppb.build_peptides(structure):
        return str(pp.get_sequence())
    return ""


class PDBStructure:
    """Per-chain primary sequence + residue-ID view of a PDB structure.

    Attributes:
        sequence: One-letter amino-acid string for the chain, in PDB
            residue order (gaps/non-standard residues skipped).
        residue_ids: list of int, same length as ``sequence``;
            ``residue_ids[i]`` is the PDB residue number (the sequence
            number field of Biopython's residue ``.id`` tuple) for the
            amino-acid at position ``i`` of ``sequence``.
        chain_id: Chain identifier (e.g. ``"A"``) loaded.
        structure_id: ID given to the underlying Bio.PDB structure.
        structure: The underlying ``Bio.PDB.Structure.Structure``.
    """

    def __init__(
        self,
        structure: Structure,
        chain_id: str,
        sequence: str,
        residue_ids: list[int],
        structure_id: Optional[str] = None,
    ):
        self.structure = structure
        self.chain_id = chain_id
        self.sequence = sequence
        self.residue_ids = list(residue_ids)
        self.structure_id = structure_id or structure.id

    def __len__(self) -> int:
        return len(self.sequence)

    def residue_id_for(self, seq_idx: int) -> int:
        """Raw-sequence index (0-based) → PDB residue number."""
        return int(self.residue_ids[seq_idx])

    @classmethod
    def from_file(
        cls,
        path: str,
        *,
        chain: Optional[str] = None,
        structure_id: Optional[str] = None,
        quiet: bool = True,
    ) -> "PDBStructure":
        """Load a PDB file; optionally select a chain.

        When ``chain`` is None, the first chain in the first model is
        used. Standard amino-acid residues are included; hetero- or
        non-standard residues (HETATM rows, modified amino acids that
        ``Bio.PDB.is_aa`` rejects in strict mode) are skipped.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"PDB file not found: {path}")
        sid = structure_id or os.path.splitext(os.path.basename(path))[0]
        parser = PDBParser(QUIET=quiet)
        struct = parser.get_structure(sid, path)
        chain_obj, resolved_chain_id = _select_chain(struct, chain)
        seq, resids = _chain_sequence_and_ids(chain_obj)
        if not seq:
            raise ValueError(
                f"No standard amino-acid residues found on chain "
                f"{resolved_chain_id!r} in {path}."
            )
        return cls(
            structure=struct,
            chain_id=resolved_chain_id,
            sequence=seq,
            residue_ids=resids,
            structure_id=sid,
        )


def _select_chain(struct: Structure, chain: Optional[str]):
    model = next(iter(struct))
    if chain is None:
        chain_obj = next(iter(model))
        return chain_obj, chain_obj.id
    try:
        return model[chain], chain
    except KeyError:
        available = [c.id for c in model]
        raise KeyError(
            f"Chain {chain!r} not found; available: {available}"
        )


def _chain_sequence_and_ids(chain_obj):
    """Extract one-letter sequence and aligned PDB residue IDs.

    Skips HETATM / non-standard residues (anything ``is_aa(res,
    standard=True)`` rejects). Residues are taken in chain order as
    Biopython iterates them.
    """
    seq_chars = []
    resids = []
    for res in chain_obj:
        if not is_aa(res, standard=True):
            continue
        one = seq1(res.get_resname())
        if one in ("", "X"):
            continue
        seq_chars.append(one)
        resids.append(int(res.id[1]))
    return "".join(seq_chars), resids
