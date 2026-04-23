"""PDB / tertiary-structure integration for SCA.

Bridges a PDB structure through ``mysca.project`` so IC-group memberships
can be expressed in PDB residue-number coordinates.

Public surface:

    from mysca.structure import (
        PDBStructure, struct2seq,
        SequencePdbMap, PdbEntry,
        PdbProjection, project_pdb, project_groups_to_pdb,
    )
"""

from mysca.structure.pdb import PDBStructure, struct2seq
from mysca.structure.mapping import SequencePdbMap, PdbEntry
from mysca.structure.projection import (
    PdbProjection,
    project_pdb,
    project_groups_to_pdb,
)

__all__ = [
    "PDBStructure",
    "struct2seq",
    "SequencePdbMap",
    "PdbEntry",
    "PdbProjection",
    "project_pdb",
    "project_groups_to_pdb",
]
