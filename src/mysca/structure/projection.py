"""Project a PDB structure onto an existing SCA result.

Thin composition over ``mysca.project.project_sequences``. The PDB's
primary sequence is run through the standard project pipeline; this
module adds the PDB-residue coordinate system on top by composing
with ``PDBStructure.residue_ids``.
"""

import os
import tempfile
from typing import Optional

from mysca.project import project_sequences, ProjectionResult, SequenceProjection
from mysca.structure.pdb import PDBStructure


class PdbProjection:
    """Result of projecting a single PDB structure onto an SCA result.

    Wraps the underlying ``SequenceProjection`` (residue-index
    coordinates) plus a parallel ``ic_pdb_residues`` list that reports
    the same memberships in PDB residue-number coordinates.
    """

    FIELD_DESCRIPTIONS = {
        "structure_id": "PDB structure ID (from filename or explicit override).",
        "chain_id": "Chain selected on the PDB structure.",
        "sequence_projection": (
            "Underlying SequenceProjection from mysca.project, in "
            "raw-residue-index coordinates."
        ),
        "ic_pdb_residues": (
            "Per-IC list of PDB residue numbers "
            "(from PDBStructure.residue_ids) corresponding to "
            "sequence_projection.ic_memberships."
        ),
    }

    def __init__(
        self,
        structure_id: str,
        chain_id: str,
        sequence_projection: SequenceProjection,
        ic_pdb_residues: list[list[int]],
    ):
        self.structure_id = structure_id
        self.chain_id = chain_id
        self.sequence_projection = sequence_projection
        self.ic_pdb_residues = ic_pdb_residues

    def to_dict(self) -> dict:
        return {
            "structure_id": self.structure_id,
            "chain_id": self.chain_id,
            "sequence_projection": self.sequence_projection.to_dict(),
            "ic_pdb_residues": [list(xs) for xs in self.ic_pdb_residues],
        }


def project_groups_to_pdb(
    sequence_projection: SequenceProjection,
    pdb: PDBStructure,
) -> list[list[int]]:
    """Map per-IC raw-residue indices to PDB residue numbers.

    ``sequence_projection.ic_memberships[i]`` is a list of 0-based
    residue indices into the PDB's primary sequence; this function
    looks each up in ``pdb.residue_ids`` to yield biologist-facing PDB
    residue numbers per IC.

    Raises ``IndexError`` if the projection's raw sequence length
    disagrees with the PDB's primary sequence length — this is the
    signal that the alignment did not match the structure.
    """
    if len(sequence_projection.raw_sequence) != len(pdb.sequence):
        raise ValueError(
            f"Raw-sequence length mismatch: projection has "
            f"{len(sequence_projection.raw_sequence)} residues, "
            f"PDB {pdb.structure_id}:{pdb.chain_id} has "
            f"{len(pdb.sequence)}. Cannot map to PDB residue numbers."
        )
    out = []
    for members in sequence_projection.ic_memberships:
        out.append([int(pdb.residue_ids[int(r)]) for r in members])
    return out


def project_pdb(
    pdb: PDBStructure,
    *,
    sca_result_dir: str,
    preproc_result_dir: str,
    seq_id: Optional[str] = None,
    aligner: str = "mafft_add",
    workdir: Optional[str] = None,
    aligner_kwargs: Optional[dict] = None,
) -> PdbProjection:
    """Project a PDB structure's primary sequence onto an SCA result.

    ``seq_id`` controls the FASTA header used for the project step.
    When it matches an entry in the preprocessing output's
    ``msa_obj_orig``, projection short-circuits through the in-sample
    path (no external aligner invoked). Otherwise the default
    ``mafft_add`` aligner is used.
    """
    header = seq_id or pdb.structure_id
    if workdir is None:
        ctx = tempfile.TemporaryDirectory(prefix="mysca_structure_")
        active_workdir = ctx.name
    else:
        os.makedirs(workdir, exist_ok=True)
        ctx = None
        active_workdir = workdir

    try:
        fasta_path = os.path.join(active_workdir, "pdb_seq.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">{header}\n{pdb.sequence}\n")
        result: ProjectionResult = project_sequences(
            fasta_path,
            sca_result_dir=sca_result_dir,
            preproc_result_dir=preproc_result_dir,
            aligner=aligner,
            workdir=active_workdir,
            aligner_kwargs=aligner_kwargs,
        )
    finally:
        if ctx is not None:
            ctx.cleanup()

    [seq_proj] = result.projections
    ic_pdb_residues = project_groups_to_pdb(seq_proj, pdb)
    return PdbProjection(
        structure_id=pdb.structure_id,
        chain_id=pdb.chain_id,
        sequence_projection=seq_proj,
        ic_pdb_residues=ic_pdb_residues,
    )
