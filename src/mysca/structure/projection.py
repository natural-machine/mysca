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
        "pdb_path": (
            "Absolute path to the PDB file this projection came from. "
            "Consumed by sca-pymol to re-load the structure for "
            "rendering without requiring a separate --pdb_dir flag."
        ),
        "sequence_projection": (
            "Underlying SequenceProjection from mysca.project, in "
            "raw-residue-index coordinates."
        ),
        "ic_pdb_residues": (
            "Per-IC list of PDB residue numbers "
            "(from PDBStructure.residue_ids) corresponding to "
            "sequence_projection.ic_residues."
        ),
    }

    def __init__(
        self,
        structure_id: str,
        chain_id: str,
        sequence_projection: SequenceProjection,
        ic_pdb_residues: list[list[int]],
        pdb_path: Optional[str] = None,
    ):
        self.structure_id = structure_id
        self.chain_id = chain_id
        self.pdb_path = pdb_path
        self.sequence_projection = sequence_projection
        self.ic_pdb_residues = ic_pdb_residues

    def to_dict(self) -> dict:
        return {
            "structure_id": self.structure_id,
            "chain_id": self.chain_id,
            "pdb_path": self.pdb_path,
            "sequence_projection": self.sequence_projection.to_dict(),
            "ic_pdb_residues": [list(xs) for xs in self.ic_pdb_residues],
        }


def project_groups_to_pdb(
    sequence_projection: SequenceProjection,
    pdb: PDBStructure,
) -> list[list[int]]:
    """Map per-IC raw-residue indices to PDB residue numbers.

    ``sequence_projection.ic_residues[i]`` is a list of 0-based
    indices into ``raw_sequence`` — the post-alignment, gapless
    subset of the input that landed in match columns of the reference
    MSA. Column-preserving aligners may drop input residues that
    don't fit; ``sequence_projection.input_residue_indices`` carries
    the surviving subset of the ORIGINAL input positions, so we can
    compose:

        raw_idx -> input_idx -> pdb.residue_ids[input_idx]

    Raises:
        ValueError: when any ``input_residue_indices`` entry is
            out of range for ``pdb.residue_ids`` — typically the
            signal that the projection and the PDB came from
            different primary sequences.
    """
    input_ids = sequence_projection.input_residue_indices
    n_pdb = len(pdb.residue_ids)
    if input_ids and max(input_ids) >= n_pdb:
        raise ValueError(
            f"input_residue_indices references position "
            f"{max(input_ids)} but PDB {pdb.structure_id}:{pdb.chain_id}"
            f" has only {n_pdb} residues. The SequenceProjection and "
            "the PDBStructure appear to come from different primary "
            "sequences."
        )
    out = []
    for members in sequence_projection.ic_residues:
        out.append([
            int(pdb.residue_ids[input_ids[int(r)]]) for r in members
        ])
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
        pdb_path=pdb.pdb_path,
    )
