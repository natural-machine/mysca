"""Project new (or in-sample) primary sequences onto an existing SCA result.

Given a ``PreprocessingResults`` + ``SCAResults`` pair, this module maps
each input sequence's raw residue indices onto the processed-MSA columns
(via alignment to the original MSA when needed) and reads off each
residue's IC-group membership.

The core operation composes three coordinate-system transforms already
formalized elsewhere in ``mysca``:

1. New-sequence residue ↔ original MSA column (provided by alignment —
   either in-sample lookup on ``msa_obj_orig`` or out-of-sample
   ``mafft --add --keeplength``).
2. Original MSA column ↔ processed MSA column (``retained_positions``).
3. Processed MSA column ↔ IC group membership (``groups``).
"""

import logging
import os
import tempfile
from typing import Iterable, Optional, Sequence

import numpy as np
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from mysca.helpers import get_rawseq_indices_of_msa
from mysca.results import PreprocessingResults, SCAResults, _format_info_table
from mysca.project.alignment import align_to_msa

logger = logging.getLogger("mysca.project.projection")


class SequenceProjection:
    """One projected sequence's per-IC residue membership."""

    FIELD_DESCRIPTIONS = {
        "seq_id": "Sequence ID from the input FASTA (or the MSA record).",
        "raw_sequence": (
            "Ungapped amino-acid sequence as provided (in-sample) or "
            "input (out-of-sample)."
        ),
        "aligned_sequence": (
            "Sequence aligned to the original MSA column layout "
            "(length = original alignment length)."
        ),
        "residue_by_processed_col": (
            "Int array of length L_proc. Entry j is the raw residue "
            "index (0-based in raw_sequence) at processed column j, or "
            "-1 if the sequence has a gap at the original MSA column "
            "corresponding to processed column j."
        ),
        "ic_residues": (
            "Per-IC list of raw residue indices in this sequence that "
            "fall in that IC's group (length = n_components)."
        ),
        "ic_loadings": (
            "Per-IC list of v_ica_normalized loadings parallel to "
            "ic_residues (same shape)."
        ),
        "ic_processed_cols": (
            "Per-IC list of processed-MSA column indices parallel to "
            "ic_residues. Useful for tracing a residue back to its "
            "group coordinate."
        ),
        "in_sample": (
            "True iff seq_id was found in msa_obj_orig and no new "
            "alignment was performed."
        ),
        "input_residue_indices": (
            "List of length len(raw_sequence): for each raw-residue "
            "position, the index into the ORIGINAL input sequence that "
            "survived column-preserving alignment. For in-sample "
            "records this is the identity ``list(range(len(raw)))``. "
            "For out-of-sample records where the aligner dropped some "
            "input residues (insert columns under hmmalign, "
            "`--keeplength` drops under mafft), this lets callers "
            "(notably project_groups_to_pdb) resolve raw residue "
            "indices back to positions in the full input sequence — "
            "and therefore to PDB residue numbers — without assuming "
            "all input residues survived."
        ),
        "up_score": (
            "1D float array of length n_components — this sequence's "
            "coordinates in the SCA IC sequence space (Uᵖ row), per "
            "Rivoire et al. (2016) Eqs. 14–15. None when the source "
            "SCAResults lacks the required eigendecomposition fields."
        ),
    }

    def __init__(
        self,
        seq_id: str,
        raw_sequence: str,
        aligned_sequence: str,
        residue_by_processed_col: np.ndarray,
        ic_residues: list[np.ndarray],
        ic_loadings: list[np.ndarray],
        ic_processed_cols: list[np.ndarray],
        in_sample: bool,
        input_residue_indices: list[int],
        up_score: Optional[np.ndarray] = None,
    ):
        self.seq_id = seq_id
        self.raw_sequence = raw_sequence
        self.aligned_sequence = aligned_sequence
        self.residue_by_processed_col = residue_by_processed_col
        self.ic_residues = ic_residues
        self.ic_loadings = ic_loadings
        self.ic_processed_cols = ic_processed_cols
        self.in_sample = in_sample
        self.input_residue_indices = list(input_residue_indices)
        self.up_score = up_score

    def info(self) -> str:
        return _format_info_table(
            f"SequenceProjection({self.seq_id!r})",
            self.FIELD_DESCRIPTIONS,
            lambda name: getattr(self, name, None),
        )

    def to_dict(self) -> dict:
        return {
            "seq_id": self.seq_id,
            "raw_sequence": self.raw_sequence,
            "aligned_sequence": self.aligned_sequence,
            "residue_by_processed_col": (
                self.residue_by_processed_col.tolist()
            ),
            "ic_residues": [
                arr.tolist() for arr in self.ic_residues
            ],
            "ic_loadings": [
                arr.tolist() for arr in self.ic_loadings
            ],
            "ic_processed_cols": [
                arr.tolist() for arr in self.ic_processed_cols
            ],
            "in_sample": bool(self.in_sample),
            "input_residue_indices": list(self.input_residue_indices),
            "up_score": (
                self.up_score.tolist() if self.up_score is not None
                else None
            ),
        }


class ProjectionResult:
    """Batch result from ``project_sequences``."""

    FIELD_DESCRIPTIONS = {
        "projections": (
            "List of SequenceProjection, one per input sequence, in "
            "input order."
        ),
        "args": (
            "Dict of arguments used for this projection run "
            "(aligner, source dirs, etc.)."
        ),
        "n_components": (
            "Number of ICs in the source SCA result (len(groups))."
        ),
        "n_retained_positions": (
            "L_proc from the source preprocessing run."
        ),
        "original_length": (
            "L_orig, length of the reference MSA that new sequences "
            "were aligned to."
        ),
    }

    def __init__(
        self,
        projections: list[SequenceProjection],
        args: dict,
        n_components: int,
        n_retained_positions: int,
        original_length: int,
    ):
        self.projections = projections
        self.args = args
        self.n_components = n_components
        self.n_retained_positions = n_retained_positions
        self.original_length = original_length

    def info(self) -> str:
        return _format_info_table(
            "ProjectionResult",
            self.FIELD_DESCRIPTIONS,
            lambda name: getattr(self, name, None),
        )

    def to_dict(self) -> dict:
        return {
            "args": self.args,
            "n_components": int(self.n_components),
            "n_retained_positions": int(self.n_retained_positions),
            "original_length": int(self.original_length),
            "projections": [p.to_dict() for p in self.projections],
        }

    def by_id(self, seq_id: str) -> SequenceProjection:
        for p in self.projections:
            if p.seq_id == seq_id:
                return p
        raise KeyError(f"seq_id {seq_id!r} not found in projections")

    @property
    def up_scores(self) -> Optional[np.ndarray]:
        """Stack of per-projection Uᵖ rows (M × n_components) or None
        when no projection has up_score populated."""
        rows = [p.up_score for p in self.projections]
        if all(r is None for r in rows):
            return None
        if any(r is None for r in rows):
            raise RuntimeError(
                "Inconsistent up_score state: some projections have "
                "Uᵖ rows and others don't. This should not happen — "
                "Up is computed in batch over all projections."
            )
        return np.stack(rows, axis=0)

    def to_dataframe(self):
        """Return a pandas DataFrame with seq_id, sequence, and Uᵖ columns.

        Columns: ``seq_id``, ``aligned_sequence``, ``raw_sequence``,
        ``in_sample``, ``up_0``, ``up_1``, ..., ``up_{n_components-1}``.

        Raises ImportError if pandas is not installed; raises
        RuntimeError if up_score has not been populated on the
        projections.
        """
        import pandas as pd
        if any(p.up_score is None for p in self.projections):
            raise RuntimeError(
                "to_dataframe requires up_score on every projection. "
                "Re-run projection on a SCAResults that has the full "
                "eigendecomposition (evecs_sca, evals_sca, w_ica)."
            )
        rows = []
        for p in self.projections:
            row = {
                "seq_id": p.seq_id,
                "aligned_sequence": p.aligned_sequence,
                "raw_sequence": p.raw_sequence,
                "in_sample": p.in_sample,
            }
            for k, v in enumerate(p.up_score):
                row[f"up_{k}"] = float(v)
            rows.append(row)
        return pd.DataFrame(rows)


# Gap chars recognized in an aligned sequence. `-` is the universal gap
# character; `.` is the Stockholm convention for insert-column gaps. Our
# pipeline never produces `.` internally (Biopython normalizes Stockholm
# `.` to `-` on parse, and `_hmmalign` strips `.` during insert-column
# filtering), but we treat both defensively so a hand-crafted MSA that
# leaks `.` through an out-of-band path doesn't silently get counted as
# a residue.
_GAP_CHARS = frozenset("-.")


def _gapless(aligned_seq: str) -> str:
    """Strip every recognized gap char from an aligned sequence."""
    return "".join(c for c in aligned_seq if c not in _GAP_CHARS)


def _residue_indices_for_aligned(aligned_seq: str) -> np.ndarray:
    """Raw-residue-index array for one aligned sequence (length = len(aligned_seq)).

    Entry j is the 0-based residue index at aligned column j, or -1 if
    that column is a gap (either ``-`` or ``.``). Mirrors
    ``get_rawseq_indices_of_msa`` for a single sequence.
    """
    out = np.full(len(aligned_seq), -1, dtype=int)
    idx = 0
    for j, ch in enumerate(aligned_seq):
        if ch in _GAP_CHARS:
            continue
        out[j] = idx
        idx += 1
    return out


def _aligned_to_xmsa(
    aligned_seqs: Sequence[str],
    retained_positions: np.ndarray,
    aa_list: Sequence[str],
) -> np.ndarray:
    """Convert column-aligned sequences to a one-hot tensor in
    processed-MSA coordinates.

    Output shape (M, L_proc, D), bool. Gap, missing, and non-canonical
    symbols at any processed column produce an all-zero row, matching
    the ``onehot_without_gap`` convention used by ``preprocess_msa``.
    """
    aa_to_col = {c: i for i, c in enumerate(aa_list)}
    D = len(aa_list)
    M = len(aligned_seqs)
    L_proc = len(retained_positions)
    out = np.zeros((M, L_proc, D), dtype=bool)
    for m, aligned in enumerate(aligned_seqs):
        for j, pos in enumerate(retained_positions):
            col = aa_to_col.get(aligned[pos])
            if col is not None:
                out[m, j, col] = True
    return out


def _recover_input_indices(input_raw: str, kept_raw: str) -> list[int]:
    """Given the original ungapped input sequence and the gapless
    aligned output (a subsequence of ``input_raw`` that preserves
    order and letter identity), return the indices in ``input_raw``
    that were retained.

    Column-preserving aligners (mafft --add --keeplength; hmmalign's
    insert-column strip) drop input residues *in place* rather than
    substituting them, so ``kept_raw`` is always an ordered
    subsequence of ``input_raw``. A linear scan recovers the mapping.
    """
    indices: list[int] = []
    h = 0
    for c in kept_raw:
        while h < len(input_raw) and input_raw[h] != c:
            h += 1
        if h >= len(input_raw):
            raise RuntimeError(
                f"Cannot recover input indices: "
                f"{kept_raw!r} is not a subsequence of {input_raw!r}. "
                "The aligner appears to have substituted residues "
                "rather than merely dropped them."
            )
        indices.append(h)
        h += 1
    return indices


def project_sequences(
    sequences_fpath: str,
    *,
    sca_result_dir: str,
    preproc_result_dir: str,
    aligner: str = "mafft_add",
    workdir: Optional[str] = None,
    aligner_kwargs: Optional[dict] = None,
) -> ProjectionResult:
    """Project one or more primary sequences onto an existing SCA result.

    Args:
        sequences_fpath: Path to an input FASTA. Each record is projected
            independently. Records whose ID matches an entry in the
            reference MSA ``msa_obj_orig`` are resolved in-sample (no
            external alignment performed).
        sca_result_dir: Directory written by ``sca-core``
            (read via ``SCAResults.load``).
        preproc_result_dir: Directory written by ``sca-preprocess``
            (read via ``PreprocessingResults.load``).
        aligner: Registered entry in ``project.alignment.ALIGNERS``.
            Default ``"mafft_add"``.
        workdir: Working directory for intermediate alignment files.
            Created if missing. When ``None``, a temp dir is used and
            cleaned up after the call.
        aligner_kwargs: Extra kwargs forwarded to the aligner callable
            (e.g. ``{"bin_path": "/usr/local/bin/mafft", "threads": 4}``).

    Returns:
        ProjectionResult with one ``SequenceProjection`` per input record.
    """
    prep = PreprocessingResults.load(preproc_result_dir)
    sca = SCAResults.load(sca_result_dir)

    if prep.msa_obj_orig is None:
        raise FileNotFoundError(
            f"Reference MSA not available in {preproc_result_dir}; "
            "msa_orig.fasta-aln is required for projection."
        )
    if sca.ic_positions is None:
        raise FileNotFoundError(
            f"No IC positions found in {sca_result_dir}; sca-core must "
            "produce ic_positions/ic_*_msaproc.npy for "
            "projection."
        )
    if sca.v_ica is None:
        raise FileNotFoundError(
            f"No v_ica_normalized.npy in {sca_result_dir}; required for "
            "IC loadings."
        )

    msa_obj_orig = prep.msa_obj_orig
    retained_positions = np.asarray(prep.retained_positions, dtype=int)
    groups = sca.ic_positions
    v_ica = sca.v_ica
    n_components = len(groups)
    L_orig = msa_obj_orig.get_alignment_length()
    L_proc = len(retained_positions)

    # Parse input sequences. We keep records in input order and decide
    # per-record whether to in-sample-short-circuit or queue for
    # out-of-sample alignment.
    with open(sequences_fpath) as f:
        input_records = list(SeqIO.parse(f, "fasta"))
    if not input_records:
        raise ValueError(
            f"No sequences parsed from {sequences_fpath}"
        )
    ids_in_msa = {rec.id: i for i, rec in enumerate(msa_obj_orig)}

    needs_align: list[SeqRecord] = []
    per_record_in_sample: list[bool] = []
    for rec in input_records:
        is_in_sample = rec.id in ids_in_msa
        per_record_in_sample.append(is_in_sample)
        if not is_in_sample:
            # Feed the raw (ungapped) sequence to the aligner, not the
            # input FASTA line, since the aligner expects ungapped input.
            raw = _gapless(str(rec.seq))
            needs_align.append(SeqRecord(Seq(raw), id=rec.id, description=""))

    # Out-of-sample alignment (if any).
    aligned_by_id: dict[str, str] = {}
    workdir_ctx = None
    try:
        if needs_align:
            if workdir is None:
                workdir_ctx = tempfile.TemporaryDirectory(
                    prefix="mysca_project_"
                )
                active_workdir = workdir_ctx.name
            else:
                os.makedirs(workdir, exist_ok=True)
                active_workdir = workdir
            new_fasta = os.path.join(active_workdir, "new_input.fasta")
            with open(new_fasta, "w") as fout:
                SeqIO.write(needs_align, fout, "fasta")
            info = align_to_msa(
                new_fasta, msa_obj_orig, active_workdir,
                method=aligner,
                **(aligner_kwargs or {}),
            )
            aligned_fpath = info["aligned_new_fpath"]
            with open(aligned_fpath) as f:
                for rec in SeqIO.parse(f, "fasta"):
                    aligned_by_id[rec.id] = str(rec.seq)

        projections = []
        for rec, is_in_sample in zip(input_records, per_record_in_sample):
            if is_in_sample:
                aligned_seq = str(
                    msa_obj_orig[ids_in_msa[rec.id]].seq
                )
                raw = _gapless(aligned_seq)
                # In-sample: raw came from the MSA row, so its indices
                # trivially match the input. There's nothing to recover.
                input_residue_indices = list(range(len(raw)))
            else:
                aligned_seq = aligned_by_id.get(rec.id)
                if aligned_seq is None:
                    raise RuntimeError(
                        f"Aligner did not return an aligned sequence "
                        f"for id {rec.id!r}"
                    )
                # Derive raw_sequence from the aligned output, not the
                # input FASTA. Column-preserving aligners (mafft --add
                # --keeplength; hmmalign --outformat afa followed by
                # insert-column stripping) drop residues that don't fit
                # the reference column structure. The retained indices
                # inside ic_residues count non-gap characters of
                # aligned_seq, so raw_sequence must do the same for
                # `raw_sequence[ic_residues[i]]` to dereference
                # correctly downstream.
                raw = _gapless(aligned_seq)
                # Recover which positions in the ORIGINAL input survived
                # alignment. raw is always an ordered subsequence of
                # the input (aligners drop residues; they don't
                # substitute). project_groups_to_pdb composes
                # input_residue_indices with pdb.residue_ids to handle
                # the common case where a PDB's primary sequence is
                # slightly longer than what fits the MSA columns.
                input_raw = _gapless(str(rec.seq))
                input_residue_indices = _recover_input_indices(input_raw, raw)
            if len(aligned_seq) != L_orig:
                raise RuntimeError(
                    f"Aligned sequence for {rec.id!r} has length "
                    f"{len(aligned_seq)}; expected {L_orig}"
                )
            resi_by_orig = _residue_indices_for_aligned(aligned_seq)
            resi_by_proc = resi_by_orig[retained_positions]

            ic_residues = []
            ic_loadings = []
            ic_processed_cols = []
            for i, g in enumerate(groups):
                g = np.asarray(g, dtype=int)
                if g.size == 0:
                    ic_residues.append(np.array([], dtype=int))
                    ic_loadings.append(np.array([], dtype=float))
                    ic_processed_cols.append(np.array([], dtype=int))
                    continue
                resi_at_group = resi_by_proc[g]
                keep = resi_at_group >= 0
                ic_residues.append(resi_at_group[keep])
                ic_loadings.append(v_ica[g[keep], i])
                ic_processed_cols.append(g[keep])

            projections.append(SequenceProjection(
                seq_id=rec.id,
                raw_sequence=raw,
                aligned_sequence=aligned_seq,
                residue_by_processed_col=resi_by_proc,
                ic_residues=ic_residues,
                ic_loadings=ic_loadings,
                ic_processed_cols=ic_processed_cols,
                in_sample=is_in_sample,
                input_residue_indices=input_residue_indices,
            ))

        # Sequence-space Uᵖ scores (Rivoire et al. Eqs. 14–15) for every
        # projected sequence. Best-effort: skip silently if the source
        # SCAResults lacks any of the eigendecomposition / ICA fields
        # (older sca-core runs predate eigendecomp persistence) or the
        # SymMap is unavailable.
        sym_map = getattr(prep, "sym_map", None)
        aa_list = getattr(sym_map, "aa_list", None)
        if aa_list is not None:
            try:
                xmsa_new = _aligned_to_xmsa(
                    [p.aligned_sequence for p in projections],
                    retained_positions, aa_list,
                )
                up_all = sca.project_sequences(xmsa_new)
            except (RuntimeError, AttributeError) as e:
                logger.warning(
                    "Skipping Uᵖ scores on projections: %s", e,
                )
            else:
                for p, row in zip(projections, up_all):
                    p.up_score = row
        else:
            logger.warning(
                "PreprocessingResults.sym_map has no aa_list; skipping "
                "Uᵖ scores on projections.",
            )
    finally:
        if workdir_ctx is not None:
            workdir_ctx.cleanup()

    args = {
        "sca_result_dir": os.path.abspath(sca_result_dir),
        "preproc_result_dir": os.path.abspath(preproc_result_dir),
        "sequences_fpath": os.path.abspath(sequences_fpath),
        "aligner": aligner,
    }
    return ProjectionResult(
        projections=projections,
        args=args,
        n_components=n_components,
        n_retained_positions=L_proc,
        original_length=L_orig,
    )
