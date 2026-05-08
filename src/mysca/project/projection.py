"""Project new (or in-sample) primary sequences onto an existing SCA result.

Given a ``PreprocessingResults`` + ``SCAResults`` pair, this module maps
each input sequence's raw residue indices onto the processed-MSA columns
(via alignment to the original MSA when needed) and reads off each
residue's IC-group membership.

The core operation composes three coordinate-system transforms already
formalized elsewhere in ``mysca``:

1. New-sequence residue ↔ original MSA column (provided by alignment —
   either in-sample lookup on ``msa_obj_loaded`` or out-of-sample
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
            "True iff seq_id was found in msa_obj_loaded and no new "
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
        "gap_fraction_per_ic": (
            "1D float array of length n_components. Entry i is the "
            "fraction of IC i's training-time support "
            "(``sca.ic_positions[i]``) that is gapped or non-canonical "
            "in this projection — i.e. positions that contribute zero "
            "mass to the Uᵖ math. 0.0 means full coverage; 1.0 means "
            "the projection has no informative residues at any of "
            "IC i's defining positions. None when SCAResults lacks "
            "ic_positions."
        ),
        "informative_positions_per_ic": (
            "1D int array of length n_components. Entry i is the count "
            "of positions in IC i's training-time support where this "
            "projection contributes a non-zero one-hot row to xsi "
            "(i.e. the position holds a canonical, non-gap residue). "
            "Companion to gap_fraction_per_ic; gap_fraction_per_ic[i] "
            "= 1 - informative_positions_per_ic[i] / "
            "len(sca.ic_positions[i]). None when SCAResults lacks "
            "ic_positions."
        ),
        "align_target": (
            "Which reference MSA the aligner used for this projection. "
            "'original' = the unfiltered MSA loaded from "
            "msa_orig.fasta-aln (length L_orig); 'processed' = the "
            "post-preprocessing MSA (length L_proc, sliced from "
            "msa_obj_loaded by retained_sequences and "
            "retained_positions). The choice affects "
            "len(aligned_sequence) and the derivation of "
            "residue_by_processed_col, but NOT the meaning of "
            "ic_residues / ic_loadings / ic_processed_cols (which are "
            "anchored in processed-MSA / raw-residue coordinates "
            "regardless)."
        ),
        "n_input_residues_dropped": (
            "Count of residues in the user-provided input sequence that "
            "did not survive alignment (i.e. did not land in any "
            "reference column). Equals len(input) - len(raw_sequence). "
            "Always 0 for in-sample records; can be nonzero for "
            "out-of-sample inputs longer than the reference, or under "
            "--align_target processed where the reference is narrower."
        ),
        "input_coverage_fraction": (
            "Fraction of the user-provided input that survived "
            "alignment, len(raw_sequence) / max(1, len(input)). 1.0 "
            "means every input residue landed in a reference column. "
            "Below 0.95 triggers a per-record WARNING in the "
            "projection log."
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
        gap_fraction_per_ic: Optional[np.ndarray] = None,
        informative_positions_per_ic: Optional[np.ndarray] = None,
        align_target: str = "original",
        n_input_residues_dropped: int = 0,
        input_coverage_fraction: float = 1.0,
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
        self.gap_fraction_per_ic = gap_fraction_per_ic
        self.informative_positions_per_ic = informative_positions_per_ic
        self.align_target = align_target
        self.n_input_residues_dropped = int(n_input_residues_dropped)
        self.input_coverage_fraction = float(input_coverage_fraction)

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
            "gap_fraction_per_ic": (
                self.gap_fraction_per_ic.tolist()
                if self.gap_fraction_per_ic is not None else None
            ),
            "informative_positions_per_ic": (
                self.informative_positions_per_ic.tolist()
                if self.informative_positions_per_ic is not None else None
            ),
            "align_target": self.align_target,
            "n_input_residues_dropped": int(self.n_input_residues_dropped),
            "input_coverage_fraction": float(self.input_coverage_fraction),
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
        "sequence_metadata": (
            "Optional pandas DataFrame with a `seq_id` column plus any "
            "user-supplied columns (e.g. taxid, kingdom, phylum). "
            "Loaded from `--seq_metadata <tsv>`; persisted as "
            "sequence_metadata.tsv next to projection.json and merged "
            "into seq_projections.tsv via left-join on seq_id when "
            "--save_dataframe is set. None when no metadata was "
            "supplied."
        ),
    }

    def __init__(
        self,
        projections: list[SequenceProjection],
        args: dict,
        n_components: int,
        n_retained_positions: int,
        original_length: int,
        sequence_metadata=None,
    ):
        self.projections = projections
        self.args = args
        self.n_components = n_components
        self.n_retained_positions = n_retained_positions
        self.original_length = original_length
        self.sequence_metadata = sequence_metadata

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
        """Return a pandas DataFrame with seq_id, sequence, Uᵖ, and
        per-IC quality-metric columns.

        Columns: ``seq_id``, ``aligned_sequence``, ``raw_sequence``,
        ``in_sample``, ``Up_0`` ... ``Up_{n_components-1}``,
        ``gap_frac_ic_0`` ... ``gap_frac_ic_{n_components-1}``,
        ``n_inform_ic_0`` ... ``n_inform_ic_{n_components-1}``. When
        ``sequence_metadata`` is set, its non-`seq_id` columns are
        left-joined onto the result on `seq_id`.

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
                row[f"Up_{k}"] = float(v)
            if p.gap_fraction_per_ic is not None:
                for k, v in enumerate(p.gap_fraction_per_ic):
                    row[f"gap_frac_ic_{k}"] = float(v)
            if p.informative_positions_per_ic is not None:
                for k, v in enumerate(p.informative_positions_per_ic):
                    row[f"n_inform_ic_{k}"] = int(v)
            rows.append(row)
        df = pd.DataFrame(rows)
        if self.sequence_metadata is not None:
            df = df.merge(self.sequence_metadata, on="seq_id", how="left")
        return df


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


def _per_ic_quality_metrics(xmsa, ic_positions):
    """Per-IC gap-fraction and informative-position counts for a batch
    of projected sequences.

    Parameters
    ----------
    xmsa : np.ndarray, shape (M, L_proc, D), bool/int
        One-hot-encoded sequences in processed-MSA coordinates.
        ``xmsa[m, j, :]`` is all-zero when sequence m has a gap or
        non-canonical symbol at processed column j (the convention
        established by ``_aligned_to_xmsa``).
    ic_positions : sequence of np.ndarray
        ``sca.ic_positions`` — for each IC, the processed-MSA column
        indices that define the IC's training-time support.

    Returns
    -------
    gap_fraction : np.ndarray, shape (M, n_components), float
        ``1 - n_inform / n_full`` per (sequence, IC). 0.0 when the IC
        has empty training support (defensive; should not occur).
    n_inform : np.ndarray, shape (M, n_components), int
        Count of positions in IC i's full training support where
        sequence m contributes a non-zero one-hot row.
    """
    informative = xmsa.any(axis=-1)
    ic_full_support = [np.asarray(g, dtype=int) for g in ic_positions]
    n_full = np.array([g.size for g in ic_full_support], dtype=float)
    M = informative.shape[0]
    n_comp = len(ic_full_support)
    n_inform = np.zeros((M, n_comp), dtype=int)
    for i, g in enumerate(ic_full_support):
        if g.size == 0:
            continue
        n_inform[:, i] = informative[:, g].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap_fraction = np.where(
            n_full > 0,
            1.0 - (n_inform / n_full),
            0.0,
        )
    return gap_fraction.astype(float), n_inform


def _aligned_to_xmsa(
    aligned_seqs: Sequence[str],
    retained_positions: np.ndarray,
    aa_list: Sequence[str],
    *,
    aligned_in_processed_coords: bool = False,
) -> np.ndarray:
    """Convert column-aligned sequences to a one-hot tensor in
    processed-MSA coordinates.

    Output shape (M, L_proc, D), bool. Gap, missing, and non-canonical
    symbols at any processed column produce an all-zero row, matching
    the ``onehot_without_gap`` convention used by ``preprocess_msa``.

    Parameters
    ----------
    aligned_seqs
        Column-aligned sequences. Length per sequence is L_orig when
        ``aligned_in_processed_coords=False`` (default), else L_proc.
    retained_positions
        Maps processed column j → original column ``retained_positions[j]``.
        Used to pick the right column from each aligned sequence under
        the default (original-coords) mode. Its length sets L_proc in
        either mode.
    aa_list
        Canonical alphabet (no gap).
    aligned_in_processed_coords
        Set to True when the aligned sequences are already in processed-
        MSA coordinates (e.g. ``--align_target processed``). In that
        case we index the j-th column directly instead of indirecting
        through ``retained_positions``.
    """
    aa_to_col = {c: i for i, c in enumerate(aa_list)}
    D = len(aa_list)
    M = len(aligned_seqs)
    L_proc = len(retained_positions)
    out = np.zeros((M, L_proc, D), dtype=bool)
    for m, aligned in enumerate(aligned_seqs):
        for j, pos in enumerate(retained_positions):
            char = aligned[j] if aligned_in_processed_coords else aligned[pos]
            col = aa_to_col.get(char)
            if col is not None:
                out[m, j, col] = True
    return out


ALIGN_TARGET_CHOICES = ("original", "processed")
PROCESSED_REFERENCE_FNAME = "processed_reference.fasta-aln"
COVERAGE_WARN_THRESHOLD = 0.95


def _build_processed_reference_msa(
        msa_obj_loaded: MultipleSeqAlignment,
        retained_sequences: np.ndarray,
        retained_positions: np.ndarray,
        outdir: Optional[str] = None,
) -> MultipleSeqAlignment:
    """Build a character-space ``MultipleSeqAlignment`` of length L_proc
    from the unfiltered loaded MSA, sliced by ``retained_sequences`` rows
    and ``retained_positions`` columns. Optionally write the materialized
    reference to ``<outdir>/processed_reference.fasta-aln`` for transparency.

    The aligners only require ``MultipleSeqAlignment.get_alignment_length()``
    + FASTA-serializable records, both of which work on the sliced view.
    """
    rows = list(msa_obj_loaded)
    sliced_records = []
    for src_idx in retained_sequences:
        rec = rows[int(src_idx)]
        seq_chars = str(rec.seq)
        sliced_seq = "".join(seq_chars[int(c)] for c in retained_positions)
        sliced_records.append(
            SeqRecord(Seq(sliced_seq), id=rec.id, description=rec.description)
        )
    sliced = MultipleSeqAlignment(sliced_records)
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, PROCESSED_REFERENCE_FNAME)
        SeqIO.write(sliced_records, out_path, "fasta")
        logger.info(
            "Materialized processed reference MSA (%d rows x %d cols) at %s",
            len(sliced_records), len(retained_positions), out_path,
        )
    return sliced


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
    align_target: str = "original",
    workdir: Optional[str] = None,
    aligner_kwargs: Optional[dict] = None,
    seq_metadata_path: Optional[str] = None,
) -> ProjectionResult:
    """Project one or more primary sequences onto an existing SCA result.

    Args:
        sequences_fpath: Path to an input FASTA. Each record is projected
            independently. Records whose ID matches an entry in the
            reference MSA ``msa_obj_loaded`` are resolved in-sample (no
            external alignment performed).
        sca_result_dir: Directory written by ``sca-core``
            (read via ``SCAResults.load``).
        preproc_result_dir: Directory written by ``sca-preprocess``
            (read via ``PreprocessingResults.load``).
        aligner: Registered entry in ``project.alignment.ALIGNERS``.
            Default ``"mafft_add"``.
        align_target: Which reference MSA to align against. ``"original"``
            (default) uses the unfiltered loaded MSA (length L_orig);
            ``"processed"`` uses the post-preprocessing MSA (length
            L_proc, sliced from msa_obj_loaded by retained_sequences and
            retained_positions). ``"processed"`` typically yields a
            denser, higher-quality reference but more aggressively clips
            input residues that exceed the reference column count;
            inspect ``input_coverage_fraction`` per record to detect.
        workdir: Working directory for intermediate alignment files.
            Created if missing. When ``None``, a temp dir is used and
            cleaned up after the call.
        aligner_kwargs: Extra kwargs forwarded to the aligner callable
            (e.g. ``{"bin_path": "/usr/local/bin/mafft", "threads": 4}``).
        seq_metadata_path: Optional path to a TSV with a `seq_id`
            column plus arbitrary user-supplied columns. Loaded into
            the returned ``ProjectionResult.sequence_metadata`` and
            merged into ``to_dataframe()`` output via left-join on
            ``seq_id``. Mirrors ``sca-core``'s ``--seq_metadata`` flag.

    Returns:
        ProjectionResult with one ``SequenceProjection`` per input record.
    """
    if align_target not in ALIGN_TARGET_CHOICES:
        raise ValueError(
            f"Unknown align_target {align_target!r}. "
            f"Choices: {ALIGN_TARGET_CHOICES}"
        )
    prep = PreprocessingResults.load(preproc_result_dir)
    sca = SCAResults.load(sca_result_dir)

    if prep.msa_obj_loaded is None:
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
    _missing_up = [
        name for name in ("phi_ia", "fia", "evecs_sca", "evals_sca", "w_ica")
        if getattr(sca, name, None) is None
    ]
    if _missing_up:
        raise FileNotFoundError(
            f"SCAResults at {sca_result_dir} is missing eigendecomposition / "
            f"ICA fields required for sequence-space (Uᵖ) projection: "
            f"{_missing_up}. Re-run sca-core (without --save_minimal, if it "
            "was used) to repopulate these fields, or pin the calling code "
            "to a release that does not require Uᵖ scoring."
        )

    msa_obj_loaded = prep.msa_obj_loaded
    retained_positions = np.asarray(prep.retained_positions, dtype=int)
    retained_sequences = np.asarray(prep.retained_sequences, dtype=int)
    groups = sca.ic_positions
    v_ica = sca.v_ica
    n_components = len(groups)
    L_orig = msa_obj_loaded.get_alignment_length()
    L_proc = len(retained_positions)
    expected_aligned_len = L_orig if align_target == "original" else L_proc

    # Parse input sequences. We keep records in input order and decide
    # per-record whether to in-sample-short-circuit or queue for
    # out-of-sample alignment.
    with open(sequences_fpath) as f:
        input_records = list(SeqIO.parse(f, "fasta"))
    if not input_records:
        raise ValueError(
            f"No sequences parsed from {sequences_fpath}"
        )
    ids_in_msa = {rec.id: i for i, rec in enumerate(msa_obj_loaded)}

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
            if align_target == "processed":
                active_ref = _build_processed_reference_msa(
                    msa_obj_loaded, retained_sequences, retained_positions,
                    outdir=active_workdir,
                )
            else:
                active_ref = msa_obj_loaded
            new_fasta = os.path.join(active_workdir, "new_input.fasta")
            with open(new_fasta, "w") as fout:
                SeqIO.write(needs_align, fout, "fasta")
            info = align_to_msa(
                new_fasta, active_ref, active_workdir,
                method=aligner,
                **(aligner_kwargs or {}),
            )
            aligned_fpath = info["aligned_new_fpath"]
            with open(aligned_fpath) as f:
                for rec in SeqIO.parse(f, "fasta"):
                    aligned_by_id[rec.id] = str(rec.seq)

        projections = []
        low_coverage_records: list[tuple[str, int, float]] = []
        for rec, is_in_sample in zip(input_records, per_record_in_sample):
            input_raw = _gapless(str(rec.seq))
            input_len = len(input_raw)
            if is_in_sample:
                row_seq = str(msa_obj_loaded[ids_in_msa[rec.id]].seq)
                if align_target == "processed":
                    # Slice the loaded row down to the L_proc processed
                    # columns. No aligner invocation; no residue clipping
                    # beyond what preprocessing already discarded.
                    aligned_seq = "".join(
                        row_seq[int(c)] for c in retained_positions
                    )
                else:
                    aligned_seq = row_seq
                raw = _gapless(aligned_seq)
                # In-sample: raw came from the MSA row, so its indices
                # trivially match the input order. Recover indices via
                # subsequence walk (handles align_target=processed where
                # the slice may drop some of the original input residues
                # that landed on truncated columns).
                input_residue_indices = _recover_input_indices(
                    input_raw, raw,
                ) if input_raw != raw else list(range(len(raw)))
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
                input_residue_indices = _recover_input_indices(input_raw, raw)
            if len(aligned_seq) != expected_aligned_len:
                raise RuntimeError(
                    f"Aligned sequence for {rec.id!r} has length "
                    f"{len(aligned_seq)}; expected {expected_aligned_len} "
                    f"(align_target={align_target!r})"
                )
            resi_by_orig = _residue_indices_for_aligned(aligned_seq)
            if align_target == "processed":
                # aligned_seq is already in processed-MSA column space;
                # no further slicing by retained_positions is needed.
                resi_by_proc = resi_by_orig
            else:
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

            n_dropped = max(0, input_len - len(raw))
            coverage = (len(raw) / input_len) if input_len > 0 else 1.0
            if n_dropped > 0 and coverage < COVERAGE_WARN_THRESHOLD:
                logger.warning(
                    "Low alignment coverage for %r: %d/%d input residues "
                    "retained (coverage=%.3f). Input is longer than the "
                    "reference, so the aligner clipped residues that "
                    "didn't fall in any reference column.",
                    rec.id, len(raw), input_len, coverage,
                )
                low_coverage_records.append((rec.id, n_dropped, coverage))
            elif n_dropped > 0:
                logger.info(
                    "Alignment dropped %d/%d residues from %r "
                    "(coverage=%.3f).",
                    n_dropped, input_len, rec.id, coverage,
                )

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
                align_target=align_target,
                n_input_residues_dropped=n_dropped,
                input_coverage_fraction=coverage,
            ))

        if low_coverage_records:
            logger.warning(
                "%d/%d projections fell below coverage=%.2f. Affected "
                "seq_ids (up to 10): %s",
                len(low_coverage_records), len(projections),
                COVERAGE_WARN_THRESHOLD,
                ", ".join(
                    f"{sid}({cov:.2f})"
                    for sid, _, cov in low_coverage_records[:10]
                ),
            )

        # Sequence-space Uᵖ scores (Rivoire et al. Eqs. 14–15) for every
        # projected sequence. The eigendecomposition / ICA precheck above
        # already guarantees `sca.project_sequences()` will succeed; the
        # only remaining recoverable failure mode is a missing SymMap on
        # the preprocessing bundle (legacy bundles), which we surface as
        # a warning + None up_score rather than failing the whole run.
        sym_map = getattr(prep, "sym_map", None)
        aa_list = getattr(sym_map, "aa_list", None)
        if aa_list is not None:
            xmsa_new = _aligned_to_xmsa(
                [p.aligned_sequence for p in projections],
                retained_positions, aa_list,
                aligned_in_processed_coords=(align_target == "processed"),
            )
            up_all = sca.project_sequences(xmsa_new)
            gap_frac_all, n_inform_all = _per_ic_quality_metrics(
                xmsa_new, groups,
            )
            for m, p in enumerate(projections):
                p.up_score = up_all[m]
                p.gap_fraction_per_ic = gap_frac_all[m]
                p.informative_positions_per_ic = n_inform_all[m]
        else:
            logger.warning(
                "PreprocessingResults.sym_map has no aa_list; skipping "
                "Uᵖ scores and per-IC quality metrics on projections.",
            )
    finally:
        if workdir_ctx is not None:
            workdir_ctx.cleanup()

    args = {
        "sca_result_dir": os.path.abspath(sca_result_dir),
        "preproc_result_dir": os.path.abspath(preproc_result_dir),
        "sequences_fpath": os.path.abspath(sequences_fpath),
        "aligner": aligner,
        "align_target": align_target,
    }

    sequence_metadata = None
    if seq_metadata_path is not None:
        import pandas as pd
        sequence_metadata = pd.read_csv(seq_metadata_path, sep="\t")
        if "seq_id" not in sequence_metadata.columns:
            raise ValueError(
                f"--seq_metadata TSV {seq_metadata_path!r} is missing "
                f"required 'seq_id' column. Got columns: "
                f"{list(sequence_metadata.columns)!r}"
            )
        args["seq_metadata_path"] = os.path.abspath(seq_metadata_path)
        logger.info(
            "Loaded sequence metadata: %d rows, %d cols (%s)",
            len(sequence_metadata),
            len(sequence_metadata.columns),
            ", ".join(sequence_metadata.columns),
        )

    return ProjectionResult(
        projections=projections,
        args=args,
        n_components=n_components,
        n_retained_positions=L_proc,
        original_length=L_orig,
        sequence_metadata=sequence_metadata,
    )
