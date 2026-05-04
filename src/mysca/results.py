"""Result container classes for SCA preprocessing and core analysis.

These classes provide attribute-based access to SCA results and encapsulate
save/load logic. The on-disk format uses standard numpy (.npz, .npy) and
JSON files, so results can be loaded without mysca installed.

On-disk formats are documented in each class's docstring.
"""

import os
import json
import numpy as np
from numpy.typing import NDArray
from scipy import sparse


# File name constants — shared with run_preprocessing.py and run_sca.py
PREPROCESSING_RESULTS_FNAME = "preprocessing_results.npz"
PREPROCESSING_SYMMAP_FNAME = "sym2int.json"
PREPROCESSING_ARGS_FNAME = "preprocessing_args.json"
PREPROCESSING_MSAORIG_FNAME = "msa_orig.fasta-aln"
PREPROCESSING_BINARY2D_FNAME = "msa_binary2d_sp.npz"
PREPROCESSING_FILTER_HISTORY_FNAME = "filter_history.json"

def _symmap_from_sym2int(sym2int):
    """Rebuild a SymMap from a flat {symbol: int} dict.

    The saved format carries no explicit gap marker, so gap identification
    relies on the codebase convention ``"-"``. If ``"-"`` is absent, the
    raw dict is returned unchanged so downstream ``sym_map[sym]`` lookups
    still work.
    """
    from mysca.mappings import SymMap
    if not isinstance(sym2int, dict):
        return sym2int
    gapsym = "-"
    if gapsym not in sym2int:
        return sym2int
    sym_list = sorted(sym2int.keys(), key=lambda s: sym2int[s])
    gap_value = sym_list.index(gapsym)
    aa_list = [s for s in sym_list if s != gapsym]
    return SymMap(
        "".join(aa_list), gapsym, gap_value=gap_value,
    )


def _filter_history_to_jsonable(filter_history):
    """Convert a filter_history list to JSON-serializable form.

    ``stat_values`` entries are numpy arrays (or None); numpy scalars may
    appear in ``n_sequences`` / ``n_filtered`` fields on some stages.
    """
    out = []
    for entry in filter_history:
        cleaned = {}
        for k, v in entry.items():
            if isinstance(v, np.ndarray):
                cleaned[k] = v.tolist()
            elif isinstance(v, np.generic):
                cleaned[k] = v.item()
            else:
                cleaned[k] = v
        out.append(cleaned)
    return out


def _filter_history_from_jsonable(raw):
    """Inverse of _filter_history_to_jsonable: restore numpy arrays for
    ``stat_values`` so downstream plotting code sees the same shape it
    produced in-process.
    """
    out = []
    for entry in raw:
        restored = dict(entry)
        sv = restored.get("stat_values")
        if sv is not None:
            restored["stat_values"] = np.asarray(sv)
        out.append(restored)
    return out


SCARUN_RESULTS_FNAME = "scarun_results.npz"
SCARUN_ARGS_FNAME = "scarun_args.json"
SCARUN_EIGENDECOMP_FNAME = "sca_eigendecomp.npz"
IC_RESIDUES_PER_SEQ_FNAME = "ic_residues_per_seq.npz"
IC_LOADINGS_PER_SEQ_FNAME = "ic_loadings_per_seq.npz"
EVALS_SHUFF_FNAME = "evals_shuff.npy"
SEQUENCE_METADATA_FNAME = "sequence_metadata.tsv"


def _describe_value(val):
    """Compact shape/type description for a results field.

    Returned as the ``value`` column in ``<container>.info()`` tables.
    """
    if val is None:
        return "(none)"
    if isinstance(val, np.ndarray):
        return f"ndarray{tuple(val.shape)} {val.dtype}"
    if isinstance(val, dict):
        return f"dict (n={len(val)})"
    if isinstance(val, list):
        return f"list (n={len(val)})"
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{type(val).__name__}={val}"
    return type(val).__name__


def _format_info_table(header, descriptions, value_fn):
    lines = [header, "-" * len(header)]
    name_w = max(len(n) for n in descriptions)
    val_w = max(
        (len(_describe_value(value_fn(n))) for n in descriptions), default=10,
    )
    val_w = max(val_w, 8)
    lines.append(
        f"{'field'.ljust(name_w)}  {'value'.ljust(val_w)}  description"
    )
    lines.append(f"{'-' * name_w}  {'-' * val_w}  {'-' * 11}")
    for name, desc in descriptions.items():
        lines.append(
            f"{name.ljust(name_w)}  {_describe_value(value_fn(name)).ljust(val_w)}  {desc}"
        )
    return "\n".join(lines)


class PreprocessingResults:
    """Container for SCA preprocessing results.

    Provides named attribute access to preprocessing outputs and handles
    persistence to/from a directory of npz/json files.

    Per-field descriptions are available at class level as
    ``PreprocessingResults.FIELD_DESCRIPTIONS`` and can be rendered for
    any instance via ``results.info()``.

    On-disk format (usable without mysca):
        preprocessing_results.npz
            msa                   : int array (M x L), processed MSA
            retained_sequences    : int array, indices into original MSA rows
            retained_positions    : int array, indices into original MSA columns
            retained_sequence_ids : str array, IDs of retained sequences
            sequence_weights      : float array (M,)
            fi0_pretruncation     : float array, gap freq before truncation
        preprocessing_args.json   : dict of preprocessing parameters
        sym2int.json              : dict mapping symbols to integers
        msa_binary2d_sp.npz       : sparse CSR (M x 20L), one-hot MSA
        msa_orig.fasta-aln        : original MSA in FASTA format (optional)
        filter_history.json       : list of per-stage filter diagnostics
                                    (retained counts, threshold used, and
                                    the stat distribution that fed the
                                    filter). Needed to replay the
                                    filter-diagnostic plots.
    """

    FIELD_DESCRIPTIONS = {
        "msa": (
            "Processed MSA, int array (M x L). Each entry is the SymMap "
            "integer for the residue at that (retained_sequence, "
            "retained_position)."
        ),
        "msa_binary3d": (
            "One-hot MSA, array (M x L x D) with D=len(alphabet). Gap "
            "symbol is all-zero across the D axis."
        ),
        "retained_sequences": (
            "Indices of retained sequences in the original MSA (1D int "
            "array of length M)."
        ),
        "retained_positions": (
            "Indices of retained positions in the original MSA (1D int "
            "array of length L). Bridges processed→original coordinates: "
            "original_col = retained_positions[processed_col]."
        ),
        "retained_sequence_ids": (
            "IDs (strings) of retained sequences, aligned to "
            "retained_sequences. Preserves input order."
        ),
        "sequence_weights": (
            "Sampling weights per retained sequence (1D float array of "
            "length M). Used throughout SCA for weighted frequencies."
        ),
        "fi0_pretruncation": (
            "Per-position gap frequency before the first truncation step "
            "(1D float array, length = original alignment length)."
        ),
        "args": (
            "Dict of CLI arguments used to produce this result, for "
            "reproducibility. Includes reference_id, thresholds, etc."
        ),
        "sym_map": (
            "SymMap instance: the alphabet and gap symbol used. Integer "
            "encoding of every symbol is stable across save/load."
        ),
        "msa_obj_loaded": (
            "Original MSA as a Biopython MultipleSeqAlignment (before any "
            "filtering). Needed for raw-residue coordinate mapping."
        ),
        "filter_history": (
            "List of per-stage filter diagnostics (counts, threshold, and "
            "the statistic distribution that fed the filter). Consumed "
            "by sca-plots for filter-diagnostic figures. May be None for "
            "legacy bundles that pre-date the filter-history persistence."
        ),
    }

    def info(self):
        """Return a human-readable summary of field descriptions and
        current values on this instance. ``print(results.info())`` to
        display. Field metadata lives on ``FIELD_DESCRIPTIONS``.
        """
        return _format_info_table(
            "PreprocessingResults",
            self.FIELD_DESCRIPTIONS,
            lambda name: getattr(self, name, None),
        )

    def __init__(
        self,
        msa,
        msa_binary3d,
        retained_sequences,
        retained_positions,
        retained_sequence_ids,
        sequence_weights,
        fi0_pretruncation,
        args,
        sym_map=None,
        msa_obj_loaded=None,
        filter_history=None,
    ):
        self.msa = msa
        self.msa_binary3d = msa_binary3d
        self.retained_sequences = retained_sequences
        self.retained_positions = retained_positions
        self.retained_sequence_ids = retained_sequence_ids
        self.sequence_weights = sequence_weights
        self.fi0_pretruncation = fi0_pretruncation
        self.args = args
        self.sym_map = sym_map
        self.msa_obj_loaded = msa_obj_loaded
        self.filter_history = filter_history

    @property
    def n_sequences(self):
        return self.msa.shape[0]

    @property
    def n_positions(self):
        return self.msa.shape[1]

    @classmethod
    def from_preprocess_output(cls, msa, results_dict, sym_map=None,
                               msa_obj_loaded=None):
        """Construct from the (msa, results_dict) returned by preprocess_msa().
        """
        return cls(
            msa=msa,
            msa_binary3d=results_dict["msa_binary3d"],
            retained_sequences=results_dict["retained_sequences"],
            retained_positions=results_dict["retained_positions"],
            retained_sequence_ids=results_dict["retained_sequence_ids"],
            sequence_weights=results_dict["sequence_weights"],
            fi0_pretruncation=results_dict["fi0_pretruncation"],
            args=results_dict["args"],
            sym_map=sym_map,
            msa_obj_loaded=msa_obj_loaded,
            filter_history=results_dict.get("filter_history"),
        )

    def save(self, outdir):
        """Save all results to the given directory."""
        os.makedirs(outdir, exist_ok=True)

        np.savez(
            os.path.join(outdir, PREPROCESSING_RESULTS_FNAME),
            msa=self.msa,
            retained_sequences=self.retained_sequences,
            retained_positions=self.retained_positions,
            retained_sequence_ids=self.retained_sequence_ids,
            sequence_weights=self.sequence_weights,
            fi0_pretruncation=self.fi0_pretruncation,
        )

        with open(os.path.join(outdir, PREPROCESSING_ARGS_FNAME), "w") as f:
            json.dump(self.args, f)

        if self.sym_map is not None:
            sym2int = (self.sym_map.sym2int
                       if hasattr(self.sym_map, "sym2int")
                       else self.sym_map)
            with open(os.path.join(outdir, PREPROCESSING_SYMMAP_FNAME), "w") as f:
                json.dump(sym2int, f)

        if self.msa_binary3d is not None:
            sparse.save_npz(
                os.path.join(outdir, PREPROCESSING_BINARY2D_FNAME),
                sparse.csr_matrix(
                    self.msa_binary3d.reshape(
                        [self.msa_binary3d.shape[0], -1]
                    )
                ),
            )

        if self.msa_obj_loaded is not None:
            from Bio import AlignIO
            AlignIO.write(
                self.msa_obj_loaded,
                os.path.join(outdir, PREPROCESSING_MSAORIG_FNAME),
                format="fasta",
            )

        if self.filter_history is not None:
            with open(
                os.path.join(outdir, PREPROCESSING_FILTER_HISTORY_FNAME), "w"
            ) as f:
                json.dump(_filter_history_to_jsonable(self.filter_history), f)

    @classmethod
    def load(cls, dirpath):
        """Load results from a directory previously created by save()."""
        data = np.load(
            os.path.join(dirpath, PREPROCESSING_RESULTS_FNAME),
            allow_pickle=True,
        )
        msa = data["msa"]
        retained_sequences = data["retained_sequences"]
        retained_positions = data["retained_positions"]
        retained_sequence_ids = data["retained_sequence_ids"]
        sequence_weights = data["sequence_weights"]
        fi0_pretruncation = data["fi0_pretruncation"]

        # Load args
        args_path = os.path.join(dirpath, PREPROCESSING_ARGS_FNAME)
        args = None
        if os.path.isfile(args_path):
            with open(args_path, "r") as f:
                args = json.load(f)

        # Load sym_map and reconstruct a SymMap when possible. The on-disk
        # representation is a flat symbol->int dict; the gap symbol is
        # identified by the convention "-". Fall back to returning the raw
        # dict if "-" is not present (the only downstream use that matters
        # is indexed lookup via sym_map[aa], which works for both types).
        sym_map = None
        symmap_path = os.path.join(dirpath, PREPROCESSING_SYMMAP_FNAME)
        if os.path.isfile(symmap_path):
            with open(symmap_path, "r") as f:
                sym2int = json.load(f)
            sym_map = _symmap_from_sym2int(sym2int)

        # Load sparse binary MSA -> dense 3d
        msa_binary3d = None
        binary_path = os.path.join(dirpath, PREPROCESSING_BINARY2D_FNAME)
        if os.path.isfile(binary_path):
            msa_binary3d = sparse.load_npz(binary_path).toarray().reshape(
                [len(retained_sequences), len(retained_positions), -1]
            )

        # Try to load original MSA (requires Biopython)
        msa_obj_loaded = None
        orig_path = os.path.join(dirpath, PREPROCESSING_MSAORIG_FNAME)
        if os.path.isfile(orig_path):
            try:
                from Bio import AlignIO
                msa_obj_loaded = AlignIO.read(orig_path, "fasta")
            except ImportError:
                pass

        # Filter history (needed to replay filter diagnostic plots)
        filter_history = None
        fh_path = os.path.join(dirpath, PREPROCESSING_FILTER_HISTORY_FNAME)
        if os.path.isfile(fh_path):
            with open(fh_path, "r") as f:
                filter_history = _filter_history_from_jsonable(json.load(f))

        return cls(
            msa=msa,
            msa_binary3d=msa_binary3d,
            retained_sequences=retained_sequences,
            retained_positions=retained_positions,
            retained_sequence_ids=retained_sequence_ids,
            sequence_weights=sequence_weights,
            fi0_pretruncation=fi0_pretruncation,
            args=args,
            sym_map=sym_map,
            msa_obj_loaded=msa_obj_loaded,
            filter_history=filter_history,
        )


class SCAResults:
    """Container for SCA core + analysis results.

    Per-field descriptions are available at class level as
    ``SCAResults.FIELD_DESCRIPTIONS`` and can be rendered for any instance
    via ``results.info()``.

    Sequence-space projection:
        :meth:`project_sequences` maps one-hot sequences (M x L_proc x D)
        onto the IC coordinate system (Uᵖ, M x n_components) on demand,
        following Rivoire et al. (2016) Eqs. 14–15 with per-position P̃
        normalization. No new on-disk fields are required — the operands
        (``phi_ia``, ``fia``, ``significant_evecs_sca``,
        ``significant_evals_sca``, ``w_ica``) are already persisted below.

    On-disk format (usable without mysca):
        scarun_results.npz
            Dia          : float array (L x D), conservation per position+aa
            conservation : float array (L,), aggregated Di
            sca_matrix   : float array (L x L), weighted SCA covariance Cij
            phi_ia       : float array (L x D)
            fi0          : float array (L,)
            fia          : float array (L x D)
            (optional) Cijab_raw, fijab : large pairwise arrays
        sca_eigendecomp.npz
            evals_sca, evecs_sca, significant_evals_sca, significant_evecs_sca
        scarun_args.json              : dict of SCA parameters
        ic_residues_per_seq.npz       : per-target IC residues in raw-sequence
                                        coordinates, keyed `ic_<i>_<seqid>`
        ic_loadings_per_seq.npz       : per-residue IC loadings parallel to
                                        ic_residues_per_seq, same key format
        ic_positions/
            ic_<i>_msaproc.npy        : high-load processed-MSA cols of IC i
            ic_<i>_msaorig.npy        : same positions in original-MSA cols
                                        (recovered via retained_positions)
            ic_<i>_loadings.npy       : the IC's loadings at those positions
        sca_results/
            kstar.txt                 : number of significant eigenvalues used
            kstar_identified.txt      : number identified by bootstrap
            n_components.txt          : number of ICs computed (>= kstar)
            eigenvalue_cutoff.txt     : bootstrap cutoff value
            v_ica_normalized.npy      : normalized ICA components
            w_ica.npy                 : ICA unmixing matrix
            t_dists_info.json         : t-distribution fit parameters
            evals_shuff.npy           : bootstrap eigenvalues
            sca_matrix_sector_subset.npy

        Conditional outputs:
        seq_projections.tsv (only with --save_dataframe)
            Per-retained-sequence Uᵖ table. Columns: `seq_id`,
            `aligned_sequence`, `Up_0` ... `Up_{n_components-1}`, plus
            any columns merged in from `sequence_metadata`.
        sequence_metadata.tsv (only with --seq_metadata)
            Verbatim copy of the user-supplied per-sequence metadata
            TSV. Carries a `seq_id` column plus arbitrary user columns
            (e.g. taxid, kingdom, phylum); merged into
            `seq_projections.tsv` via left-join on `seq_id` when
            `--save_dataframe` is also set.
        images/ (only with --plot, the default)
            Diagnostic plots (conservation, SCA matrix, spectrum, IC
            scatters, sector subset, sequence-projection scatter, etc.).
    """

    FIELD_DESCRIPTIONS = {
        # Core SCA
        "Dia": (
            "Per-(position, amino acid) conservation contribution D_i^a "
            "(float array L x D). Sums across the D axis give `conservation`."
        ),
        "conservation": (
            "Position-wise relative entropy D_i (float array length L). "
            "`conservation[i]` = sum_a D_i^a."
        ),
        "sca_matrix": (
            "Weighted SCA covariance matrix φ_i^a ⊗ φ_j^b collapsed "
            "to a (L x L) float array (what is eigendecomposed)."
        ),
        "phi_ia": (
            "Per-(position, amino acid) weighting applied to covariances "
            "(float array L x D)."
        ),
        "fi0": (
            "Gap frequency at each position (float array length L), "
            "weighted by `sequence_weights`."
        ),
        "fia": (
            "Per-(position, amino acid) frequency (float array L x D), "
            "weighted by `sequence_weights`."
        ),
        # Optional large arrays
        "Cijab_raw": (
            "Unreduced covariance tensor (L x L x D x D). Only populated "
            "when `--save_all` was passed to sca-core."
        ),
        "fijab": (
            "Unreduced joint-frequency tensor (L x L x D x D). Only "
            "populated when `--save_all` was passed."
        ),
        "Cij_raw": (
            "Reduced covariance matrix (L x L) prior to the phi_ia-weighted "
            "step that yields `sca_matrix`. Persisted to disk so sca-plots "
            "can replay the covariance-matrix plot without rerunning core."
        ),
        # Eigendecomposition
        "evals_sca": (
            "All eigenvalues of `sca_matrix`, sorted descending (float "
            "array length L)."
        ),
        "evecs_sca": (
            "All eigenvectors of `sca_matrix`, columns matching `evals_sca` "
            "(float array L x L)."
        ),
        "significant_evals_sca": (
            "Top-kstar eigenvalues (float array length kstar)."
        ),
        "significant_evecs_sca": (
            "Top-kstar eigenvectors (float array L x kstar)."
        ),
        # Bootstrap / significance
        "kstar": (
            "Number of significant eigenvalues actually used (after any "
            "--kstar override and the kstar>=1 fallback)."
        ),
        "kstar_identified": (
            "Raw kstar from the bootstrap null (before any --kstar override)."
        ),
        "n_components": (
            "Number of ICs computed by ICA; always >= kstar. Controlled by "
            "--n_components (int or 'all')."
        ),
        "cutoff": (
            "Eigenvalue significance cutoff derived from the bootstrap "
            "null (float scalar)."
        ),
        "evals_shuff": (
            "Bootstrap null eigenvalue spectrum (float array N_BOOT x L)."
        ),
        # ICA
        "v_ica": (
            "Normalized IC components (float array L x n_components). "
            "Columns are the independent components of `evecs_sca`."
        ),
        "w_ica": (
            "ICA unmixing matrix (float array n_components x n_components)."
        ),
        # IC positions (high-load, per-IC)
        "ic_positions": (
            "Per-IC list of high-load position indices (processed-MSA "
            "coordinates) that cleared the per-IC t-distribution cutoff. "
            "Length n_components; each entry is a 1D int array."
        ),
        "group_scores": (
            "Per-IC list of IC loadings for the positions in "
            "`ic_positions[i]` (same shape structure as `ic_positions`)."
        ),
        "t_dists_info": (
            "List of per-IC t-distribution fits (df, loc, scale, cutoff) "
            "used to nominate positions. Length n_components."
        ),
        "ic_residues_per_seq": (
            "Per-target IC residues keyed `ic_{i}_{seqid}` in "
            "raw-sequence residue coordinates. Only populated for the "
            "top-kstar ICs and for sequences selected by `--sectors_for`."
        ),
        "ic_loadings_per_seq": (
            "IC loadings parallel to `ic_residues_per_seq`, same "
            "`ic_{i}_{seqid}` key format. The j-th value of "
            "`ic_loadings_per_seq[ic_{i}_{seqid}]` is the IC i loading "
            "at the residue `ic_residues_per_seq[ic_{i}_{seqid}][j]`."
        ),
        "sca_matrix_sector_subset": (
            "Submatrix of `sca_matrix` restricted to all group positions "
            "(concatenated). Shape (sum_i len(groups[i])) squared."
        ),
        "sequence_metadata": (
            "Optional pandas DataFrame with a `seq_id` column plus any "
            "number of user-supplied columns (e.g. taxid, kingdom, "
            "phylum). Loaded from `--seq_metadata <tsv>` at sca-core "
            "invocation; persisted as `sequence_metadata.tsv`. None "
            "when no metadata was supplied."
        ),
        # Args
        "args": (
            "Dict of CLI arguments used for this run (regularization, "
            "kstar, n_components, pstar, assignment, seed, n_boot, "
            "n_logged_comps, plot, freq_method, accelerator, precision, "
            "bootstrap_chunk)."
        ),
    }

    def info(self):
        """Return a human-readable summary of field descriptions and
        current values on this instance. ``print(results.info())`` to
        display. Field metadata lives on ``FIELD_DESCRIPTIONS``.
        """
        return _format_info_table(
            "SCAResults",
            self.FIELD_DESCRIPTIONS,
            lambda name: getattr(self, name, None),
        )

    def __init__(
        self,
        # Core SCA
        Dia=None,
        conservation=None,
        sca_matrix=None,
        phi_ia=None,
        fi0=None,
        fia=None,
        # Optional large arrays
        Cijab_raw=None,
        fijab=None,
        Cij_raw=None,
        # Eigendecomposition
        evals_sca=None,
        evecs_sca=None,
        significant_evals_sca=None,
        significant_evecs_sca=None,
        # Bootstrap / significance
        kstar=None,
        kstar_identified=None,
        n_components=None,
        cutoff=None,
        evals_shuff=None,
        # ICA
        v_ica=None,
        w_ica=None,
        # IC positions (high-load, per-IC)
        ic_positions=None,
        group_scores=None,
        t_dists_info=None,
        ic_residues_per_seq=None,
        ic_loadings_per_seq=None,
        sca_matrix_sector_subset=None,
        sequence_metadata=None,
        # Args
        args=None,
    ):
        self.Dia = Dia
        self.conservation = conservation
        self.sca_matrix = sca_matrix
        self.phi_ia = phi_ia
        self.fi0 = fi0
        self.fia = fia
        self.Cijab_raw = Cijab_raw
        self.fijab = fijab
        self.Cij_raw = Cij_raw
        self.evals_sca = evals_sca
        self.evecs_sca = evecs_sca
        self.significant_evals_sca = significant_evals_sca
        self.significant_evecs_sca = significant_evecs_sca
        self.kstar = kstar
        self.kstar_identified = kstar_identified
        self.n_components = n_components
        self.cutoff = cutoff
        self.evals_shuff = evals_shuff
        self.v_ica = v_ica
        self.w_ica = w_ica
        self.ic_positions = ic_positions
        self.group_scores = group_scores
        self.t_dists_info = t_dists_info
        self.ic_residues_per_seq = ic_residues_per_seq
        self.ic_loadings_per_seq = ic_loadings_per_seq
        self.sca_matrix_sector_subset = sca_matrix_sector_subset
        self.sequence_metadata = sequence_metadata
        self.args = args

    @property
    def n_ic_positions(self):
        if self.ic_positions is None:
            return None
        return len(self.ic_positions)

    @property
    def n_positions(self):
        if self.conservation is not None:
            return len(self.conservation)
        return None

    def _require_fields(self, *names):
        missing = [n for n in names if getattr(self, n, None) is None]
        if missing:
            raise RuntimeError(
                f"{type(self).__name__} is missing required fields "
                f"{missing} (got None). Was this loaded from a partial run?"
            )

    def project_sequences(self, xmsa):
        """Project one-hot sequences onto IC coordinates (Uᵖ).

        Implements the Rivoire et al. (2016) sequence-position mapping
        (Eqs. 14, 15) with per-position P̃ normalization (Box 1, p. 6):

            P̃[i,a] = φ_ia · f_ia / sqrt(Σ_b (φ_ib · f_ib)²)
            xsi[m,i] = Σ_a P̃[i,a] · xmsa[m,i,a]
            Ũ        = xsi · V̌ · diag(λ̃^-½)
            Uᵖ       = Ũ · Wᵀ

        Requires `phi_ia`, `fia`, `evecs_sca`, `evals_sca`, and `w_ica`
        to be present on this instance. Uses the first ``w_ica.shape[0]``
        eigenmodes (== `n_components`, the dimension ICA was run on),
        which may exceed `kstar` when sca-core was invoked with
        ``--n_components`` greater than the bootstrap-significant count.

        Parameters
        ----------
        xmsa : array-like, shape (M, L_proc, D)
            One-hot encoded sequences in processed-MSA coordinates. Gaps
            are all-zero rows (matches `PreprocessingResults.msa_binary3d`
            and the output of `mysca.preprocess.onehot_without_gap`).

        Returns
        -------
        up : np.ndarray, shape (M, n_components), float64
            Sequence-space IC scores Uᵖ.
        """
        self._require_fields(
            "phi_ia", "fia", "evecs_sca", "evals_sca", "w_ica",
        )
        L, D = self.fia.shape
        xmsa = np.asarray(xmsa)
        if xmsa.ndim != 3 or xmsa.shape[1:] != (L, D):
            raise ValueError(
                f"xmsa has shape {xmsa.shape}; expected (M, {L}, {D}) to "
                f"match this SCAResults' fia dimensions."
            )
        n_comp = self.w_ica.shape[0]
        evecs = self.evecs_sca[:, :n_comp]
        evals = self.evals_sca[:n_comp]
        pf = self.phi_ia * self.fia
        denom = np.sqrt(np.sum(pf * pf, axis=-1, keepdims=True))
        pia = pf / denom
        xf = xmsa.astype(np.float64, copy=False)
        xsi = np.einsum("ia,mia->mi", pia, xf)
        utilde = (xsi @ evecs) / np.sqrt(evals)
        return utilde @ self.w_ica.T

    def to_dataframe(self, prep):
        """In-sample sequence DataFrame keyed by retained_sequence_ids.

        Columns: ``seq_id``, ``aligned_sequence`` (length L_orig, the
        original-MSA-coordinate sequence from msa_obj_loaded),
        ``Up_0``, ``Up_1``, ..., ``Up_{n_components-1}`` (Uᵖ row from
        ``project_sequences(prep.msa_binary3d)``). When
        ``sequence_metadata`` is populated on this instance, its
        non-``seq_id`` columns are merged in via left-join on
        ``seq_id``.

        Requires pandas installed and a PreprocessingResults instance
        with ``msa_obj_loaded``, ``retained_sequence_ids``,
        ``retained_sequences``, and ``msa_binary3d`` populated.
        """
        import pandas as pd
        if prep.msa_obj_loaded is None:
            raise RuntimeError(
                "to_dataframe requires PreprocessingResults.msa_obj_loaded; "
                "the source preprocessing run did not persist msa_orig.fasta-aln."
            )
        if prep.msa_binary3d is None:
            raise RuntimeError(
                "to_dataframe requires PreprocessingResults.msa_binary3d "
                "(persisted as msa_binary2d_sp.npz)."
            )
        up = self.project_sequences(prep.msa_binary3d)
        ids_to_row = {rec.id: i for i, rec in enumerate(prep.msa_obj_loaded)}
        rows = []
        for i, sid in enumerate(prep.retained_sequence_ids):
            rec = prep.msa_obj_loaded[ids_to_row[sid]]
            row = {
                "seq_id": sid,
                "aligned_sequence": str(rec.seq),
            }
            for k, v in enumerate(up[i]):
                row[f"Up_{k}"] = float(v)
            rows.append(row)
        df = pd.DataFrame(rows)
        if self.sequence_metadata is not None:
            df = df.merge(self.sequence_metadata, on="seq_id", how="left")
        return df

    @classmethod
    def from_core_output(cls, sca_results_dict, args=None):
        """Construct from the dict returned by core.run_sca().

        Maps core key names to the canonical attribute names used on disk:
            Di -> conservation, Cij_corr -> sca_matrix, etc.
        """
        return cls(
            Dia=sca_results_dict.get("Dia"),
            conservation=sca_results_dict.get("Di"),
            sca_matrix=sca_results_dict.get("Cij_corr"),
            phi_ia=sca_results_dict.get("phi_ia"),
            fi0=sca_results_dict.get("fi0"),
            fia=sca_results_dict.get("fia"),
            Cijab_raw=sca_results_dict.get("Cijab_raw"),
            fijab=sca_results_dict.get("fijab"),
            Cij_raw=sca_results_dict.get("Cij_raw"),
            args=args,
        )

    def save(self, outdir, save_all=False, *, retained_positions=None):
        """Save results to the given directory.

        Args:
            outdir: Output directory path.
            save_all: If True, include large arrays (Cijab_raw, fijab).
            retained_positions: Optional 1D int array mapping
                processed-MSA col -> original-MSA col. When supplied,
                the per-IC ic_positions/ic_<i>_msaorig.npy sibling is
                written alongside ic_<i>_msaproc.npy.
        """
        scadir = os.path.join(outdir, "sca_results")
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(scadir, exist_ok=True)

        # Core SCA results
        if self.conservation is not None:
            tosave = {
                "Dia": self.Dia,
                "conservation": self.conservation,
                "sca_matrix": self.sca_matrix,
                "phi_ia": self.phi_ia,
                "fi0": self.fi0,
                "fia": self.fia,
            }
            # Cij_raw is small (L x L) and needed for sca-plots replay of
            # the covariance-matrix figure, so it rides along here rather
            # than being gated on --save_all.
            if self.Cij_raw is not None:
                tosave["Cij_raw"] = self.Cij_raw
            if save_all:
                if self.Cijab_raw is not None:
                    tosave["Cijab_raw"] = self.Cijab_raw
                if self.fijab is not None:
                    tosave["fijab"] = self.fijab
            np.savez_compressed(
                os.path.join(outdir, SCARUN_RESULTS_FNAME),
                **tosave,
            )

        # Args
        if self.args is not None:
            with open(os.path.join(outdir, SCARUN_ARGS_FNAME), "w") as f:
                json.dump(self.args, f)

        # Eigendecomposition
        if self.evals_sca is not None:
            np.savez_compressed(
                os.path.join(outdir, SCARUN_EIGENDECOMP_FNAME),
                evals_sca=self.evals_sca,
                evecs_sca=self.evecs_sca,
                significant_evals_sca=self.significant_evals_sca,
                significant_evecs_sca=self.significant_evecs_sca,
            )

        # Bootstrap / significance scalars and arrays
        if self.kstar_identified is not None:
            np.savetxt(
                os.path.join(scadir, "kstar_identified.txt"),
                [self.kstar_identified], fmt="%d",
            )
        if self.kstar is not None:
            np.savetxt(
                os.path.join(scadir, "kstar.txt"),
                [self.kstar], fmt="%d",
            )
        if self.n_components is not None:
            np.savetxt(
                os.path.join(scadir, "n_components.txt"),
                [self.n_components], fmt="%d",
            )
        if self.cutoff is not None:
            np.savetxt(
                os.path.join(scadir, "eigenvalue_cutoff.txt"),
                [self.cutoff],
            )
        if self.evals_sca is not None:
            np.save(os.path.join(scadir, "all_evals_sca.npy"), self.evals_sca)
            np.save(os.path.join(scadir, "all_evecs_sca.npy"), self.evecs_sca)
        if self.significant_evals_sca is not None:
            np.save(
                os.path.join(scadir, "significant_evals_sca.npy"),
                self.significant_evals_sca,
            )
            np.save(
                os.path.join(scadir, "significant_evecs_sca.npy"),
                self.significant_evecs_sca,
            )
        if self.evals_shuff is not None and len(self.evals_shuff) > 0:
            np.save(os.path.join(scadir, EVALS_SHUFF_FNAME), self.evals_shuff)

        # ICA
        if self.v_ica is not None:
            np.save(os.path.join(scadir, "v_ica_normalized.npy"), self.v_ica)
        if self.w_ica is not None:
            np.save(os.path.join(scadir, "w_ica.npy"), self.w_ica)

        # t-distribution info
        if self.t_dists_info is not None:
            with open(os.path.join(scadir, "t_dists_info.json"), "w") as f:
                json.dump(self.t_dists_info, f)

        # Sector subset of SCA matrix
        if self.sca_matrix_sector_subset is not None:
            np.save(
                os.path.join(scadir, "sca_matrix_sector_subset.npy"),
                self.sca_matrix_sector_subset,
            )

        # Per-IC bundle: high-load positions in both coord spaces, plus
        # the IC's loadings at those positions. Single source of truth
        # at the top level; from_directory() loads from here.
        if self.ic_positions is not None:
            ic_pos_dir = os.path.join(outdir, "ic_positions")
            os.makedirs(ic_pos_dir, exist_ok=True)
            rp = (
                np.asarray(retained_positions, dtype=int)
                if retained_positions is not None else None
            )
            for i, positions in enumerate(self.ic_positions):
                np.save(
                    os.path.join(ic_pos_dir, f"ic_{i}_msaproc.npy"),
                    positions,
                )
                if rp is not None:
                    np.save(
                        os.path.join(ic_pos_dir, f"ic_{i}_msaorig.npy"),
                        rp[np.asarray(positions, dtype=int)],
                    )
                if self.group_scores is not None:
                    np.save(
                        os.path.join(ic_pos_dir, f"ic_{i}_loadings.npy"),
                        self.group_scores[i],
                    )
            # Combined mapping. Guard against the edge case where every IC
            # has an empty position set — np.concatenate of all-empty lists
            # raises.
            from mysca.run_sca import _safe_concat_int
            group_idxs_all = _safe_concat_int(self.ic_positions)
            group_idx_labels = _safe_concat_int(
                [np.full(len(g), i, dtype=int)
                 for i, g in enumerate(self.ic_positions)]
            )
            msapos_to_groupidx = np.vstack([group_idxs_all, group_idx_labels])
            np.save(
                os.path.join(scadir, "msapos_to_groupidx.npy"),
                msapos_to_groupidx,
            )
            # All important positions
            all_imp = np.unique(group_idxs_all)
            np.save(
                os.path.join(scadir, "all_important_positions.npy"), all_imp
            )

        # Per-target IC residues (raw-sequence coords) and parallel
        # IC loadings.
        if self.ic_residues_per_seq is not None:
            np.savez_compressed(
                os.path.join(outdir, IC_RESIDUES_PER_SEQ_FNAME),
                **self.ic_residues_per_seq,
            )
        if self.ic_loadings_per_seq is not None:
            np.savez_compressed(
                os.path.join(outdir, IC_LOADINGS_PER_SEQ_FNAME),
                **self.ic_loadings_per_seq,
            )

        # Optional sequence metadata table (always TSV — pandas is a
        # hard dep so loaders can rely on its presence).
        if self.sequence_metadata is not None:
            self.sequence_metadata.to_csv(
                os.path.join(outdir, SEQUENCE_METADATA_FNAME),
                sep="\t", index=False,
            )

    @classmethod
    def load(cls, dirpath):
        """Load results from a directory previously created by save()."""
        scadir = os.path.join(dirpath, "sca_results")

        # Core SCA results
        Dia = conservation = sca_matrix = phi_ia = fi0 = fia = None
        Cij_raw = Cijab_raw = fijab = None
        results_path = os.path.join(dirpath, SCARUN_RESULTS_FNAME)
        if os.path.isfile(results_path):
            data = np.load(results_path)
            Dia = data.get("Dia")
            conservation = data.get("conservation")
            sca_matrix = data.get("sca_matrix")
            phi_ia = data.get("phi_ia")
            fi0 = data.get("fi0")
            fia = data.get("fia")
            Cij_raw = data.get("Cij_raw")
            Cijab_raw = data.get("Cijab_raw")
            fijab = data.get("fijab")

        # Args
        args = None
        args_path = os.path.join(dirpath, SCARUN_ARGS_FNAME)
        if os.path.isfile(args_path):
            with open(args_path, "r") as f:
                args = json.load(f)

        # Eigendecomposition
        evals_sca = evecs_sca = None
        significant_evals_sca = significant_evecs_sca = None
        eigen_path = os.path.join(dirpath, SCARUN_EIGENDECOMP_FNAME)
        if os.path.isfile(eigen_path):
            data = np.load(eigen_path)
            evals_sca = data.get("evals_sca")
            evecs_sca = data.get("evecs_sca")
            significant_evals_sca = data.get("significant_evals_sca")
            significant_evecs_sca = data.get("significant_evecs_sca")

        # Bootstrap / significance
        kstar = _load_scalar_txt(os.path.join(scadir, "kstar.txt"), int)
        kstar_identified = _load_scalar_txt(
            os.path.join(scadir, "kstar_identified.txt"), int
        )
        n_components = _load_scalar_txt(
            os.path.join(scadir, "n_components.txt"), int
        )
        cutoff = _load_scalar_txt(
            os.path.join(scadir, "eigenvalue_cutoff.txt"), float
        )
        evals_shuff = _load_npy_or_none(
            os.path.join(scadir, EVALS_SHUFF_FNAME)
        )

        # ICA
        v_ica = _load_npy_or_none(
            os.path.join(scadir, "v_ica_normalized.npy")
        )
        w_ica = _load_npy_or_none(os.path.join(scadir, "w_ica.npy"))

        # t-distribution info
        t_dists_info = None
        tdist_path = os.path.join(scadir, "t_dists_info.json")
        if os.path.isfile(tdist_path):
            with open(tdist_path, "r") as f:
                t_dists_info = json.load(f)

        # Sector subset
        sca_matrix_sector_subset = _load_npy_or_none(
            os.path.join(scadir, "sca_matrix_sector_subset.npy")
        )

        # IC positions + loadings from the top-level ic_positions/ dir.
        ic_positions = None
        group_scores = None
        ic_pos_dir = os.path.join(dirpath, "ic_positions")
        if os.path.isdir(ic_pos_dir):
            ic_positions = []
            group_scores = []
            i = 0
            while True:
                gpath = os.path.join(ic_pos_dir, f"ic_{i}_msaproc.npy")
                if not os.path.isfile(gpath):
                    break
                ic_positions.append(np.load(gpath))
                spath = os.path.join(ic_pos_dir, f"ic_{i}_loadings.npy")
                if os.path.isfile(spath):
                    group_scores.append(np.load(spath))
                i += 1
            if not ic_positions:
                ic_positions = None
                group_scores = None
            elif not group_scores:
                group_scores = None

        # Per-target IC residues + loadings.
        ic_residues_per_seq = None
        residues_path = os.path.join(dirpath, IC_RESIDUES_PER_SEQ_FNAME)
        if os.path.isfile(residues_path):
            ic_residues_per_seq = dict(
                np.load(residues_path, allow_pickle=True)
            )

        ic_loadings_per_seq = None
        loadings_path = os.path.join(dirpath, IC_LOADINGS_PER_SEQ_FNAME)
        if os.path.isfile(loadings_path):
            ic_loadings_per_seq = dict(
                np.load(loadings_path, allow_pickle=True)
            )

        sequence_metadata = None
        md_path = os.path.join(dirpath, SEQUENCE_METADATA_FNAME)
        if os.path.isfile(md_path):
            import pandas as pd
            sequence_metadata = pd.read_csv(md_path, sep="\t")

        return cls(
            Dia=Dia,
            conservation=conservation,
            sca_matrix=sca_matrix,
            phi_ia=phi_ia,
            fi0=fi0,
            fia=fia,
            Cij_raw=Cij_raw,
            Cijab_raw=Cijab_raw,
            fijab=fijab,
            evals_sca=evals_sca,
            evecs_sca=evecs_sca,
            significant_evals_sca=significant_evals_sca,
            significant_evecs_sca=significant_evecs_sca,
            kstar=kstar,
            kstar_identified=kstar_identified,
            n_components=n_components,
            cutoff=cutoff,
            evals_shuff=evals_shuff,
            v_ica=v_ica,
            w_ica=w_ica,
            ic_positions=ic_positions,
            group_scores=group_scores,
            ic_residues_per_seq=ic_residues_per_seq,
            ic_loadings_per_seq=ic_loadings_per_seq,
            t_dists_info=t_dists_info,
            sca_matrix_sector_subset=sca_matrix_sector_subset,
            sequence_metadata=sequence_metadata,
            args=args,
        )


def _load_npy_or_none(path):
    """Load a .npy file if it exists, otherwise return None."""
    if os.path.isfile(path):
        return np.load(path)
    return None


def _load_scalar_txt(path, dtype):
    """Load a single scalar from a text file, or return None."""
    if os.path.isfile(path):
        return dtype(np.loadtxt(path))
    return None
