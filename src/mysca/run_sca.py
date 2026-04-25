"""SCA core pipeline

Runs SCA on preprocessed data: computes the SCA covariance matrix,
performs its eigendecomposition + bootstrap for significance, runs
ICA on the top eigenvectors, and assigns positions to IC groups
(sectors). Reference:
    [1] SI to Rivoire et al., 2016

See `docs/cli_reference.md` for the canonical per-argument table.

-------------------------------------------------------------------------------
COMMAND LINE ARGUMENTS:

Required:
    -i --indir            : sca-preprocess output directory.
    -o --outdir           : output directory for SCA results.

SCA parameters (see SI of [1]):
    --regularization λ    : SCA regularization (default 0.03).
    --background          : optional JSON file of background frequencies
                            (default: built-in DEFAULT_BACKGROUND_FREQ).
    -nb --n_boot          : bootstrap iterations used to set the
                            significance cutoff (default 10). 0 reuses an
                            existing bootstrap from --load_data; -1 skips
                            bootstrapping entirely (all evals significant).
    -k --kstar            : override the bootstrap-derived kstar (0 keeps
                            the bootstrap estimate).
    --n_components        : number of ICs to compute (>= kstar). Integer
                            or "all" (=L, the number of retained
                            positions). Default: kstar.
    -p --pstar            : percentile defining the t-distribution cutoff
                            that nominates positions per IC (default 95).
    --assignment          : "overlap" (default) keeps a qualifying
                            position in every IC that nominates it;
                            "exclusive" assigns only to the IC with the
                            maximal projection.
    --weak_assignment     : IC indices to exclude from the
                            "exclusive" tie-break (ignored otherwise).
    --n_logged_comps      : top-N ICs to emit in the human-readable
                            summary log (default 10; 0 disables).
    --sectors_for         : which sequences get per-sequence sector
                            mappings. Default: reference only. "all" or a
                            text file of IDs.

Optional:
    --seed                : random seed (None or non-positive auto-picks).
    --load_data           : previous sca-core output directory to reload.
    --save_all            : also write the large Cijab_raw / fijab arrays.
    --use_jax             : use JAX in the core computations.
    --nodendro            : skip the sequence-similarity / dendrogram plots.
    --sector_cmap         : {"default", "none"} sector palette for the
                            sector-subset plot.
    --pbar                : tqdm progress bars for bootstrap iterations.
    -v --verbosity        : 0=warnings only; higher = more detail.

-------------------------------------------------------------------------------
OUTPUTS (see docs/cli_reference.md ## sca-core for the exhaustive list):

scarun_results.npz
    Dia, conservation, sca_matrix, phi_ia, fi0, fia, Cij_raw; plus
    Cijab_raw, fijab if --save_all.

sca_eigendecomp.npz
    Full + significant eigenvalues/eigenvectors of sca_matrix.

scarun_args.json
    CLI arguments used for the run.

ic_residues_per_seq.npz
    Per-target IC residues in raw-sequence coordinates, keyed
    `ic_{i}_{seqid}`. Only top-kstar ICs expanded per sequence.

ic_loadings_per_seq.npz
    Per-residue IC loadings parallel to ic_residues_per_seq, same
    `ic_{i}_{seqid}` key format.

sca_results/
    v_ica_normalized.npy, w_ica.npy, t_dists_info.json, evals_shuff.npy,
    sca_matrix_sector_subset.npy, scalar txt files, and per-IC position
    / loading arrays under msa_sectors/.

ic_positions/
    ic_{i}_msaproc.npy, ic_{i}_msaorig.npy (one per IC, processed-MSA
    and original-MSA coordinates respectively).

scarun.log
    Run log including the human-readable top-N IC summary.

images/
    Conservation, SCA-matrix, spectrum, dendrogram, t-distribution,
    EV/IC 2D/3D scatter, and sector-subset figures.

-------------------------------------------------------------------------------
EXAMPLE USAGE:

    sca-core -i </preprocess-out> -o </scacore-out> --regularization 0.03

    sca-core -i </preprocess-out> -o </scacore-out> \\
        --kstar 6 --n_components 10 --sectors_for all --seed 42

    sca-core -i </preprocess-out> -o </scacore-out> \\
        --n_boot 0 --load_data </existing-scacore-out>

"""

import argparse
import logging
import os, sys
import warnings
import numpy as np
import tqdm as tqdm
import json

import scipy
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform

from mysca.logging_config import configure_logging
from mysca.results import (
    PreprocessingResults,
    SCAResults,
    SCARUN_RESULTS_FNAME as OUTPUT_RESULTS_FNAME,
    SCARUN_ARGS_FNAME as OUTPUT_ARGS_FNAME,
    EVALS_SHUFF_FNAME as EVALS_SHUFF_SAVEAS,
)
from mysca.core import run_sca, run_ica
from mysca.preprocess import onehot_without_gap
from mysca.helpers import get_rawseq_positions_in_groups
from mysca.helpers import get_rawseq_scores_in_groups
from mysca.helpers import get_group_rawseq_positions_by_entry
from mysca.helpers import get_group_rawseq_scores_by_entry
from mysca.helpers import get_rawseq_indices_of_msa
from mysca.constants import SECTOR_COLORS, DEFAULT_BACKGROUND_FREQ

from mysca.pl import (
    plot_conservation,
    plot_conservation_positional,
    plot_conservation_top,
    plot_covariance_matrix,
    plot_data_2d,
    plot_data_3d,
    plot_dendrogram,
    plot_sca_matrix,
    plot_sca_matrix_sector_subset,
    plot_sca_spectrum,
    plot_sca_spectrum_vs_null,
    plot_sequence_similarity,
    plot_t_distributions,
)

SCARUN_LOG_FNAME = "scarun.log"

logger = logging.getLogger("mysca.run_sca")


def _n_components_type(s):
    """argparse type converter for --n_components: positive int or 'all'."""
    if isinstance(s, str) and s.lower() == "all":
        return "all"
    try:
        v = int(s)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"--n_components must be a positive integer or 'all'; got {s!r}"
        )
    if v < 1:
        raise argparse.ArgumentTypeError(
            f"--n_components must be >= 1 or 'all'; got {v}"
        )
    return v


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", type=str, required=True,
                        help="Path to preprocessed data.")
    parser.add_argument("-o", "--outdir", type=str, required=True, 
                        help="Output directory.")
    parser.add_argument("--pbar", action="store_true",
                        help="Enable tqdm progress bars during bootstrapping.")
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="Verbosity level (0=warnings only).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility. None or a "
                        "non-positive value auto-generates one.")
    
    parser.add_argument("--use_jax", action="store_true", 
                        help="Use JAX in computations.")
    
    parser.add_argument("--nodendro", action="store_true", 
                        help="Skip dendrogram plots")
    parser.add_argument("--save_all", action="store_true", 
                        help="Save all SCA results (includes large files).")
    parser.add_argument("--load_data", type=str, default="", 
                        help="SCA directory to load precomputed data.")
    parser.add_argument("--sector_cmap", type=str, default="default",
                        choices=["none", "default"])
    
    sca_params = parser.add_argument_group("SCA parameters")
    sca_params.add_argument("--regularization", type=float, default=0.03,
                    help="SCA regularization parameter λ")
    sca_params.add_argument("--background", type=str, default=None,
                    help="Optional json file specifying background q.")
    sca_params.add_argument("-nb", "--n_boot", type=int, default=10, 
                    help="Number of bootstraps to use for eval threshold.")
    sca_params.add_argument("-k", "--kstar", type=int, default=0,
                    help="Value of k_start to override bootstrap estimate.")
    sca_params.add_argument(
        "--n_components", type=_n_components_type, default=None,
        help="Number of ICs to compute and save. Accepts a positive integer "
        "or the string 'all' (meaning L, the number of retained positions). "
        "Default: kstar (the number of significant eigenvalues). "
        "Clamped to kstar as a lower bound; i.e. if n_components < kstar the "
        "effective value is kstar.",
    )
    sca_params.add_argument("-p", "--pstar", type=int, default=95,
                    help="Percentile defining IC groups.")
    sca_params.add_argument(
        "--n_logged_comps", type=int, default=10,
        help="Number of top ICs to summarize in the log after assignment "
        "(significance marker, eigenvalue, MSA positions in processed / "
        "unprocessed / reference coordinates). Default 10.",
    )
    sca_params.add_argument(
        "--assignment", type=str, default="overlap",
        choices=["overlap", "exclusive"],
        help="How to assign a position that clears the t-distribution "
        "cutoff for multiple ICs. 'overlap' (default): keep it in every "
        "IC group where it qualifies. 'exclusive': assign only to the IC "
        "where its IC-projection is maximal (this was the previous "
        "default). --weak_assignment only applies under 'exclusive'.",
    )
    sca_params.add_argument("--weak_assignment", type=int, nargs="*",
                        default=[])

    parser.add_argument("--sectors_for", type=str, default=None,
                        help="Which sequences to generate per-sequence sector "
                             "mappings for. Default: only the reference "
                             "sequence. 'all' for every retained sequence, "
                             "or a path to a text file with one sequence ID "
                             "per line.")

    return parser.parse_args(args)


def main(args):

    # Process command line args
    indir = args.indir
    OUTDIR = args.outdir
    verbosity = args.verbosity
    N_BOOT = args.n_boot
    PBAR = args.pbar
    SEED = args.seed
    DENDRO = not args.nodendro
    LOAD_DATA = args.load_data
    USE_JAX = args.use_jax
    SAVE_ALL = args.save_all
    sector_cmap = args.sector_cmap
    assignment_method = args.assignment
    weak_assignment = args.weak_assignment
    sectors_for = args.sectors_for

    regularization = args.regularization
    background_freq = args.background
    kstar = args.kstar
    pstar = args.pstar
    n_components_arg = args.n_components
    n_logged_comps = args.n_logged_comps

    if N_BOOT < 0:
        N_BOOT = 0
    
    ####################
    ##  Housekeeping  ##
    ####################

    # Create subdirectories within the specified output directory.
    SCADIR = os.path.join(OUTDIR, "sca_results")
    IMGDIR = os.path.join(OUTDIR, "images")
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(SCADIR, exist_ok=True)
    os.makedirs(IMGDIR, exist_ok=True)

    configure_logging(
        verbosity=verbosity,
        logfile=os.path.join(OUTDIR, SCARUN_LOG_FNAME),
    )

    if SEED is None or SEED <= 0:
        SEED = np.random.randint(2**32)
    rng = np.random.default_rng(seed=SEED)

    # Load background frequencies or use the default
    use_default_background = (background_freq is None) or (
        isinstance(background_freq, str) and
            background_freq.lower() in ["default", "none"]
    )
    if use_default_background:
        background_freq = DEFAULT_BACKGROUND_FREQ
    elif isinstance(background_freq, str):
        logger.info(
            "Loading background frequencies from file: %s", background_freq
        )
        background_freq = load_background(background_freq)
    else:
        msg = f"Cannot handle given argument for background: {background_freq}"
        raise RuntimeError(msg)

    # Predefined colors for the sectors
    sector_color_set = {
        "default": SECTOR_COLORS,
        "none": None,
    }[sector_cmap]

    # Load preprocessed data
    if not os.path.isdir(indir):
        msg = f"Preprocessed data directory not found! {indir}"
        raise FileNotFoundError(msg)
    prep = PreprocessingResults.load(indir)
    sym_map = prep.sym_map
    msa = prep.msa
    retained_sequences = prep.retained_sequences
    retained_positions = prep.retained_positions
    weights = prep.sequence_weights
    msa_binary3d = prep.msa_binary3d
    NSYMS = len(sym_map)
    msa_obj_orig = prep.msa_obj_orig
    NUM_POS_ORIG = msa_obj_orig.get_alignment_length()

    # Create the background frequency distribution q
    logger.info("Background frequencies:")
    logger.info(
        "  %s",
        ", ".join(
            f"{k}: {background_freq[k]:.3g}"
            for k in np.sort(list(background_freq.keys()))
        ),
    )
    background_freq_array = np.array(
        [background_freq.get(a, 0.0) for a in sym_map.aa_list]
    )
    background_freq_array = background_freq_array / background_freq_array.sum()
    
    # Run SCA
    logger.info("Running SCA...")
    if not LOAD_DATA:
        sca_results = run_sca(
            msa_binary3d, weights,
            background_map=background_freq,
            mapping=sym_map,
            background_arr=background_freq_array,
            regularization=regularization,
            return_keys="all",
            pbar=PBAR,
            leave_pbar=True,
            use_jax=USE_JAX,
        )
        results = SCAResults.from_core_output(sca_results)
        Dia = results.Dia
        Di = results.conservation
        Cij_raw = results.Cij_raw
        Cij = results.sca_matrix
        del sca_results  # relieve memory
    else:
        existing = SCAResults.load(LOAD_DATA)
        results = existing
        Di = existing.conservation
        Cij = existing.sca_matrix
        Cij_raw = existing.Cij_raw  # None for saves from before Cij_raw persistence landed

    # Eigendecomposition of SCA matrix
    evals_sca, evecs_sca = np.linalg.eigh(Cij)
    evals_sca = np.flip(evals_sca)
    evecs_sca = np.flip(evecs_sca, axis=1)

    logger.info(
        "Eigenvalue spectrum of SCA Matrix: %.3g, %.3g",
        evals_sca.min(), evals_sca.max(),
    )
    
    # Perform bootstrapping to get eigenvalue null distribution
    # If N_BOOT is positive, we perform the specified number of bootstrap 
    # resamplings. If instead existing data is specified, we load these results
    # and proceed, inferring N_BOOT from the loaded data. If no existing data
    # is found, we skip the bootstrapping.
    # Since bootstrap results vary from seed to seed, we save these results 
    # outside of the core set of results.
    DO_SHUFFLING = N_BOOT > 0
    evals_shuff = np.full([N_BOOT, *evals_sca.shape], np.nan)
    evals_shuff_fpath = os.path.join(SCADIR, EVALS_SHUFF_SAVEAS)
    if DO_SHUFFLING:
        for iteridx in tqdm.trange(N_BOOT):
            msa_shuff = shuffle_columns(msa, rng=rng)
            xmsa_shuff = onehot_without_gap(msa_shuff, NSYMS, sym_map.gapint)
            res = run_sca(
                xmsa_shuff, weights,
                background_map=background_freq,
                mapping=sym_map,
                background_arr=background_freq_array,
                regularization=regularization,
                return_keys=["Cij_corr"],
                pbar=PBAR,
                leave_pbar=False,
            )
            cij_shuff = res["Cij_corr"]
            evals = np.linalg.eigvalsh(cij_shuff)
            evals_shuff[iteridx] = np.flip(evals)
        np.save(evals_shuff_fpath, evals_shuff)
    elif LOAD_DATA:
        evals_shuff_fpath_toload = os.path.join(LOAD_DATA, "sca_results", EVALS_SHUFF_SAVEAS)
        logger.info(
            "Skipping bootstrap. Loading existing null evals at: %s",
            evals_shuff_fpath_toload,
        )
        evals_shuff = np.load(evals_shuff_fpath_toload)
        N_BOOT = evals_shuff.shape[0]
    elif os.path.isfile(evals_shuff_fpath):
        logger.info(
            "Skipping bootstrap. Loading existing null evals at: %s",
            evals_shuff_fpath,
        )
        evals_shuff = np.load(evals_shuff_fpath)
        # Determine the bootstrap size from the loaded data
        N_BOOT = evals_shuff.shape[0]
    else:
        evals_shuff = []
        N_BOOT = 0
        logger.info("Skipping bootstrap. No existing eigenvalue data found.")

    # Determine kstar, the number of significant eigenvalues. See SI G of [1]
    cutoff = np.mean(evals_shuff[:,1]) + 2 * np.std(evals_shuff[:,1])
    kstar_id = np.sum(evals_sca > cutoff)
    logger.info("significant eigenvalue cutoff: %s", cutoff)
    logger.info(
        "Identified %d significant eigenvalues:\n%s",
        kstar_id, evals_sca[:kstar_id],
    )
    if kstar <= 0:
        kstar = kstar_id
        logger.info("Setting kstar=%d", kstar)
    else:
        kstar = min(kstar, len(evals_sca))
        logger.info("Overriding kstar from command line input!")
        logger.info("Setting kstar=%d", kstar)

    if kstar == 0:
        msg = "No significant eigenvalues (kstar=0). Proceeding with kstar=1"
        logger.warning(msg)
        warnings.warn(msg)
        kstar = 1

    # Consider top kstar values, excluding top value
    sig_evals_sca = evals_sca[:kstar]
    sig_evecs_sca = evecs_sca[:,:kstar]

    # Determine how many ICs to compute (>= kstar). Default is kstar; "all"
    # means len(evals_sca). Values below kstar are clamped up.
    L_evecs = len(evals_sca)
    if n_components_arg is None:
        n_components = kstar
    elif n_components_arg == "all":
        n_components = L_evecs
    else:
        n_components = int(n_components_arg)
    if n_components < kstar:
        logger.warning(
            "n_components=%d < kstar=%d; clamping to kstar.",
            n_components, kstar,
        )
        n_components = kstar
    if n_components > L_evecs:
        logger.warning(
            "n_components=%d exceeds number of eigenvectors (%d); clamping.",
            n_components, L_evecs,
        )
        n_components = L_evecs
    logger.info(
        "Computing ICA on top %d eigenvectors (kstar=%d, L=%d).",
        n_components, kstar, L_evecs,
    )

    # Populate eigendecomposition on results
    results.evals_sca = evals_sca
    results.evecs_sca = evecs_sca
    results.significant_evals_sca = sig_evals_sca
    results.significant_evecs_sca = sig_evecs_sca
    results.kstar = kstar
    results.kstar_identified = kstar_id
    results.n_components = n_components
    results.cutoff = cutoff
    results.evals_shuff = evals_shuff

    # Apply Independent Component Analysis (ICA)
    ica_rho = 1e-1
    ica_tol = 1e-7
    ica_maxiter = 1E6
    ica_max_attempts = 5
    ica_input_evecs = evecs_sca[:, :n_components]
    v_ica_normalized, _, w_ica = apply_ica(
        ica_input_evecs,
        rho=ica_rho, tol=ica_tol, maxiter=ica_maxiter,
        max_attempts=ica_max_attempts,
    )
    results.v_ica = v_ica_normalized
    results.w_ica = w_ica

    # Fit t-distribution to each IC
    t_dists_info, top_idxs = fit_t_distributions(v_ica_normalized, p=pstar)
    for i, idxs in enumerate(top_idxs):
        if len(idxs) == 0:
            logger.warning(
                "IC %d: no positions cleared the t-distribution cutoff "
                "at pstar=%d.",
                i, pstar,
            )
    all_imp_idxs = _safe_concat_int(top_idxs)
    all_imp_idxs_unique = np.unique(all_imp_idxs)
    logger.info(
        "Identified %d important positions (with repeats).",
        len(all_imp_idxs),
    )
    logger.info(
        "Identified %d important positions (w/o repeats).",
        len(all_imp_idxs_unique),
    )
    results.t_dists_info = t_dists_info

    # Call statistical sectors, i.e. groups of "co-evolving" positions.
    # Define groups from top p% empirical distribution.
    logger.info("Assigning positions to IC groups (method=%s).", assignment_method)
    groups, group_scores = assign_positions_to_groups(
        top_idxs,
        v_ica_normalized,
        method=assignment_method,
        weak_assignment=weak_assignment,
    )

    # Subset the SCA matrix into grouped important positions. If every IC
    # ended up empty, the subset is a 0x0 matrix — log and carry on.
    for i, g in enumerate(groups):
        if len(g) == 0:
            logger.info("IC %d: group is empty after assignment.", i)
    group_idxs_all = _safe_concat_int(groups)
    if len(group_idxs_all) == 0:
        logger.warning(
            "All IC groups are empty; sca_matrix_sector_subset will be 0x0."
        )
    sca_mat_imp = Cij[group_idxs_all,:]
    sca_mat_imp = sca_mat_imp[:,group_idxs_all]

    results.ic_positions = groups
    results.group_scores = group_scores
    results.sca_matrix_sector_subset = sca_mat_imp

    # Human-readable top-N IC summary (significance marker + eigenvalue +
    # processed / unprocessed / reference position mappings).
    ref_id_for_log = prep.args.get("reference_id") if prep.args else None
    log_top_ic_summary(
        groups, kstar, evals_sca, retained_positions,
        msa_obj_orig, ref_id_for_log,
        n_logged_comps=n_logged_comps,
    )

    # Determine which sequences to generate per-sequence sector mappings for.
    # IDs that were filtered out during preprocessing are silently skipped.
    retained_ids = set(
        msa_obj_orig[int(sidx)].id for sidx in retained_sequences
    )
    if sectors_for is not None and sectors_for.lower() == "all":
        sector_seqidxs = retained_sequences
    elif sectors_for is not None:
        with open(sectors_for, "r") as f:
            requested_ids = set(
                line.strip() for line in f if line.strip()
            )
        missing_ids = requested_ids - retained_ids
        if missing_ids:
            logger.info(
                "Note: %d requested sequence(s) not found among retained "
                "sequences (filtered during preprocessing): %s",
                len(missing_ids), ", ".join(sorted(missing_ids)),
            )
        found_ids = requested_ids & retained_ids
        sector_seqidxs = np.array([
            sidx for sidx in retained_sequences
            if msa_obj_orig[int(sidx)].id in found_ids
        ])
        logger.info(
            "Generating per-sequence sectors for %d/%d sequences.",
            len(sector_seqidxs), len(retained_sequences),
        )
    else:
        # Default: only the reference sequence
        ref_id = prep.args.get("reference_id") if prep.args else None
        if ref_id is not None and ref_id in retained_ids:
            sector_seqidxs = np.array([
                sidx for sidx in retained_sequences
                if msa_obj_orig[int(sidx)].id == ref_id
            ])
            logger.info(
                "Generating per-sequence sectors for reference sequence: %s",
                ref_id,
            )
        else:
            sector_seqidxs = np.array([], dtype=int)
            if ref_id is not None:
                logger.info(
                    "Reference sequence '%s' was filtered out during "
                    "preprocessing. Skipping per-sequence sector mappings.",
                    ref_id,
                )
            else:
                logger.info(
                    "No reference sequence specified. "
                    "Skipping per-sequence sector mappings."
                )

    # Map processed MSA positions to original sequence positions
    rawseq_idxs = get_rawseq_indices_of_msa(msa_obj_orig)
    rawseq_idxs = rawseq_idxs[retained_sequences,:]
    rawseq_idxs = rawseq_idxs[:,retained_positions]

    # Compute residue groups by raw sequence position (for subset only)
    sector_rawseq_idxs = rawseq_idxs[
        np.isin(retained_sequences, sector_seqidxs)
    ]
    group_rawseq_positions = get_rawseq_positions_in_groups(
        sector_rawseq_idxs, groups
    )
    group_rawseq_scores = get_rawseq_scores_in_groups(
        sector_rawseq_idxs, groups, group_scores
    )
    group_rawseq_positions_by_entry = get_group_rawseq_positions_by_entry(
        msa_obj_orig, sector_seqidxs, groups, group_rawseq_positions
    )
    group_rawseq_scores_by_entry = get_group_rawseq_scores_by_entry(
        msa_obj_orig, sector_seqidxs, groups, group_rawseq_scores
    )
    # Per-target IC residues (and parallel IC loadings) scale with
    # n_components × |sector_seqidxs| and dominate on-disk size when the
    # user requests many ICs plus `--sectors_for all`. Restrict to the
    # kstar significant ICs; non-significant ICs still have their
    # position arrays on disk (via SCAResults.save) but aren't expanded
    # per sequence.
    ic_residues_per_seq = {}
    ic_loadings_per_seq = {}
    n_sector_groups = min(kstar, len(groups))
    for gidx in range(n_sector_groups):
        for seqidx in sector_seqidxs:
            entry = msa_obj_orig[int(seqidx)]
            sid = entry.id
            residues = group_rawseq_positions_by_entry[sid][gidx]
            loadings = group_rawseq_scores_by_entry[sid][gidx]
            key = f"ic_{gidx}_{sid}"
            ic_residues_per_seq[key] = residues
            ic_loadings_per_seq[key] = loadings
    if len(groups) > n_sector_groups:
        logger.info(
            "Per-target IC residues generated for the top %d "
            "(significant) ICs only; %d additional non-significant IC "
            "group(s) are saved as index files but not expanded per sequence.",
            n_sector_groups, len(groups) - n_sector_groups,
        )

    results.ic_residues_per_seq = ic_residues_per_seq
    results.ic_loadings_per_seq = ic_loadings_per_seq

    # Save all results
    results.args = {
        "regularization": float(regularization),
        "n_boot": int(N_BOOT),
        "seed": int(SEED),
        "kstar": int(kstar),
        "n_components": int(n_components),
        "pstar": int(pstar),
        "assignment": assignment_method,
        "n_logged_comps": int(n_logged_comps),
    }
    results.save(
        OUTDIR, save_all=SAVE_ALL, retained_positions=retained_positions,
    )

    make_plots(
        retained_positions, 
        Di, 
        NUM_POS_ORIG,
        IMGDIR, 
        DENDRO,
        msa_binary3d,
        Cij_raw,
        Cij,
        evals_shuff,
        evals_sca,
        cutoff,
        N_BOOT,
        kstar,
        v_ica_normalized,
        t_dists_info,
        groups,
        sig_evecs_sca,
        sca_mat_imp,
        sector_color_set,
    )

    logger.info("Done!")


def load_background(fpath):
    with open(fpath, "r") as f:
        data = json.load(f)
    return data


def apply_ica(
        sig_evecs_sca, *,
        rho,
        tol,
        maxiter,
        max_attempts,
):
    n_attempts = 0
    while n_attempts < max_attempts:
        n_attempts += 1
        w_ica, ica_delta = run_ica(
            sig_evecs_sca.T,
            rho=rho,
            tol=tol,
            maxiter=maxiter,
        )
        if w_ica is None:
            logger.warning(
                "ICA did not converge with parameters rho=%.3g, tol=%.3g, "
                "maxiter=%s. (Reached tol=%.3g)",
                rho, tol, maxiter, ica_delta,
            )
            maxiter *= 2
            rho /= 2
        else:
            v_ica = sig_evecs_sca @ w_ica.T
            logger.info(
                "ICA succeeded after %d attempts. (tol=%.2g)",
                n_attempts, tol,
            )
            break

    # Check success
    if w_ica is None:
        raise RuntimeError(f"ICA failed to converge in {max_attempts} attempts.")

    # Normalize V and ensure positivity of maximum entry.
    v_ica_normalized = v_ica / np.sqrt(np.sum(np.square(v_ica), axis=0))
    for i in range(v_ica.shape[1]):
        maxpos = np.argmax(np.abs(v_ica_normalized[:,i]))
        if v_ica_normalized[maxpos,i] < 0:
            v_ica_normalized[:,i] *= -1
    return v_ica_normalized, v_ica, w_ica


_IC_LABEL_WIDTH = 15  # aligns "processed:" / "unprocessed:" / "reference ..."


def _format_list(values):
    """Render a homogeneous list to ``[a, b, c]`` without Python's
    default-repr quoting of strings (so gap-representing ``"-"`` and
    residue letters print bare)."""
    return "[" + ", ".join(str(v) for v in values) + "]"


def log_top_ic_summary(
        groups, kstar, evals_sca, retained_positions,
        msa_obj_orig, reference_id, *, n_logged_comps=10,
):
    """Write a human-readable summary of the top-N ICs to the module logger.

    Each IC gets a header line (significance marker + eigenvalue +
    position count) and a multi-line, left-aligned block showing:

    - ``processed``: positions in the *processed* (post-filter) MSA.
    - ``unprocessed``: same positions in the *original* (pre-filter)
      MSA, via ``retained_positions``.
    - ``reference pos``: residue indices in the reference's raw
      (ungapped) sequence at each unprocessed position, with ``"-"``
      where the reference has a gap.
    - ``reference res``: residue letters at each reference position
      (``"-"`` for gaps), useful for identifying the actual amino
      acids involved without cross-referencing the MSA.

    The reference block is only emitted when ``reference_id`` resolves
    to a row in ``msa_obj_orig``; the header echoes the chosen
    reference so downstream readers aren't guessing.

    No-op when ``n_logged_comps <= 0`` or ``groups`` is empty.
    """
    if n_logged_comps <= 0 or not groups:
        return

    aligned_ref = None
    ref_raw_positions = None
    if reference_id is not None:
        ids = [rec.id for rec in msa_obj_orig]
        if reference_id in ids:
            ref_row = ids.index(reference_id)
            aligned_ref = str(msa_obj_orig[ref_row].seq)
            all_raw = get_rawseq_indices_of_msa(msa_obj_orig)
            ref_raw_positions = all_raw[ref_row]  # shape (npos_orig,)

    n_show = min(n_logged_comps, len(groups))
    header_tag = (
        f"reference={reference_id}; " if aligned_ref is not None else ""
    )
    logger.info(
        "Top %d/%d ICs (%s* significant, - not). "
        "λ_i = i-th sorted SCA eigenvalue.",
        n_show, len(groups), header_tag,
    )
    for i in range(n_show):
        g = groups[i]
        marker = "*" if i < kstar else "-"
        eigval = float(evals_sca[i]) if i < len(evals_sca) else float("nan")
        processed = [int(x) for x in g]
        unprocessed = [int(retained_positions[x]) for x in g]

        logger.info(
            "IC %d: %s λ_%d=%.4g  (%d positions)",
            i, marker, i, eigval, len(g),
        )
        logger.info(
            "    %-*s%s",
            _IC_LABEL_WIDTH, "processed:", _format_list(processed),
        )
        logger.info(
            "    %-*s%s",
            _IC_LABEL_WIDTH, "unprocessed:", _format_list(unprocessed),
        )
        if aligned_ref is not None:
            ref_positions = [
                int(ref_raw_positions[mp]) if ref_raw_positions[mp] >= 0
                else "-"
                for mp in unprocessed
            ]
            ref_residues = [aligned_ref[mp] for mp in unprocessed]
            logger.info(
                "    %-*s%s",
                _IC_LABEL_WIDTH, "reference pos:", _format_list(ref_positions),
            )
            logger.info(
                "    %-*s%s",
                _IC_LABEL_WIDTH, "reference res:", _format_list(ref_residues),
            )


def _safe_concat_int(arrays):
    """Concatenate a list of 1-D int arrays, returning an empty int array if
    all inputs are empty. ``np.concatenate`` raises on an all-empty list, so
    this guard lets callers handle the "no positions pass the cutoff" edge
    case without special-casing it at every call site.
    """
    if not any(len(a) for a in arrays):
        return np.array([], dtype=int)
    return np.concatenate(arrays, axis=0)


def assign_positions_to_groups(
        top_idxs, v_ica_normalized, *,
        method="overlap",
        weak_assignment=(),
):
    """Resolve per-IC candidate positions into final groups.

    Parameters
    ----------
    top_idxs : list[np.ndarray]
        For each IC, the (ordered) indices that cleared the t-distribution
        cutoff — as returned by ``fit_t_distributions``.
    v_ica_normalized : np.ndarray, shape (L, n_components)
        IC projections. Only consulted under ``method='exclusive'``.
    method : {'overlap', 'exclusive'}
        - ``'overlap'`` (default): each IC keeps its full ``top_idxs[i]``;
          a position that clears the cutoff on multiple ICs appears in all
          of them.
        - ``'exclusive'``: a position that lands in multiple ICs' candidate
          sets is assigned only to the IC where its IC-projection is
          maximal, ignoring ICs listed in ``weak_assignment``.
    weak_assignment : iterable[int]
        IC indices to exclude from the max-projection tie-break. Only
        applies under ``method='exclusive'``.

    Returns
    -------
    groups : list[np.ndarray]
        Indices per IC, ordered as they came out of ``fit_t_distributions``.
    group_scores : list[np.ndarray]
        ``v_ica_normalized[idx, i]`` for each ``idx`` in ``groups[i]``.
    """
    if method == "overlap":
        groups = [np.asarray(idxs, dtype=int) for idxs in top_idxs]
        group_scores = [
            v_ica_normalized[idxs, i] if len(idxs) else np.array([], dtype=float)
            for i, idxs in enumerate(groups)
        ]
        return groups, group_scores

    if method == "exclusive":
        if len(top_idxs) == 0:
            return [], []
        all_idxs = _safe_concat_int(top_idxs)
        screen = ~np.isin(
            np.arange(v_ica_normalized.shape[1]), list(weak_assignment)
        )
        groups = []
        group_scores = []
        for i, idx_set in enumerate(top_idxs):
            group = []
            group_score = []
            for idx in idx_set:
                hits = int(np.sum(all_idxs == idx))
                if hits == 1:
                    group.append(idx)
                    group_score.append(v_ica_normalized[idx, i])
                elif hits > 1:
                    if np.all(
                        v_ica_normalized[idx, i] >= v_ica_normalized[idx, screen]
                    ):
                        group.append(idx)
                        group_score.append(v_ica_normalized[idx, i])
                else:
                    raise RuntimeError(
                        "Index should be found among all candidate indices."
                    )
            groups.append(np.array(group, dtype=int))
            group_scores.append(np.array(group_score))
        return groups, group_scores

    raise ValueError(
        f"Unknown assignment method: {method!r}. "
        "Expected 'overlap' or 'exclusive'."
    )


def fit_t_distributions(v, p):
    """Fit a t-dist to each IC, and return indices in the pth pctl of each.
    """
    t_dists_info = []
    top_idxs = []
    for i in range(v.shape[1]):
        vi = v[:,i]
        df, loc, scale = scipy.stats.t.fit(vi)
        cutoff = scipy.stats.t.ppf(p/100, df, loc=loc, scale=scale)
        idxs = np.where(vi >= cutoff)[0]
        order = np.flip(np.argsort(vi[idxs]))
        idxs = idxs[order] # reorder by decreasing contribution
        t_dists_info.append(
            {"df": df, "loc": loc, "scale": scale, "cutoff": cutoff}
        )
        top_idxs.append(idxs)
    return t_dists_info, top_idxs


def shuffle_columns(m, rng=None):
    rng = np.random.default_rng(rng)
    r, c = m.shape
    idx = np.argsort(rng.random((r, c)), axis=0)
    return m[idx, np.arange(c)]


EV_AXES_2D = [  # ((EVi, EVj), group_idxs)
    ((0, 1), "all"),
    ((1, 2), "all"),
    ((2, 3), "all"),
    ((3, 4), "all"),
    ((4, 5), "all"),
    ((5, 6), "all"),
    ((0, 1), [0, 1, 2]),
    ((1, 2), [0, 1, 2]),
]

EV_AXES_3D = [  # ((EVi, EVj, EVk), group_idxs)
    ((0, 1, 2), "all"),
    ((1, 2, 3), "all"),
    ((0, 1, 2), [0, 1, 2]),
    ((1, 2, 3), [0, 1, 2]),
]

# Sweep shape is identical for IC coords.
IC_AXES_2D = EV_AXES_2D
IC_AXES_3D = EV_AXES_3D


def make_plots(
        retained_positions,
        Di,
        NUM_POS_ORIG,
        IMGDIR,
        DENDRO,
        msa_binary3d,
        Cij_raw,
        Cij,
        evals_shuff,
        evals_sca,
        cutoff,
        N_BOOT,
        kstar,
        v_ica_normalized,
        t_dists_info,
        groups,
        sig_evecs_sca,
        sca_mat_imp,
        sector_color_set,
):
    plot_conservation_top(retained_positions, Di, NUM_POS_ORIG, IMGDIR)
    plot_conservation_positional(retained_positions, Di, NUM_POS_ORIG, IMGDIR)
    plot_conservation(Di, IMGDIR)

    if DENDRO:
        plot_sequence_similarity(msa_binary3d, IMGDIR)

    if Cij_raw is not None:
        plot_covariance_matrix(Cij_raw, IMGDIR)

    plot_sca_matrix(Cij, IMGDIR)
    plot_sca_spectrum(evals_sca, evals_shuff, IMGDIR)
    plot_sca_spectrum_vs_null(
        evals_sca, evals_shuff, cutoff, N_BOOT, IMGDIR,
    )

    if DENDRO:
        plot_dendrogram(Cij, IMGDIR, nclusters=kstar)

    plot_t_distributions(
        v_ica_normalized, t_dists_info, IMGDIR, max_plots=kstar,
    )

    for axidxs, group_idxs in EV_AXES_2D:
        plot_data_2d(
            "ev", axidxs, group_idxs, groups, sig_evecs_sca, IMGDIR,
        )
    for axidxs, group_idxs in EV_AXES_3D:
        plot_data_3d(
            "ev", axidxs, group_idxs, groups, sig_evecs_sca, IMGDIR,
        )
    for axidxs, group_idxs in IC_AXES_2D:
        plot_data_2d(
            "ic", axidxs, group_idxs, groups, v_ica_normalized, IMGDIR,
        )
    for axidxs, group_idxs in IC_AXES_3D:
        plot_data_3d(
            "ic", axidxs, group_idxs, groups, v_ica_normalized, IMGDIR,
        )

    plot_sca_matrix_sector_subset(
        sca_mat_imp, groups, sector_color_set, IMGDIR,
    )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
