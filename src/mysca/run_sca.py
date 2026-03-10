"""SCA core pipeline

Runs SCA, given preprocessed data.
Stores all output in a specified directory.
See references:
    [1] SI to Rivoire et al., 2016

-------------------------------------------------------------------------------
COMMAND LINE ARGUMENTS:
    -i --indir : Path to preprocessed input data.
    -o --outdir : Output directory.
    --regularization : (Default 0.03) SCA regularization parameter λ.
    -nb --n_boot : (Default 10) Number of bootstrap samples to perform in order
        to determine the number of statistically significant components. If 0 
        and `load_data` is specified, will attempt to load existing bootstrap
        results and use these to determine the significance cutoff. If -1, will 
        will skip bootstrapping and treat all components as significant.
    --load_data : (Optional) Path to existing output with SCA results, to load.

-------------------------------------------------------------------------------
OUTPUTS:

Core results are stored in a numpy archive file `scarun_results.npz`. Command 
line arguments are stored `scarun_args.json`. A subdirectory 
`sca_results` is created to store additional data. An `images` subdirectory is 
also created.

scarun_results.npz: (Computed SCA statistics)
    Dia : Conservation measure per position and amino acid $D_i^a$.
    conservation : Aggregated position-wise conservation measure $D_i$.
    sca_matrix : Weighted SCA covariance matrix $\\tilde{C}_{ij}$.
    phi_ia : Conservation weights $\\phi_i^a$.
    fi0 : Gap frequency at each position $f_i^0$.
    fia : Position-wise amino acid frequencies $f_i^a$.

scarun_args.json
    Mapping from command line SCA parameters to their values.

t_dists_info.json
    Information pertaining to empirical distributions made to call statistical 
    sectors.

sector_idxs.npy:
    Integer array containing values [0, ..., K-1] where K is the number of 
    identified statistical sectors.
    
sectors.npz:
    sector_{i} : Positional indices defining sector i, wrt the input, PROCESSED 
        MSA, for i in {0, ..., K-1}.

sectors_msaorig.npz:
    sector_{i} : Positional indices defining sector i, wrt the ORIGINAL MSA,
        for i in {0, ..., K-1}.

sectored_sequences.npz:
    group_{gidx}_{id} : 

statsectors_seq.npz:

sca_results/
    Subdirectory for additional results.
    
images/
    Subdirectory for generated images.

-------------------------------------------------------------------------------
EXAMPLE USAGE:

sca-core -i </path/to/preprocessed/data> -o </path/to/outdir> \
    --regularization 0.03 --background </path/to/background.json>

sca-core -i </path/to/preprocessed/data> -o </path/to/outdir> \
    --regularization 0.03 --load_data </path/to/existing/data>

"""

import argparse
import os, sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import tqdm as tqdm
import json
from Bio import AlignIO

import scipy
from scipy import sparse
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform

from mysca.run_preprocessing import OUTPUT_RESULTS_FNAME as PREP_RESULTS_FNAME
from mysca.run_preprocessing import OUTPUT_SYMMAP_FNAME as PREP_SYMMAP_FNAME
from mysca.run_preprocessing import OUTPUT_MSAORIG_FNAME as PREP_MSAORIG_FNAME
from mysca.core import run_sca, run_ica
from mysca.helpers import get_rawseq_positions_in_groups
from mysca.helpers import get_rawseq_scores_in_groups
from mysca.helpers import get_group_rawseq_positions_by_entry
from mysca.helpers import get_group_rawseq_scores_by_entry
from mysca.helpers import get_rawseq_indices_of_msa
from mysca.constants import SECTOR_COLORS, DEFAULT_BACKGROUND_FREQ

from mysca.pl.plotting import plot_sequence_similarity, plot_dendrogram
from mysca.pl.plotting import plot_t_distributions, plot_data_2d, plot_data_3d

OUTPUT_RESULTS_FNAME = "scarun_results.npz"
OUTPUT_ARGS_FNAME = "scarun_args.json"
OUTPUT_STATSECTORS_MSA_FNAME = "statsectors_msa.npz"
OUTPUT_STATSECTORS_SEQ_FNAME = "statsectors_seq.npz"
EVALS_SHUFF_SAVEAS = "evals_shuff.npy"


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", type=str, required=True,
                        help="Path to preprocessed data.")
    parser.add_argument("-o", "--outdir", type=str, required=True, 
                        help="Output directory.")
    parser.add_argument("--pbar", action="store_true")
    parser.add_argument("-v", "--verbosity", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    
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
    sca_params.add_argument("-p", "--pstar", type=int, default=95, 
                    help="Percentile defining IC groups.")
    sca_params.add_argument("--weak_assignment", type=int, nargs="*", 
                        default=[])

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
    weak_assignment = args.weak_assignment

    regularization = args.regularization
    background_freq = args.background
    kstar = args.kstar
    pstar = args.pstar

    if N_BOOT < 0:
        N_BOOT = 0
    
    ####################
    ##  Housekeeping  ##
    ####################

    printv = get_printv(verbosity)
    
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
        if verbosity:
            print(f"Loading background frequencies from file: {background_freq}")
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
    with open(os.path.join(indir, PREP_SYMMAP_FNAME), "rb") as f:
        sym_map = json.load(f)
    
    preprocessed_results = np.load(os.path.join(indir, PREP_RESULTS_FNAME))
    msa = preprocessed_results["msa"]
    retained_sequences = preprocessed_results["retained_sequences"]
    retained_positions = preprocessed_results["retained_positions"]
    weights = preprocessed_results["sequence_weights"]
    
    msa_binary3d = sparse.load_npz(
        os.path.join(indir, "msa_binary2d_sp.npz")
    ).toarray().reshape([
        len(retained_sequences), len(retained_positions), -1
    ])
    NSYMS = msa_binary3d.shape[-1]

    msa_obj_orig = AlignIO.read(
        os.path.join(indir, PREP_MSAORIG_FNAME), "fasta"
    )
    NUM_POS_ORIG = msa_obj_orig.get_alignment_length()
    
    # Create subdirectories within the specified output directory.
    SCADIR = os.path.join(OUTDIR, "sca_results")
    IMGDIR = os.path.join(OUTDIR, "images")
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(SCADIR, exist_ok=True)
    os.makedirs(IMGDIR, exist_ok=True)

    # Create the background frequency distribution q
    printv("Background frequencies:")
    printv("  ", ", ".join([
        f"{k}: {background_freq[k]:.3g}" 
        for k in np.sort(list(background_freq.keys()))
    ]))
    background_freq_array = np.zeros(len(background_freq))
    for a in background_freq:
        background_freq_array[sym_map[a]] = background_freq[a]    
    background_freq_array = background_freq_array / background_freq_array.sum()
    
    # Run SCA
    if verbosity:
        print("Running SCA...")
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
            verbosity=verbosity,
        )
        Dia = sca_results["Dia"]
        Di = sca_results["Di"]
        Cij_raw = sca_results["Cij_raw"]
        Cij = sca_results["Cij_corr"]

        # Save SCA results
        tosave = {
            "Dia": Dia,
            "conservation": Di,
            "sca_matrix": Cij,
            "phi_ia": sca_results["phi_ia"],
            "fi0": sca_results["fi0"],
            "fia": sca_results["fia"],
        }
        if SAVE_ALL:
            tosave.update({
                "Cijab_raw": sca_results["Cijab_raw"],
                "fijab": sca_results["fijab"],
            })
        np.savez_compressed(
            os.path.join(OUTDIR, OUTPUT_RESULTS_FNAME),
            **tosave
        )        
        del sca_results  # relieve memory
    else:
        existing_results = np.load(os.path.join(LOAD_DATA, OUTPUT_RESULTS_FNAME))
        # Di = np.load(os.path.join(LOAD_DATA, "conservation.npy"))
        # Cij = np.load(os.path.join(LOAD_DATA, "sca_matrix.npy"))
        Di = existing_results["conservation"]
        Cij = existing_results["sca_matrix"]
        Cij_raw = None

    # Eigendecomposition of SCA matrix
    evals_sca, evecs_sca = np.linalg.eigh(Cij)
    evals_sca = np.flip(evals_sca)
    evecs_sca = np.flip(evecs_sca, axis=1)

    printv("Eigenvalue spectrum of SCA Matrix: {:.3g}, {:.3g}".format(
        evals_sca.min(), evals_sca.max()
    ))
    
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
            xmsa_shuff = np.eye(NSYMS + 1, dtype=bool)[msa_shuff][:,:,:-1]
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
        evals_shuff_fpath_toload = os.path.join(LOAD_DATA, EVALS_SHUFF_SAVEAS)
        printv("Skipping bootstrap. Loading existing null evals at: {}".format(
                evals_shuff_fpath_toload
        ))
        evals_shuff = np.load(evals_shuff_fpath_toload)
        N_BOOT = evals_shuff.shape[0]
    elif os.path.isfile(evals_shuff_fpath):
        printv("Skipping bootstrap. Loading existing null evals at: {}".format(
            evals_shuff_fpath
        ))
        evals_shuff = np.load(evals_shuff_fpath)
        # Determine the bootstrap size from the loaded data
        N_BOOT = evals_shuff.shape[0]
    else:
        evals_shuff = []
        N_BOOT = 0
        printv("Skipping bootstrap. No existing eigenvalue data found.")
    
    # Determine kstar, the number of significant eigenvalues. See SI G of [1]
    cutoff = np.mean(evals_shuff[:,1]) + 2 * np.std(evals_shuff[:,1])
    kstar_id = np.sum(evals_sca > cutoff)
    printv("significant eigenvalue cutoff:", cutoff)
    printv(f"Identified {kstar_id} significant eigenvalues:\n", evals_sca[:kstar_id])
    if kstar <= 0:
        kstar = kstar_id
        printv(f"Setting kstar={kstar}")
    else:
        kstar = min(kstar, len(evals_sca))
        printv(f"Overriding kstar from command line input!")
        printv(f"Setting kstar={kstar}")
    
    if kstar == 0:
        msg = f"No significant eigenvalues (kstar=0). Proceeding with kstar=1"
        warnings.warn(msg)
        kstar = 1

    # Consider top kstar values, excluding top value
    sig_evals_sca = evals_sca[:kstar]
    sig_evecs_sca = evecs_sca[:,:kstar]

    # Save kstar, full eigendecomp, and significant eigendecomp
    # TODO: save as a consolidated file bootstrap_results.npz
    np.savetxt(f"{SCADIR}/kstar_identified.txt", [kstar_id], fmt="%d")
    np.savetxt(f"{SCADIR}/kstar.txt", [kstar], fmt="%d")
    np.savetxt(f"{SCADIR}/eigenvalue_cutoff.txt", [cutoff])
    np.save(f"{SCADIR}/all_evals_sca.npy", evals_sca)
    np.save(f"{SCADIR}/all_evecs_sca.npy", evecs_sca)
    np.save(f"{SCADIR}/significant_evals_sca.npy", sig_evals_sca)
    np.save(f"{SCADIR}/significant_evecs_sca.npy", sig_evecs_sca)

    tosave = {
        "evals_sca": evals_sca,
        "evecs_sca": evecs_sca,
        "significant_evals_sca": sig_evals_sca,
        "significant_evecs_sca": sig_evecs_sca,
    }
    np.savez_compressed(
        os.path.join(OUTDIR, "sca_eigendecomp.npz"),
        **tosave
    )
    
    # Apply Independent Component Analysis (ICA)
    ica_rho = 1e-1
    ica_tol = 1e-7
    ica_maxiter = 1E6
    ica_max_attempts = 5
    v_ica_normalized, _, w_ica = apply_ica(
        sig_evecs_sca, 
        rho=ica_rho, tol=ica_tol, maxiter=ica_maxiter, 
        max_attempts=ica_max_attempts, 
        verbosity=verbosity,
    )
    np.save(os.path.join(SCADIR, "v_ica_normalized.npy"), v_ica_normalized)
    np.save(os.path.join(SCADIR, "w_ica.npy"), w_ica)

    # Fit t-distribution to each IC
    t_dists_info, top_idxs = fit_t_distributions(v_ica_normalized, p=pstar)
    all_imp_idxs = np.concatenate(top_idxs, axis=0)
    all_imp_idxs_unique = np.unique(all_imp_idxs)
    printv(f"Identified {len(all_imp_idxs)} important positions (with repeats).")
    printv(f"Identified {len(all_imp_idxs_unique)} important positions (w/o repeats).")
    
    np.save(
        os.path.join(SCADIR, "all_important_positions.npy"), 
        all_imp_idxs_unique
    )
    with open(os.path.join(SCADIR, "t_dists_info.json"), "w") as f:
        json.dump(t_dists_info, f)
    
    # Call statistical sectors, i.e. groups of "co-evolving" positions. 
    # Define groups from top p% empirical distribution.
    # TODO: Compartmentalize this section and test.
    groups = []
    group_scores = []
    for i, idx_set in enumerate(top_idxs):
        group = []
        group_score = []
        for idx in idx_set:
            if np.sum(all_imp_idxs == idx) == 1:
                # Position is uniquely assigned to a group
                group.append(idx)
                group_score.append(v_ica_normalized[idx,i])
            elif np.sum(all_imp_idxs == idx) > 1:
                # Position is not uniquely assigned to a group.
                # Assign to group only if projection onto ith IC is maximal
                screen = ~np.isin(
                    np.arange(v_ica_normalized.shape[1]), weak_assignment
                )
                if np.all(v_ica_normalized[idx,i] >= v_ica_normalized[idx,screen]):
                    group.append(idx)
                    group_score.append(v_ica_normalized[idx,i])
            else:
                raise RuntimeError("Index should be found amoung all...")
        groups.append(np.array(group, dtype=int))
        group_scores.append(np.array(group_score))
    
    # Subset the SCA matrix into grouped important positions
    group_idxs_all = np.concatenate(groups, axis=0)
    sca_mat_imp = Cij[group_idxs_all,:]
    sca_mat_imp = sca_mat_imp[:,group_idxs_all]
    np.save(os.path.join(SCADIR, "sca_matrix_sector_subset.npy"), sca_mat_imp)

    # Save groups and group_scores in MSA coordinates
    subdir1 = os.path.join(OUTDIR, "groups")
    subdir2 = os.path.join(SCADIR, "msa_sectors")
    os.makedirs(subdir1, exist_ok=True)
    os.makedirs(subdir2, exist_ok=True)
    for i in range(len(groups)):
        np.save(os.path.join(subdir1, f"group_{i}_msapos.npy"), groups[i])
        np.save(os.path.join(subdir2, f"sector_{i}_msapos.npy"), groups[i])
        np.save(os.path.join(subdir2, f"sector_{i}_scores.npy"), group_scores[i])
    # As a single file:
    msapos_to_groupidx = np.vstack([
        group_idxs_all,
        np.concatenate([len(g) * [i] for i, g in enumerate(groups)], axis=0)
    ])
    np.save(os.path.join(SCADIR, "msapos_to_groupidx.npy"), msapos_to_groupidx)

    # Map processed MSA positions to original sequence positions
    rawseq_idxs = get_rawseq_indices_of_msa(msa_obj_orig)
    rawseq_idxs = rawseq_idxs[retained_sequences,:]
    rawseq_idxs = rawseq_idxs[:,retained_positions]
    
    # Save residue groups by raw sequence position
    group_rawseq_positions = get_rawseq_positions_in_groups(
        rawseq_idxs, groups
    )
    group_rawseq_scores = get_rawseq_scores_in_groups(
        rawseq_idxs, groups, group_scores
    )
    group_rawseq_positions_by_entry = get_group_rawseq_positions_by_entry(
        msa_obj_orig, retained_sequences, groups, group_rawseq_positions
    )
    group_rawseq_scores_by_entry = get_group_rawseq_scores_by_entry(
        msa_obj_orig, retained_sequences, groups, group_rawseq_scores
    )
    msa_stat_sectors_data = {}
    pdb_stat_sectors_data = {}
    for gidx in range(len(groups)):
        for i, seqidx in enumerate(retained_sequences):
            entry = msa_obj_orig[int(seqidx)]
            id = entry.id
            group_arr = group_rawseq_positions_by_entry[id][gidx]
            group_scores_arr = group_rawseq_scores_by_entry[id][gidx]
            msa_stat_sectors_data[f"group_{gidx}_{id}"] = group_arr
            pdb_stat_sectors_data[f"sector_{gidx}_pdbpos_{id}"] = group_arr
            pdb_stat_sectors_data[f"sector_{gidx}_scores_{id}"] = group_scores_arr
    
    np.savez_compressed(
        os.path.join(OUTDIR, OUTPUT_STATSECTORS_MSA_FNAME),
        **msa_stat_sectors_data
    )

    np.savez_compressed(
        os.path.join(OUTDIR, OUTPUT_STATSECTORS_SEQ_FNAME),
        **pdb_stat_sectors_data
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

    printv("Done!")


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
        verbosity=1,
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
            # ICA failed to converge
            if verbosity:
                msg = f"ICA did not converge with parameters rho={rho:3g}, " + \
                        f"tol={tol:.3g}, maxiter={maxiter}. " + \
                        f"(Reached tol={ica_delta:.3})"
                print(msg)
            maxiter *= 2
            rho /= 2
        else:
            # ICA succeeded
            v_ica = sig_evecs_sca @ w_ica.T
            if verbosity:
                print(f"ICA succeeded after {n_attempts} attempts. (tol={tol:.2g})")
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


def get_groups(v, p=95, method="t-dist"):
    groups = []
    to_be_assigned = np.ones(len(v), dtype=bool)
    for i in range(v.shape[1]):
        if method == "ecdf":
            screen = v[:,i] >= np.percentile(v[to_be_assigned,i], p)
        elif method == "t-dist":
            df, loc, scale = scipy.stats.t.fit(v[:,i])
            cutoff = scipy.stats.t.ppf(p/100, df, loc=loc, scale=scale)
            screen = v[:,i] >= cutoff
        else:
            raise RuntimeError(f"Unknown method `{method}` for group calling.")
        top_p_idxs = np.where(screen & to_be_assigned)[0]
        to_be_assigned[top_p_idxs] = False
        groups.append(top_p_idxs)
    return groups


def get_printv(verbosity=1) -> callable:
    """Print wrapper to print only at a given verbosity level."""
    def printv(*args, v=1, **kwargs):
        if v <= verbosity:
            print(*args, **kwargs)
    return printv


def shuffle_columns(m, rng=None):
    rng = np.random.default_rng(rng)
    r, c = m.shape
    idx = np.argsort(rng.random((r, c)), axis=0)
    return m[idx, np.arange(c)]


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
    
    # Plot conservation
    fig, ax = plt.subplots(1, 1, figsize=(10,4))
    ax.plot(
        retained_positions, Di, "o",
        color="Blue",
        alpha=0.2
    )
    ax.set_xlim(0, NUM_POS_ORIG)
    ax.set_xlabel(f"Position")
    ax.set_ylabel("Relative Entropy $D_i$")
    ax.set_title(f"Conservation")
    plt.savefig(f"{IMGDIR}/top_conservation.png")
    plt.close()

    # Plot conservation as a bar graph
    fig, ax = plt.subplots(1, 1, figsize=(10,4))
    ax.bar(
        retained_positions, Di,
        color="Blue",
        width=1.0,
        align="center",
    )
    ax.set_xlim(0, NUM_POS_ORIG)
    ax.set_xlabel(f"Position")
    ax.set_ylabel("Relative Entropy $D_i$")
    ax.set_title(f"Conservation")
    plt.savefig(f"{IMGDIR}/positional_conservation.png")
    plt.close()

    # Plot conservation as a bar graph, without mapping to original positions
    fig, ax = plt.subplots(1, 1, figsize=(10,4))
    ax.bar(
        np.arange(len(Di)), Di,
        color="Blue",
        width=1.0,
        align="center",
    )
    ax.set_xlabel(f"Position")
    ax.set_ylabel("Relative Entropy $D_i$")
    ax.set_title(f"Conservation")
    plt.savefig(f"{IMGDIR}/conservation.png")
    plt.close()

    # Plot sequence similarity
    if DENDRO:
        plot_sequence_similarity(
            msa_binary3d, IMGDIR,
        )


    # Plot Covariance Matrix
    if Cij_raw is not None:
        fig, ax = plt.subplots(1, 1)
        sc = ax.imshow(
            Cij_raw, 
            cmap="Blues", 
            origin="lower",
            interpolation="none",
            vmax=None,
        )
        fig.colorbar(sc, label="Covariation")
        ax.set_xlabel("(Retained) Position i")
        ax.set_ylabel("(Retained) Position j")
        ax.set_title("Covariance Matrix")
        plt.savefig(f"{IMGDIR}/covariance_matrix.png")
        plt.close()

    # Plot SCA Matrix
    fig, ax = plt.subplots(1, 1)
    sc = ax.imshow(
        Cij, 
        cmap="Blues", 
        origin="lower",
        interpolation="none",
        vmax=None,
    )
    fig.colorbar(sc, label="Covariation")
    ax.set_xlabel("(Retained) Position i")
    ax.set_ylabel("(Retained) Position j")
    ax.set_title("SCA Matrix")
    plt.savefig(f"{IMGDIR}/sca_matrix.png")
    plt.close()


    # Plot SCA matrix spectrum null vs data
    fig, ax = plt.subplots(1, 1)
    for e in evals_shuff:
        ax.plot(
            1 + np.arange(len(e)), e, ".",
            markersize=3
        )
    ax.plot(
        1 + np.arange(len(evals_sca)), evals_sca,
        "k.",
        markersize=2,
        label="data",
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel(f"$\\lambda$ index")
    ax.set_ylabel(f"$\\lambda$")
    ax.set_title(f"$\\tilde{{C}}_{{ij}}$ Spectrum (data vs null)")
    plt.savefig(f"{IMGDIR}/sca_matrix_spectrum.png")
    plt.close()


    # Plot eigenvalue distribution null vs data
    fig, ax = plt.subplots(1, 1)
    # Histogram of data eigenvalues
    counts, bins, patches = ax.hist(
        evals_sca, bins=100, color="black", alpha=0.8, log=True, label="Data"
    )
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    h, bin_edges = np.histogram(evals_shuff.flatten(), bins=bins)
    ax.axvline(cutoff, 0, 1, linestyle="--", color="grey")
    ax.plot(
        bin_centers, h / N_BOOT, 
        color="red", 
        lw=1.5, 
        label="Null"
    )
    ax.legend()
    ax.set_xlabel(f"$\\lambda$")
    ax.set_ylabel(f"Count")
    ax.set_title(f"Spectral decomposition")
    plt.savefig(f"{IMGDIR}/sca_matrix_spectrum_vs_null.png")
    plt.close()


    # Dendrogram of SCA matrix
    if DENDRO:
        plot_dendrogram(Cij, nclusters=kstar, imgdir=IMGDIR)


    # Plot t-distributions
    plot_t_distributions(v_ica_normalized, t_dists_info, IMGDIR)



    # Plot data and groups in EV coords (2-dimensional)
    EVIDXS_AND_GROUP_IDXS = [  # ((EVi, EVj), [group_indices])
        ((0, 1), "all"),
        ((1, 2), "all"),
        ((2, 3), "all"),
        ((3, 4), "all"),
        ((4, 5), "all"),
        ((5, 6), "all"),
        ((0, 1), [0, 1, 2]),
        ((1, 2), [0, 1, 2]),
    ]
    for evidxs, group_idxs in EVIDXS_AND_GROUP_IDXS:
        plot_data_2d(
            "ev", evidxs, group_idxs, groups, sig_evecs_sca, IMGDIR,
        )
    
    # Plot data and groups in EV coords (3-dimensional)
    EVIDXS_AND_GROUP_IDXS = [  # ((EVi, EVj, EVk), [group_indices])
        ((0, 1, 2), "all"),
        ((1, 2, 3), "all"),
        ((0, 1, 2), [0, 1, 2]),
        ((1, 2, 3), [0, 1, 2]),
    ]
    for evidxs, group_idxs in EVIDXS_AND_GROUP_IDXS:
        plot_data_3d(
            "ev", evidxs, group_idxs, groups, sig_evecs_sca, IMGDIR,
        )
    
    # Plot data and groups in IC coords (2-dimensional)
    ICIDXS_AND_GROUP_IDXS = [  # ((ICi, ICj), [group_indices])
        ((0, 1), "all"),
        ((1, 2), "all"),
        ((2, 3), "all"),
        ((3, 4), "all"),
        ((4, 5), "all"),
        ((5, 6), "all"),
        ((0, 1), [0, 1, 2]),
        ((1, 2), [0, 1, 2]),
    ]
    for icidxs, group_idxs in ICIDXS_AND_GROUP_IDXS:
        plot_data_2d(
            "ic", icidxs, group_idxs, groups, v_ica_normalized, IMGDIR,
        )
    
    # Plot data and groups in IC coords (3-dimensional)
    ICIDXS_AND_GROUP_IDXS = [  # ((ICi, ICj, ICk), [group_indices])
        ((0, 1, 2), "all"),
        ((1, 2, 3), "all"),
        ((0, 1, 2), [0, 1, 2]),
        ((1, 2, 3), [0, 1, 2]),
    ]
    for icidxs, group_idxs in ICIDXS_AND_GROUP_IDXS:
        plot_data_3d(
            "ic", icidxs, group_idxs, groups, v_ica_normalized, IMGDIR,
        )

    # Plot SCA Matrix "Important" subset
    fig, ax = plt.subplots(1, 1)
    sc = ax.imshow(
        sca_mat_imp, 
        cmap="Blues", 
        origin="lower",
        interpolation="none",
        vmax=None,
    )
    fig.colorbar(sc, label="Covariation")
    ax.set_xlabel("(Important) Position i")
    ax.set_ylabel("(Important) Position j")
    ax.set_title("SCA Matrix (Groups)")

    # Add sector divisions if specified
    group_lengths = [len(g) for g in groups]
    if sector_color_set and np.sum(group_lengths) > 0:
        group_colors = np.concatenate([
            len(g) * [colors.to_rgb(sector_color_set[i])] 
            for i, g in enumerate(groups) if len(g) > 0
        ], axis=0)       
        
        divider = make_axes_locatable(ax)
        # Top rug
        ax_top = divider.append_axes("top", size="2%", pad=0.0, sharex=ax)
        ax_top.imshow(
            group_colors[None,:,:], 
            aspect="auto", 
            extent=(0, len(group_colors), 0, 1)
        )
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_title(ax.get_title())
        ax.set_title("")
        # Right rug
        ax_right = divider.append_axes("right", size="2%", pad=0.0, sharey=ax)
        ax_right.imshow(
            np.flip(group_colors, axis=0)[:,None,:], 
            aspect="auto", 
            extent=(0, 1, 0, len(group_colors))
        )
        ax_right.set_xticks([])        
        ax_right.set_yticks([])

    plt.savefig(f"{IMGDIR}/sca_matrix_important_subset.png")
    plt.close()

    return
    

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
