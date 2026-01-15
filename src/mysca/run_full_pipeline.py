"""SCA pipeline

See references:
    [1] SI to Rivoire et al., 2016

"""

import argparse
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import tqdm as tqdm
import json

import scipy
from scipy import sparse
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform

from mysca.io import load_msa
from mysca.preprocess import preprocess_msa
from mysca.preprocess import compute_background_freqs
from mysca.core import run_sca, run_ica
from mysca.helpers import get_top_k_conserved_retained_positions
from mysca.helpers import get_rawseq_positions_in_groups
from mysca.helpers import get_rawseq_scores_in_groups
from mysca.helpers import get_group_rawseq_positions_by_entry
from mysca.helpers import get_group_rawseq_scores_by_entry
from mysca.helpers import get_rawseq_indices_of_msa
from mysca.constants import SECTOR_COLORS

from mysca.pl import plot_sequence_similarity, plot_dendrogram

DEFAULT_BACKGROUND_FREQ = {
        'A': 0.073, 'C': 0.025, 'D': 0.050, 'E': 0.061,
        'F': 0.042, 'G': 0.072, 'H': 0.023, 'I': 0.053,
        'K': 0.064, 'L': 0.089, 'M': 0.023, 'N': 0.043,
        'P': 0.052, 'Q': 0.040, 'R': 0.052, 'S': 0.073,
        'T': 0.056, 'V': 0.063, 'W': 0.013, 'Y': 0.033,
    }


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-msa", "--msa_fpath", type=str, required=True,
                        help="Filepath of input MSA in fasta format.")
    parser.add_argument("-o", "--outdir", type=str, required=True, 
                        help="Output directory.")
    parser.add_argument("--pbar", action="store_true")
    parser.add_argument("-v", "--verbosity", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--nodendro", action="store_true", 
                        help="Skip dendrogram plots")
    parser.add_argument("--use_jax", action="store_true", 
                        help="Use JAX in computations.")
    parser.add_argument("--save_all", action="store_true", 
                        help="Save all SCA results (includes large files).")
    parser.add_argument("--load_data", type=str, default="", 
                        help="SCA directory to load precomputed data from")
    parser.add_argument("--sector_cmap", type=str, default="default",
                        choices=["none", "default"])
    
    sca_params = parser.add_argument_group("SCA parameters")
    sca_params.add_argument("--gap_truncation_thresh", type=float, default=0.4,
                            help="SCA parameter gap_truncation_thresh")
    sca_params.add_argument("--sequence_gap_thresh", type=float, default=0.2,
                            help="SCA parameter sequence_gap_thresh γ_{seq}")
    sca_params.add_argument("--reference", type=str, default=None, 
                            help="SCA optional reference entry in MSA")
    sca_params.add_argument("--reference_similarity_thresh", type=float, default=0.2,
                            help="SCA parameter reference_similarity_thresh Δ")
    sca_params.add_argument("--sequence_similarity_thresh", type=float, default=0.8,
                            help="SCA parameter sequence_similarity_thresh δ")
    sca_params.add_argument("--position_gap_thresh", type=float, default=0.2,
                            help="SCA parameter position_gap_thresh γ_{pos}")
    sca_params.add_argument("--regularization", type=float, default=0.03,
                            help="SCA regularization parameter λ")
    sca_params.add_argument("--background", type=str, default=None,
                            help="Path to file describing background frequency." \
                            " If None, use default.")
    sca_params.add_argument("-nc", "--n_top_conserved", type=int, required=True, 
                            help="Number of top conserved residues to consider.")
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
    MSA_FPATH = args.msa_fpath
    reference_id = args.reference
    OUTDIR = args.outdir
    verbosity = args.verbosity
    n_top_conserved = args.n_top_conserved
    N_BOOT = args.n_boot
    PBAR = args.pbar
    SEED = args.seed
    DENDRO = not args.nodendro
    LOAD_DATA = args.load_data
    USE_JAX = args.use_jax
    SAVE_ALL = args.save_all
    sector_cmap = args.sector_cmap
    weak_assignment = args.weak_assignment

    gap_truncation_thresh = args.gap_truncation_thresh
    sequence_gap_thresh = args.sequence_gap_thresh
    reference_id = args.reference
    reference_similarity_thresh = args.reference_similarity_thresh
    sequence_similarity_thresh = args.sequence_similarity_thresh
    position_gap_thresh = args.position_gap_thresh
    regularization = args.regularization
    background_freq = args.background
    kstar = args.kstar
    pstar = args.pstar
    
    # Housekeeping
    if SEED is None or SEED <= 0:
        SEED = np.random.randint(2**32)
    rng = np.random.default_rng(seed=SEED)

    if reference_id is None or reference_id.lower() == "none":
        if verbosity:
            print("No reference entry specified.")
        reference_id = None

    do_compute_background = False
    if isinstance(background_freq, str) and background_freq.lower() == "default":
        background_freq = DEFAULT_BACKGROUND_FREQ
    elif background_freq is None or (
            isinstance(background_freq, str) and background_freq.lower() == "none"
    ):
        # Mark to compute background frequency from MSA
        do_compute_background = True
        background_freq = None
    elif isinstance(background_freq, str):
        if verbosity:
            print(f"Loading background frequencies: {background_freq}")
        background_freq = load_background(background_freq)
    else:
        msg = f"Cannot handle given argument for background: {background_freq}"
        raise RuntimeError(msg)
    
    sector_color_set = {
        "default": SECTOR_COLORS,
        "none": None,
    }[sector_cmap]

    SCADIR = os.path.join(OUTDIR, "sca_results")
    IMGDIR = os.path.join(OUTDIR, "images")
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(SCADIR, exist_ok=True)
    os.makedirs(IMGDIR, exist_ok=True)

    # Load MSA
    msa_obj_orig, msa_orig, seqids_orig, sym_map = load_msa(
        MSA_FPATH, format="fasta", 
        mapping=None,  # TODO: consider allowing for specified mapping
        verbosity=1
    )
    _, NUM_POS_ORIG = msa_orig.shape
    NSYMS = len(sym_map)
    
    if verbosity:
        print(f"Loaded MSA. shape: {msa_orig.shape} (sequences x positions)")
        print(f"Symbols: {sym_map.aa_list}")

    # Preprocessing
    msa, preprocessing_results = preprocess_msa(
        msa_orig, seqids_orig, 
        mapping=sym_map,
        gap_truncation_thresh=gap_truncation_thresh,
        sequence_gap_thresh=sequence_gap_thresh,
        reference_id=reference_id,
        reference_similarity_thresh=reference_similarity_thresh,
        sequence_similarity_thresh=sequence_similarity_thresh,
        position_gap_thresh=position_gap_thresh,
        use_pbar=PBAR,
        verbosity=1,
    )

    msa_binary3d = preprocessing_results["msa_binary3d"]
    retained_sequences = preprocessing_results["retained_sequences"]
    retained_positions = preprocessing_results["retained_positions"]
    seqids = preprocessing_results["retained_sequence_ids"]
    weights = preprocessing_results["sequence_weights"]
    fi0_pretrunc = preprocessing_results["fi0_pretruncation"]
    preprocessing_args = preprocessing_results["args"]

    np.save(f"{SCADIR}/retained_sequences.npy", retained_sequences)
    np.save(f"{SCADIR}/retained_positions.npy", retained_positions)
    np.save(f"{SCADIR}/retained_sequence_ids.npy", seqids)
    np.save(f"{SCADIR}/fi0_pretrunc.npy", fi0_pretrunc)
    np.save(f"{SCADIR}/sequence_weights.npy", weights)
    np.save(f"{SCADIR}/msa.npy", msa)
    np.savetxt(f"{SCADIR}/position_gap_thresh.txt", [position_gap_thresh])
    np.savetxt(f"{SCADIR}/npos_original.txt", [NUM_POS_ORIG], fmt="%d")

    sparse.save_npz(
        f"{SCADIR}/Xsp.npz", 
        sparse.csr_matrix(msa_binary3d.reshape([msa_binary3d.shape[0], -1]))
    )
    with open(f"{SCADIR}/sym2int.json", "w") as f:
        json.dump(sym_map.sym2int, f)

    # Plot gap frequency by position
    fig, ax = plt.subplots(1, 1)
    ax.plot(fi0_pretrunc, ".")
    ax.hlines(
        position_gap_thresh, *ax.get_xlim(), 
        linestyle='--', 
        color="r", 
        label="cutoff"
    )
    ax.legend()
    ax.set_xlim(0, 10 + msa.shape[1])
    ax.set_xlabel(f"position")
    ax.set_ylabel(f"gap frequency")
    ax.set_title(f"Gap frequency by position")
    plt.savefig(f"{IMGDIR}/gap_freq_by_position.png")
    plt.close()

    # Compute the background frequencies if needed and store as an array
    if do_compute_background:
        if verbosity:
            print("Computing background frequency from full MSA")
        background_freq = compute_background_freqs(msa_obj_orig, gapstr="-")
    if verbosity:
        print("Background frequencies:")
        print("  ", ", ".join([
            f"{k}: {background_freq[k]:.3g}" 
            for k in np.sort(list(background_freq.keys()))
        ]))
    background_freq_array = np.zeros(len(background_freq))
    for a in background_freq:
        background_freq_array[sym_map[a]] = background_freq[a]    
    background_freq_array = background_freq_array / background_freq_array.sum()

    # Plot sequence similarity
    if DENDRO:
        plot_sequence_similarity(
            msa_binary3d, IMGDIR,
        )
    
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
        np.save(f"{SCADIR}/Dia.npy", Dia)
        np.save(f"{SCADIR}/conservation.npy", Di)
        np.save(f"{SCADIR}/sca_matrix.npy", Cij)
        np.save(f"{SCADIR}/phi_ia.npy", sca_results["phi_ia"])
        np.save(f"{SCADIR}/fi0.npy", sca_results["fi0"])
        np.save(f"{SCADIR}/fia.npy", sca_results["fia"])
        if SAVE_ALL:
            np.save(f"{SCADIR}/Cijab_raw.npy", sca_results["Cijab_raw"])
        
        del sca_results  # relieve memory
    else:
        Di = np.load(f"{LOAD_DATA}/conservation.npy")
        Cij_raw = None
        Cij = np.load(f"{LOAD_DATA}/sca_matrix.npy")

    
    
    # Determine the top conserved positions
    topk_conserved_msa_pos, top_conserved_Di = get_top_k_conserved_retained_positions(
        retained_positions, Di, n_top_conserved
    )
    np.save(f"{SCADIR}/topk_conserved_msa_pos.npy", topk_conserved_msa_pos)
    np.save(f"{SCADIR}/top_conserved_Di.npy", top_conserved_Di)

    if verbosity:
        print("top k conserved MSA positions:", topk_conserved_msa_pos)

    # Plot conservation
    fig, ax = plt.subplots(1, 1, figsize=(10,4))
    ax.plot(
        retained_positions, Di, "o",
        color="Blue",
        alpha=0.2
    )
    ax.plot(
        topk_conserved_msa_pos, top_conserved_Di, "o",
        color="Green",
        alpha=0.5
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

    # Eigendecomposition of SCA matrix
    # TODO: Potential issue?
    evals_sca, evecs_sca = np.linalg.eigh(Cij)
    evals_sca = np.flip(evals_sca)
    evecs_sca = np.flip(evecs_sca, axis=1)

    if verbosity:
        print(f"Eigenvalue spectrum of SCA Matrix: " + 
            f"{evals_sca.min():.3g}, {evals_sca.max():.3f}")
    
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
    
    # Perform bootstrapping to get eigenvalue null distribution
    DO_SHUFFLING = N_BOOT > 0
    evals_shuff_saveas = "evals_shuff.npy"
    
    def shuffle_columns(m, rng=None):
        rng = np.random.default_rng(rng)
        r, c = m.shape
        idx = np.argsort(rng.random((r, c)), axis=0)
        return m[idx, np.arange(c)]

    evals_shuff = np.full([N_BOOT, *evals_sca.shape], np.nan)
    if DO_SHUFFLING:
        for iteridx in tqdm.trange(N_BOOT):
            msa_shuff = shuffle_columns(msa, rng=rng)
            xmsa_shuff = np.eye(NSYMS, dtype=bool)[msa_shuff][:,:,:-1]
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
        np.save(f"{SCADIR}/{evals_shuff_saveas}", evals_shuff)
    elif LOAD_DATA:
        if verbosity:
            print("Skipping bootstrap. Loading existing null evals at: {}".format(
                f"{LOAD_DATA}/{evals_shuff_saveas}"
            ))
        evals_shuff = np.load(f"{LOAD_DATA}/{evals_shuff_saveas}")
        N_BOOT = evals_shuff.shape[0]
    elif os.path.isfile(f"{SCADIR}/{evals_shuff_saveas}"):
        if verbosity:
            print("Skipping bootstrap. Loading existing null evals at: {}".format(
                f"{SCADIR}/{evals_shuff_saveas}"
            ))
        evals_shuff = np.load(f"{SCADIR}/{evals_shuff_saveas}")
        N_BOOT = evals_shuff.shape[0]
    else:
        evals_shuff = []
        if verbosity:
            print("Skipping bootstrap. No existing eigenvalue data found.")

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
    
    # Determine k^*, the number of significant eigenvalues. See SI G of [1]
    cutoff = np.mean(evals_shuff[:,1]) + 2 * np.std(evals_shuff[:,1])
    kstar_id = np.sum(evals_sca > cutoff)
    if verbosity:
        print("significant eigenvalue cutoff:", cutoff)
        print(f"Identified {kstar_id} significant eigenvalues:\n", 
              evals_sca[:kstar_id])
    if kstar <= 0:
        kstar = kstar_id
        if verbosity:
            print(f"Setting kstar={kstar}")
    else:
        kstar = min(kstar, len(evals_sca))
        if verbosity:
            print(f"Overriding kstar from command line input!")
            print(f"Setting kstar={kstar}")
    
    # Consider top kstar values, excluding top value
    sig_evals_sca = evals_sca[:kstar]
    sig_evecs_sca = evecs_sca[:,:kstar]

    # Save kstar, full eigendecomp, and significant eigendecomp
    np.savetxt(f"{SCADIR}/kstar_identified.txt", [kstar_id], fmt="%d")
    np.savetxt(f"{SCADIR}/kstar.txt", [kstar], fmt="%d")
    np.savetxt(f"{SCADIR}/eigenvalue_cutoff.txt", [cutoff])
    np.save(f"{SCADIR}/all_evals_sca.npy", evals_sca)
    np.save(f"{SCADIR}/all_evecs_sca.npy", evecs_sca)
    np.save(f"{SCADIR}/significant_evals_sca.npy", sig_evals_sca)
    np.save(f"{SCADIR}/significant_evecs_sca.npy", sig_evecs_sca)

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
    
    # Apply ICA
    rho = 1e-1
    tol = 1e-7
    v_ica_normalized, _, w_ica = apply_ica(
        sig_evecs_sca, 
        rho=rho, tol=tol, maxiter=1E6, 
        max_attempts=5, 
        verbosity=verbosity,
    )
    np.save(f"{SCADIR}/v_ica_normalized.npy", v_ica_normalized)
    np.save(f"{SCADIR}/w_ica.npy", w_ica)

    # Fit t-distribution to each IC
    t_dists_info, top_idxs = fit_t_distributions(
        v_ica_normalized, p=pstar
    )
    all_imp_idxs = np.concatenate(top_idxs, axis=0)
    if verbosity:
        print(f"Identified {len(all_imp_idxs)} important positions (with repeats).")
    all_imp_idxs_unique = np.unique(all_imp_idxs)
    if verbosity:
        print(f"Identified {len(all_imp_idxs_unique)} important positions (w/o repeats).")
    
    np.save(f"{SCADIR}/all_important_positions.npy", all_imp_idxs_unique)
    with open(f"{SCADIR}/t_dists_info.json", "w") as f:
        json.dump(t_dists_info, f)
    
    # Plot t-distributions
    plot_t_distributions(v_ica_normalized, t_dists_info, IMGDIR)
    
    # Get groups from top p% empirical distribution
    # groups = get_groups(v_ica_normalized, p=pstar, method="t-dist")
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
        groups.append(np.array(group))
        group_scores.append(np.array(group_score))

    # Subset the SCA matrix into grouped important positions
    group_idxs_all = np.concatenate(groups, axis=0)
    sca_mat_imp = Cij[group_idxs_all,:]
    sca_mat_imp = sca_mat_imp[:,group_idxs_all]
    np.save(f"{SCADIR}/sca_matrix_sector_subset.npy", sca_mat_imp)

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
    if sector_color_set:
        group_colors = np.concatenate([
            len(g) * [colors.to_rgb(sector_color_set[i])] for i, g in enumerate(groups)
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

    # Save groups and group_scores in MSA coordinates
    subdir1 = f"{OUTDIR}/groups"
    subdir2 = f"{SCADIR}/msa_sectors"
    os.makedirs(subdir1, exist_ok=True)
    os.makedirs(subdir2, exist_ok=True)
    for i in range(len(groups)):
        np.save(f"{subdir1}/group_{i}_msapos.npy", groups[i])
        np.save(f"{subdir2}/sector_{i}_msapos.npy", groups[i])
        np.save(f"{subdir2}/sector_{i}_scores.npy", group_scores[i])
    # As a single file:
    msapos_to_groupidx = np.vstack([
        group_idxs_all,
        np.concatenate([len(g) * [i] for i, g in enumerate(groups)], axis=0)
    ])
    np.save(f"{SCADIR}/msapos_to_groupidx.npy", msapos_to_groupidx)

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

    # Map MSA positions to raw sequence positions, then save
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
    for gidx in range(len(groups)):
        subdir1 = f"{OUTDIR}/sca_groups/group_{gidx}"
        subdir2 = f"{OUTDIR}/pdb_sectors/sector_{gidx}"
        os.makedirs(subdir1, exist_ok=True)
        os.makedirs(subdir2, exist_ok=True)
        for i, seqidx in enumerate(retained_sequences):
            entry = msa_obj_orig[int(seqidx)]
            id = entry.id
            group_arr = group_rawseq_positions_by_entry[id][gidx]
            group_scores_arr = group_rawseq_scores_by_entry[id][gidx]
            np.save(f"{subdir1}/group_{gidx}_{id}.npy", group_arr)
            np.save(f"{subdir2}/sector_{gidx}_pdbpos_{id}.npy", group_arr)
            np.save(f"{subdir2}/sector_{gidx}_scores_{id}.npy", group_scores_arr)
    
    if verbosity:
        print("Done!")


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


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
