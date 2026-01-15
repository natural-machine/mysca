import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

import scipy
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform


def plot_data_2d(
        ic_or_ev, axidxs, group_idxs, groups, 
        data, 
        imgdir,
):
    ic_or_ev = ic_or_ev.lower()
    if ic_or_ev.lower() == "ic":
        title = f"Groups in IC space"
    elif ic_or_ev.lower() == "ev":
        title = "Groups in EV space"
    else:
        raise RuntimeError("ic_or_ev should be `ic` or `ev`!")
    if group_idxs == "all":
        group_idxs = list(range(len(groups)))
    axi, axj = axidxs
    if axj >= data.shape[1]:
        return
    fig, ax = plt.subplots(1, 1)
    # ax.axis("equal")
    sc = ax.scatter(
        data[:,axi], data[:,axj],
        c='k', 
        alpha=0.2, 
        edgecolor='k',
    )
    for i, gidx in enumerate(group_idxs):
        if gidx >= len(groups):
            continue
        g = groups[gidx]
        ax.scatter(
            data[g,axi], data[g,axj],
            alpha=1, 
            edgecolor='k',
            label=f"group {gidx}",
        )
    ax.plot(0, 0, "ro")
    rx, ry = ax.get_xlim()[1], ax.get_ylim()[1]
    ax.plot([0, rx], [0, 0], "k-", alpha=0.5)
    ax.plot([0, 0], [0, ry], "k-", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel(f"{ic_or_ev.upper()} {axi}")
    ax.set_ylabel(f"{ic_or_ev.upper()} {axj}")
    ax.set_title(title)
    groupstr = "".join([str(i) for i in group_idxs])
    plt.tight_layout()
    plt.savefig(f"{imgdir}/{ic_or_ev}{axi}{axj}_groups_{groupstr}.png",
                bbox_inches="tight")
    plt.close()
    return


def plot_data_3d(
        ic_or_ev, axidxs, group_idxs, groups, 
        data, 
        imgdir,
):
    ic_or_ev = ic_or_ev.lower()
    if ic_or_ev.lower() == "ic":
        title = f"ICA and identified groups"
    elif ic_or_ev.lower() == "ev":
        title = ""
    else:
        raise RuntimeError("ic_or_ev should be `ic` or `ev`!")
    if group_idxs == "all":
        group_idxs = list(range(len(groups)))
    axi, axj, axk = axidxs
    if axk >= data.shape[1]:
        return
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111, projection='3d')
    # ax.axis("equal")
    sc = ax.scatter(
        data[:,axi], data[:,axj], data[:,axk], 
        c="k", 
        alpha=0.2, 
        edgecolor='k',
    )
    for i, gidx in enumerate(group_idxs):
        if gidx >= len(groups):
            continue
        g = groups[gidx]
        ax.scatter(
            data[g,axi], data[g,axj], data[g,axk], 
            alpha=1, 
            edgecolor='k',
            label=f"group {gidx}",
        )
    ax.plot(0, 0, "ro")
    rx, ry, rz = ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]
    ax.plot([0, rx], [0, 0], [0, 0], "k-", alpha=0.5)
    ax.plot([0, 0], [0, ry], [0, 0], "k-", alpha=0.5)
    ax.plot([0, 0], [0, 0], [0, rz], "k-", alpha=0.5)
    ax.view_init(elev=30, azim=40)   # elev ~ tilt, azim ~ around z
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel(f"{ic_or_ev.upper()} {axi}")
    ax.set_ylabel(f"{ic_or_ev.upper()} {axj}")
    ax.set_zlabel(f"{ic_or_ev.upper()} {axk}")
    ax.set_title(title)
    groupstr = "".join([str(i) for i in group_idxs])
    plt.tight_layout()
    plt.savefig(f"{imgdir}/{ic_or_ev}{axi}{axj}{axk}_groups_{groupstr}.png", 
                bbox_inches="tight")
    plt.close()
    return


def plot_dendrogram(
        Cij, *, 
        nclusters=10,
        imgdir,
):
    Z = sch.linkage(pdist(Cij, metric='euclidean'), method='ward')
    clusters = sch.fcluster(Z, t=nclusters, criterion='maxclust')
    dendro = sch.dendrogram(Z, no_plot=True)
    leaf_indices = dendro['leaves']
    cmap = plt.cm.turbo
    cluster_colors = [to_hex(cmap(i)) for i in np.linspace(0, 1, nclusters)]
    def color_func(link_idx):
        if link_idx < len(clusters):  # Only color leaf nodes
            return cluster_colors[clusters[link_idx] - 1]
        return "#000000"
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7, 6), 
        gridspec_kw={'width_ratios': [0.2, 1]}
    )
    sch.dendrogram(
        Z,
        orientation='left',
        ax=ax1,
    #    color_threshold=max(Z[-nclusters+1, 2], 0.1),
        link_color_func=color_func,
        above_threshold_color='k'
    )
    ax1.set_ylabel('Position', fontsize='x-large')
    ax1.set_xticks([])
    ax1.set_yticks([])
    rearranged_data = Cij[leaf_indices][:, leaf_indices]
    im = ax2.imshow(
        rearranged_data, 
        aspect='auto', 
        cmap='Blues',
        interpolation='none', 
        origin='lower', 
        # vmin=0, vmax=1,
    )
    boundaries = np.where(np.diff(clusters[leaf_indices]))[0]
    for b in boundaries:
        ax2.axhline(b + 0.5, color='black', linestyle='--')
        ax2.axvline(b + 0.5, color='black', linestyle='--')
    ax2.set_title('Clustering of Positions', fontsize='x-large')
    ax2.set_xlabel('Position', fontsize='x-large')
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{imgdir}/dendrogram.png", bbox_inches="tight")
    plt.close()
    return


def plot_sequence_similarity(
        xmsa, imgdir
):
    npos = xmsa.shape[1]
    xmsa = xmsa.argmax(axis=-1)  # conversion
    distances = pdist(xmsa, metric="hamming")
    similarities = 1 - distances
    similarity_matrix = 1 - squareform(distances)
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8,5))

    Z = sch.linkage(distances, method="complete", metric="hamming")
    dendro = sch.dendrogram(Z, no_plot=True)
    idxs = dendro["leaves"]
    
    ax1.hist(similarities, int(round(npos/2)))
    ax1.set_xlabel("Pairwise sequence identities")
    ax1.set_ylabel("Count")

    sc = ax2.imshow(
        similarity_matrix[np.ix_(idxs, idxs)],
        vmin=0, vmax=1,
        interpolation="none",
    )
    plt.colorbar(sc)
    plt.tight_layout()
    plt.savefig(f"{imgdir}/sequence_similarity.png", bbox_inches="tight")
    plt.close()
    return


def plot_t_distributions(v, t_dists_info, imgdir):
    """Plot t distributions"""
    npos, nics = v.shape
    fig, axes = plt.subplots(nics, 1, figsize=(5, 3 * nics))
    for i in range(v.shape[1]):
        vi = v[:,i]
        tinfo = t_dists_info[i]
        if nics > 1:
            ax = axes[i]
        else:
            ax = axes
        ax.hist(
            vi, bins=20, density=True, alpha=0.5, color="skyblue",
        )
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        x = np.linspace(*xlims, 100)
        y = scipy.stats.t.pdf(
            x, df=tinfo["df"], loc=tinfo["loc"], scale=tinfo["scale"], 
        )
        ax.vlines(tinfo["cutoff"], *ylims, colors="k", linestyles="--")
        ax.plot(x, y)
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        ax.set_xlabel(f"IC {i}")
        ax.set_ylabel(f"p")
        ax.set_title(f"IC {i} Student's $t$")
    plt.tight_layout()
    plt.savefig(f"{imgdir}/t_distributions.png", bbox_inches="tight")
    plt.close()
