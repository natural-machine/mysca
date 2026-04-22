import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

import scipy
import scipy.cluster.hierarchy as sch
import scipy.stats
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


def plot_filter_history(filter_history, imgdir):
    """Plot the change in MSA size (sequences and positions) across filter stages.

    Args:
        filter_history: list of dicts as emitted by preprocess_msa. Each entry
            must have `label`, `n_sequences`, `n_positions`, `n_filtered`,
            and `axis` ("sequences", "positions", or None for "initial").
        imgdir: output directory.
    """
    labels = [entry["label"] for entry in filter_history]
    n_seqs = [entry["n_sequences"] for entry in filter_history]
    n_pos = [entry["n_positions"] for entry in filter_history]
    n_stages = len(filter_history)
    x = np.arange(n_stages)

    fig, (ax_seq, ax_pos) = plt.subplots(2, 1, figsize=(max(7, 1.5 * n_stages), 7))

    def _draw(ax, counts, affected_axis, axis_label, color):
        bar_colors = [
            color if (entry["stage"] == "initial" or entry["axis"] == affected_axis)
            else "lightgray"
            for entry in filter_history
        ]
        ax.bar(x, counts, color=bar_colors, edgecolor="k")
        for i, (xi, ci) in enumerate(zip(x, counts)):
            ax.text(xi, ci, f"{ci:,}", ha="center", va="bottom", fontsize=9)
            if i > 0 and filter_history[i]["axis"] == affected_axis:
                delta = counts[i] - counts[i - 1]
                if delta != 0:
                    ax.text(
                        xi, ci * 0.5, f"{delta:+,}",
                        ha="center", va="center", fontsize=9, color="white",
                        fontweight="bold",
                    )
        ax.plot(x, counts, "k-", alpha=0.3, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(axis_label)
        ax.set_ylim(0, max(counts) * 1.12 if max(counts) > 0 else 1)
        ax.spines[["top", "right"]].set_visible(False)

    _draw(ax_seq, n_seqs, "sequences", "# sequences", "steelblue")
    _draw(ax_pos, n_pos, "positions", "# positions", "indianred")
    ax_seq.set_title("MSA size across filter stages")

    plt.tight_layout()
    plt.savefig(f"{imgdir}/filter_history.png", bbox_inches="tight")
    plt.close()
    return


def plot_filter_distributions(filter_history, imgdir):
    """Plot the distribution of the per-stage filter statistic with threshold.

    For each filtering stage (skipping the 'initial' record), draw a histogram
    of the statistic used to decide the filter (e.g. per-position gap
    frequency), with the threshold marked and the excluded region shaded.

    Args:
        filter_history: list of dicts as emitted by preprocess_msa.
        imgdir: output directory.
    """
    entries = [e for e in filter_history if e.get("stat_values") is not None]
    if not entries:
        return

    n = len(entries)
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.8 * n))
    if n == 1:
        axes = [axes]

    for ax, entry in zip(axes, entries):
        values = np.asarray(entry["stat_values"])
        thresh = entry["threshold"]
        direction = entry["filter_direction"]
        n_filtered = entry["n_filtered"]
        n_total = values.size

        nbins = min(60, max(10, int(np.sqrt(n_total))))
        counts, bins, patches = ax.hist(
            values, bins=nbins, color="steelblue", edgecolor="k", alpha=0.75,
        )
        # Shade patches whose bin lies in the rejected region
        for patch, left, right in zip(patches, bins[:-1], bins[1:]):
            center = 0.5 * (left + right)
            rejected = (
                (direction == "above" and center >= thresh)
                or (direction == "below" and center < thresh)
            )
            if rejected:
                patch.set_facecolor("lightcoral")
                patch.set_alpha(0.8)

        ax.axvline(
            thresh, color="k", linestyle="--", linewidth=1.5,
            label=f"{entry['threshold_symbol']} = {thresh}",
        )
        comparator = "≥" if direction == "above" else "<"
        ax.set_title(
            f"{entry['label']} — filter {entry['axis']} with "
            f"{entry['stat_name']} {comparator} {entry['threshold_symbol']} "
            f"({n_filtered:,} / {n_total:,} removed)"
        )
        ax.set_xlabel(entry["stat_name"])
        ax.set_ylabel("count")
        ax.legend(loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{imgdir}/filter_distributions.png", bbox_inches="tight")
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
