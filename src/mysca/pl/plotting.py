import logging
import os

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


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Each plotter follows the same save/return contract:
#
#   - outdir   : directory to save into. Default "." for library use;
#                scripts pass their image output dir.
#   - filename : output filename (within outdir). Default is the plot's
#                conventional name; data-dependent plots (plot_data_2d/3d)
#                derive one from the axis indices and group selection.
#   - save     : if True (default), write the figure to disk and close it.
#                Set save=False when using from a notebook / importing
#                interactively — the function returns the open figure or
#                axes so you can continue manipulating it.
#
# Single-axis plots return the Axes. Multi-axis plots return the Figure.
# The __all__ in pl/__init__.py documents the public surface.
# ---------------------------------------------------------------------------


def _maybe_save(fig, save, outdir, filename):
    """Save `fig` to ``{outdir}/{filename}`` and close it, iff save is True."""
    if not save:
        return
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, filename), bbox_inches="tight")
    plt.close(fig)


def plot_data_2d(
        ic_or_ev, axidxs, group_idxs, groups, data,
        outdir=".",
        *,
        filename=None,
        save=True,
):
    ic_or_ev = ic_or_ev.lower()
    if ic_or_ev == "ic":
        title = "Groups in IC space"
    elif ic_or_ev == "ev":
        title = "Groups in EV space"
    else:
        raise RuntimeError("ic_or_ev should be `ic` or `ev`!")
    if group_idxs == "all":
        group_idxs = list(range(len(groups)))
    axi, axj = axidxs
    if axj >= data.shape[1]:
        logger.debug(
            "plot_data_2d: skipping %s axes (%d,%d); data has only %d columns.",
            ic_or_ev.upper(), axi, axj, data.shape[1],
        )
        return None
    fig, ax = plt.subplots(1, 1)
    ax.scatter(
        data[:, axi], data[:, axj],
        c='k', alpha=0.2, edgecolor='k',
    )
    for gidx in group_idxs:
        if gidx >= len(groups):
            continue
        g = groups[gidx]
        ax.scatter(
            data[g, axi], data[g, axj],
            alpha=1, edgecolor='k', label=f"group {gidx}",
        )
    ax.plot(0, 0, "ro")
    rx, ry = ax.get_xlim()[1], ax.get_ylim()[1]
    ax.plot([0, rx], [0, 0], "k-", alpha=0.5)
    ax.plot([0, 0], [0, ry], "k-", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel(f"{ic_or_ev.upper()} {axi}")
    ax.set_ylabel(f"{ic_or_ev.upper()} {axj}")
    ax.set_title(title)
    fig.tight_layout()
    if filename is None:
        groupstr = "".join(str(i) for i in group_idxs)
        filename = f"{ic_or_ev}{axi}{axj}_groups_{groupstr}.png"
    _maybe_save(fig, save, outdir, filename)
    return ax


def plot_data_3d(
        ic_or_ev, axidxs, group_idxs, groups, data,
        outdir=".",
        *,
        filename=None,
        save=True,
):
    ic_or_ev = ic_or_ev.lower()
    if ic_or_ev == "ic":
        title = "ICA and identified groups"
    elif ic_or_ev == "ev":
        title = ""
    else:
        raise RuntimeError("ic_or_ev should be `ic` or `ev`!")
    if group_idxs == "all":
        group_idxs = list(range(len(groups)))
    axi, axj, axk = axidxs
    if axk >= data.shape[1]:
        logger.debug(
            "plot_data_3d: skipping %s axes (%d,%d,%d); data has only %d columns.",
            ic_or_ev.upper(), axi, axj, axk, data.shape[1],
        )
        return None
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        data[:, axi], data[:, axj], data[:, axk],
        c="k", alpha=0.2, edgecolor='k',
    )
    for gidx in group_idxs:
        if gidx >= len(groups):
            continue
        g = groups[gidx]
        ax.scatter(
            data[g, axi], data[g, axj], data[g, axk],
            alpha=1, edgecolor='k', label=f"group {gidx}",
        )
    ax.plot(0, 0, "ro")
    rx, ry, rz = ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]
    ax.plot([0, rx], [0, 0], [0, 0], "k-", alpha=0.5)
    ax.plot([0, 0], [0, ry], [0, 0], "k-", alpha=0.5)
    ax.plot([0, 0], [0, 0], [0, rz], "k-", alpha=0.5)
    ax.view_init(elev=30, azim=40)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel(f"{ic_or_ev.upper()} {axi}")
    ax.set_ylabel(f"{ic_or_ev.upper()} {axj}")
    ax.set_zlabel(f"{ic_or_ev.upper()} {axk}")
    ax.set_title(title)
    fig.tight_layout()
    if filename is None:
        groupstr = "".join(str(i) for i in group_idxs)
        filename = f"{ic_or_ev}{axi}{axj}{axk}_groups_{groupstr}.png"
    _maybe_save(fig, save, outdir, filename)
    return ax


def plot_seq_projection_2d(
        up, axidxs,
        outdir=".",
        *,
        color_values=None,
        color_label=None,
        filename=None,
        save=True,
):
    """Scatter sequences in IC sequence-space (Uᵖ).

    One point per row of `up` (one sequence). Implements the canonical
    Rivoire et al. (2016) sequence-position mapping output (Eqs. 14, 15);
    typically called with `up = sca.project_sequences(prep.msa_binary3d)`.

    Parameters
    ----------
    up : np.ndarray, shape (M, n_components)
        Sequence-space IC scores.
    axidxs : (int, int)
        IC pair to plot on the (x, y) axes.
    color_values : array-like of length M, optional
        Per-sequence color values. Numeric arrays render with a
        continuous colormap + colorbar. Object/string arrays render as
        discrete categories with a legend (NA values shown in grey).
    color_label : str, optional
        Axis label for the colorbar / legend title.
    """
    axi, axj = axidxs
    if axj >= up.shape[1]:
        logger.debug(
            "plot_seq_projection_2d: skipping IC axes (%d,%d); up has only "
            "%d columns.", axi, axj, up.shape[1],
        )
        return None
    fig, ax = plt.subplots(1, 1)

    if color_values is None:
        ax.scatter(
            up[:, axi], up[:, axj],
            s=6, c="k", alpha=0.3, edgecolor="none",
        )
    else:
        cv = np.asarray(color_values)
        if cv.shape[0] != up.shape[0]:
            raise ValueError(
                f"color_values length {cv.shape[0]} does not match "
                f"up rows {up.shape[0]}."
            )
        is_numeric = np.issubdtype(cv.dtype, np.number)
        if is_numeric:
            sc = ax.scatter(
                up[:, axi], up[:, axj],
                c=cv, s=8, alpha=0.7, edgecolor="none", cmap="viridis",
            )
            cbar = fig.colorbar(sc, ax=ax)
            if color_label is not None:
                cbar.ax.set_ylabel(color_label)
        else:
            cv_str = np.where(
                np.array([v is None for v in cv]) |
                np.array([
                    isinstance(v, float) and np.isnan(v) for v in cv
                ]),
                "NA",
                cv.astype(str),
            )
            categories = sorted(set(cv_str) - {"NA"})
            palette = plt.get_cmap("tab20")(
                np.linspace(0, 1, max(len(categories), 1))
            )
            cat_to_color = {c: palette[i] for i, c in enumerate(categories)}
            cat_to_color["NA"] = (0.7, 0.7, 0.7, 0.5)
            for cat in categories + ["NA"]:
                mask = cv_str == cat
                if not mask.any():
                    continue
                ax.scatter(
                    up[mask, axi], up[mask, axj],
                    color=[cat_to_color[cat]], label=cat,
                    s=8, alpha=0.7, edgecolor="none",
                )
            ax.legend(
                bbox_to_anchor=(1.05, 1), loc="upper left",
                fontsize=7, title=color_label,
            )

    ax.set_xlabel(f"seq score IC {axi} ($\\tilde{{U}}^p_{{{axi}}}$)")
    ax.set_ylabel(f"seq score IC {axj} ($\\tilde{{U}}^p_{{{axj}}}$)")
    ax.set_title("Sequences in IC space")
    fig.tight_layout()
    if filename is None:
        suffix = f"_by_{color_label}" if color_label else ""
        filename = f"seq_proj_ic{axi}v{axj}{suffix}.png"
    _maybe_save(fig, save, outdir, filename)
    return ax


def plot_dendrogram(
        Cij,
        outdir=".",
        *,
        nclusters=10,
        filename="dendrogram.png",
        save=True,
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
        Z, orientation='left', ax=ax1,
        link_color_func=color_func,
        above_threshold_color='k',
    )
    ax1.set_ylabel('Position', fontsize='x-large')
    ax1.set_xticks([])
    ax1.set_yticks([])
    rearranged_data = Cij[leaf_indices][:, leaf_indices]
    ax2.imshow(
        rearranged_data, aspect='auto', cmap='Blues',
        interpolation='none', origin='lower',
    )
    boundaries = np.where(np.diff(clusters[leaf_indices]))[0]
    for b in boundaries:
        ax2.axhline(b + 0.5, color='black', linestyle='--')
        ax2.axvline(b + 0.5, color='black', linestyle='--')
    ax2.set_title('Clustering of Positions', fontsize='x-large')
    ax2.set_xlabel('Position', fontsize='x-large')
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.tight_layout()
    _maybe_save(fig, save, outdir, filename)
    return fig


def plot_sequence_similarity(
        xmsa,
        outdir=".",
        *,
        filename="sequence_similarity.png",
        save=True,
):
    npos = xmsa.shape[1]
    xmsa = xmsa.argmax(axis=-1)  # convert one-hot to int MSA
    distances = pdist(xmsa, metric="hamming")
    similarities = 1 - distances
    similarity_matrix = 1 - squareform(distances)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    Z = sch.linkage(distances, method="complete", metric="hamming")
    dendro = sch.dendrogram(Z, no_plot=True)
    idxs = dendro["leaves"]

    ax1.hist(similarities, int(round(npos / 2)))
    ax1.set_xlabel("Pairwise sequence identities")
    ax1.set_ylabel("Count")

    sc = ax2.imshow(
        similarity_matrix[np.ix_(idxs, idxs)],
        vmin=0, vmax=1, interpolation="none",
    )
    fig.colorbar(sc, ax=ax2)
    fig.tight_layout()
    _maybe_save(fig, save, outdir, filename)
    return fig


def plot_filter_history(
        filter_history,
        outdir=".",
        *,
        filename="filter_history.png",
        save=True,
):
    """Plot the change in MSA size (sequences and positions) across filter stages.

    Args:
        filter_history: list of dicts as emitted by preprocess_msa. Each entry
            must have ``label``, ``n_sequences``, ``n_positions``,
            ``n_filtered``, and ``axis`` ("sequences", "positions", or None
            for "initial").
    """
    labels = [entry["label"] for entry in filter_history]
    n_seqs = [entry["n_sequences"] for entry in filter_history]
    n_pos = [entry["n_positions"] for entry in filter_history]
    n_stages = len(filter_history)
    x = np.arange(n_stages)

    fig, (ax_seq, ax_pos) = plt.subplots(
        2, 1, figsize=(max(7, 1.5 * n_stages), 7)
    )

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

    fig.tight_layout()
    _maybe_save(fig, save, outdir, filename)
    return fig


def plot_prealign_filter_history(
        filter_history,
        outdir=".",
        *,
        filename="prealign_filter_history.png",
        save=True,
):
    """Sequence-count drop across prealign stages.

    Prealign has no per-position filtering (alignment re-derives positions),
    so this is a single-panel waterfall of sequence counts across the
    initial → cluster → align stages.

    Args:
        filter_history: list of dicts. Each entry must have ``label``,
            ``n_sequences``, ``n_filtered``, and ``stage`` (``"initial"``,
            ``"cluster"``, or ``"align"``).
    """
    labels = [entry["label"] for entry in filter_history]
    n_seqs = [entry["n_sequences"] for entry in filter_history]
    n_stages = len(filter_history)
    x = np.arange(n_stages)

    fig, ax = plt.subplots(1, 1, figsize=(max(5, 1.5 * n_stages), 4))
    bar_colors = [
        "lightgray" if entry["stage"] == "initial" else "steelblue"
        for entry in filter_history
    ]
    ax.bar(x, n_seqs, color=bar_colors, edgecolor="k")
    for i, (xi, ci) in enumerate(zip(x, n_seqs)):
        ax.text(xi, ci, f"{ci:,}", ha="center", va="bottom", fontsize=9)
        if i > 0:
            delta = n_seqs[i] - n_seqs[i - 1]
            if delta != 0:
                ax.text(
                    xi, ci * 0.5, f"{delta:+,}",
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold",
                )
    ax.plot(x, n_seqs, "k-", alpha=0.3, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("# sequences")
    ax.set_ylim(0, max(n_seqs) * 1.12 if max(n_seqs) > 0 else 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Sequence count across prealign stages")

    fig.tight_layout()
    _maybe_save(fig, save, outdir, filename)
    return ax


def plot_filter_distributions(
        filter_history,
        outdir=".",
        *,
        filename="filter_distributions.png",
        save=True,
):
    """Plot the distribution of the per-stage filter statistic with threshold.

    For each filtering stage (skipping the 'initial' record), draw a histogram
    of the statistic used to decide the filter (e.g. per-position gap
    frequency), with the threshold marked and the excluded region shaded.
    Returns None when there are no entries with stat_values, since there is
    no figure to return in that case.
    """
    entries = [e for e in filter_history if e.get("stat_values") is not None]
    if not entries:
        return None

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

    fig.tight_layout()
    _maybe_save(fig, save, outdir, filename)
    return fig


def plot_t_distributions(
        v, t_dists_info,
        outdir=".",
        *,
        max_plots=None,
        filename="t_distributions.png",
        save=True,
):
    """Plot per-IC t-distribution fit with cutoff marker.

    By default one subplot is rendered per column of ``v``. Pass
    ``max_plots=N`` to render only the first ``N`` ICs (useful from
    entrypoints that want to skip non-significant ICs without discarding
    the saved ``t_dists_info`` for them).
    """
    _, nics = v.shape
    if max_plots is not None:
        nics = min(max_plots, nics)
    if nics <= 0:
        return None
    fig, axes = plt.subplots(nics, 1, figsize=(5, 3 * nics))
    for i in range(nics):
        vi = v[:, i]
        tinfo = t_dists_info[i]
        ax = axes[i] if nics > 1 else axes
        ax.hist(vi, bins=20, density=True, alpha=0.5, color="skyblue")
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
        ax.set_ylabel("p")
        ax.set_title(f"IC {i} Student's $t$")
    fig.tight_layout()
    _maybe_save(fig, save, outdir, filename)
    return fig


def plot_conservation_top(
        retained_positions, Di, num_positions_orig,
        outdir=".",
        *,
        filename="top_conservation.png",
        save=True,
):
    """Scatter of per-position relative entropy D_i at retained original
    positions, with the x-axis spanning the original alignment length.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(retained_positions, Di, "o", color="Blue", alpha=0.2)
    ax.set_xlim(0, num_positions_orig)
    ax.set_xlabel("Position")
    ax.set_ylabel(r"Relative Entropy $D_i$")
    ax.set_title("Conservation")
    fig.tight_layout()
    _maybe_save(fig, save, outdir, filename)
    return ax


def plot_conservation_positional(
        retained_positions, Di, num_positions_orig,
        outdir=".",
        *,
        filename="positional_conservation.png",
        save=True,
):
    """Bar chart of D_i at retained original positions, with the x-axis
    spanning the original alignment length.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.bar(retained_positions, Di, color="Blue", width=1.0, align="center")
    ax.set_xlim(0, num_positions_orig)
    ax.set_xlabel("Position")
    ax.set_ylabel(r"Relative Entropy $D_i$")
    ax.set_title("Conservation")
    fig.tight_layout()
    _maybe_save(fig, save, outdir, filename)
    return ax


def plot_conservation(
        Di,
        outdir=".",
        *,
        filename="conservation.png",
        save=True,
):
    """Bar chart of D_i over the retained-position index axis (no mapping
    back to original-MSA coordinates).
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.bar(np.arange(len(Di)), Di, color="Blue", width=1.0, align="center")
    ax.set_xlabel("Position")
    ax.set_ylabel(r"Relative Entropy $D_i$")
    ax.set_title("Conservation")
    fig.tight_layout()
    _maybe_save(fig, save, outdir, filename)
    return ax


def _plot_matrix_imshow(
        matrix, title, *, xlabel, ylabel, cbar_label="Covariation",
):
    fig, ax = plt.subplots(1, 1)
    sc = ax.imshow(
        matrix, cmap="Blues", origin="lower", interpolation="none", vmax=None,
    )
    fig.colorbar(sc, label=cbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax


def plot_covariance_matrix(
        Cij_raw,
        outdir=".",
        *,
        filename="covariance_matrix.png",
        save=True,
):
    """Imshow of the raw (pre-weighting) covariance matrix."""
    fig, _ax = _plot_matrix_imshow(
        Cij_raw, "Covariance Matrix",
        xlabel="(Retained) Position i",
        ylabel="(Retained) Position j",
    )
    _maybe_save(fig, save, outdir, filename)
    return fig


def plot_sca_matrix(
        Cij,
        outdir=".",
        *,
        filename="sca_matrix.png",
        save=True,
):
    """Imshow of the weighted SCA covariance matrix."""
    fig, _ax = _plot_matrix_imshow(
        Cij, "SCA Matrix",
        xlabel="(Retained) Position i",
        ylabel="(Retained) Position j",
    )
    _maybe_save(fig, save, outdir, filename)
    return fig


def plot_sca_spectrum(
        evals_sca, evals_shuff,
        outdir=".",
        *,
        filename="sca_matrix_spectrum.png",
        save=True,
):
    """Overlay the SCA eigenvalue spectrum with each bootstrap null
    sample's eigenvalue spectrum.
    """
    fig, ax = plt.subplots(1, 1)
    for e in evals_shuff:
        ax.plot(1 + np.arange(len(e)), e, ".", markersize=3)
    ax.plot(
        1 + np.arange(len(evals_sca)), evals_sca,
        "k.", markersize=2, label="data",
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel(r"$\lambda$ index")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(r"$\tilde{C}_{ij}$ Spectrum (data vs null)")
    fig.tight_layout()
    _maybe_save(fig, save, outdir, filename)
    return fig


def plot_sca_spectrum_vs_null(
        evals_sca, evals_shuff, cutoff, n_boot,
        outdir=".",
        *,
        filename="sca_matrix_spectrum_vs_null.png",
        save=True,
):
    """Histograms of the SCA eigenvalues vs the pooled bootstrap null,
    with the significance cutoff marked.
    """
    fig, ax = plt.subplots(1, 1)
    _counts, bins, _patches = ax.hist(
        evals_sca, bins=100, color="black", alpha=0.8, log=True, label="Data",
    )
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    h, _ = np.histogram(np.asarray(evals_shuff).flatten(), bins=bins)
    ax.axvline(cutoff, 0, 1, linestyle="--", color="grey")
    denom = n_boot if n_boot else 1
    ax.plot(bin_centers, h / denom, color="red", lw=1.5, label="Null")
    ax.legend()
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Count")
    ax.set_title("Spectral decomposition")
    fig.tight_layout()
    _maybe_save(fig, save, outdir, filename)
    return fig


def plot_sca_matrix_sector_subset(
        sca_mat_imp, groups, sector_color_set=None,
        outdir=".",
        *,
        filename="sca_matrix_important_subset.png",
        save=True,
):
    """Imshow of the sector-subset SCA matrix, optionally decorated with
    sector-colored rugs along the top and right axes.
    """
    fig, ax = plt.subplots(1, 1)
    sc = ax.imshow(
        sca_mat_imp, cmap="Blues", origin="lower",
        interpolation="none", vmax=None,
    )
    fig.colorbar(sc, label="Covariation")
    ax.set_xlabel("(Important) Position i")
    ax.set_ylabel("(Important) Position j")
    ax.set_title("SCA Matrix (Groups)")

    group_lengths = [len(g) for g in groups]
    if sector_color_set and np.sum(group_lengths) > 0:
        group_colors = np.concatenate([
            len(g) * [colors.to_rgb(sector_color_set[i])]
            for i, g in enumerate(groups) if len(g) > 0
        ], axis=0)

        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="2%", pad=0.0, sharex=ax)
        ax_top.imshow(
            group_colors[None, :, :], aspect="auto",
            extent=(0, len(group_colors), 0, 1),
        )
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_title(ax.get_title())
        ax.set_title("")
        ax_right = divider.append_axes("right", size="2%", pad=0.0, sharey=ax)
        ax_right.imshow(
            np.flip(group_colors, axis=0)[:, None, :], aspect="auto",
            extent=(0, 1, 0, len(group_colors)),
        )
        ax_right.set_xticks([])
        ax_right.set_yticks([])

    _maybe_save(fig, save, outdir, filename)
    return fig
