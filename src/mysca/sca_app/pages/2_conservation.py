"""Conservation — per-position relative entropy with sector overlay.

Bar chart of ``SCAResults.conservation`` (D_i, the position-wise
relative entropy from sca-core). Positions that belong to any IC's
high-load set get colored by the sector palette, with a winner-take-all
rule on overlap (last assignment wins to keep the legend small).
Positions outside every sector stay neutral.

X axis can switch between processed-MSA column index (the natural
SCA coordinate) and the original-MSA column index recovered via
``PreprocessingResults.retained_positions``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mysca.results import PreprocessingResults, SCAResults
from mysca.sca_app._chrome import resolve_bundle
from mysca.sca_app._results_io import ResultsBundle
from mysca.sca_app._structure_view import (
    PALETTE_BG,
    PALETTE_INK,
    sector_color,
)
from mysca.sca_app._styles import inject_styles, keep_state_alive

NEUTRAL_BAR = "#A8A29E"
TOP_N_DEFAULT = 20


st.set_page_config(
    page_title="mysca — Conservation",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def _load_artifacts(bundle_root: str, preproc_dir: str | None) -> dict:
    sca = SCAResults.load(bundle_root)
    out = {
        "conservation": (
            np.asarray(sca.conservation, dtype=np.float64)
            if sca.conservation is not None else None
        ),
        "ic_positions": (
            [np.asarray(p, dtype=int) for p in sca.ic_positions]
            if sca.ic_positions is not None else None
        ),
        "retained_positions": None,
    }
    if preproc_dir is not None:
        prep = PreprocessingResults.load(preproc_dir)
        out["retained_positions"] = np.asarray(
            prep.retained_positions, dtype=int,
        )
    return out


def _sector_assignment(
    n_positions: int, ic_positions: list[np.ndarray] | None,
) -> np.ndarray:
    """Length-``n_positions`` int array of IC index per position, or -1 for none.

    On overlap the last IC wins — keeps the legend tight and matches the
    overlapping-assignment policy used by sca-core's plots.
    """
    assign = np.full(n_positions, -1, dtype=int)
    if not ic_positions:
        return assign
    for ic_idx, positions in enumerate(ic_positions):
        if positions is None:
            continue
        idxs = np.asarray(positions, dtype=int)
        idxs = idxs[(idxs >= 0) & (idxs < n_positions)]
        assign[idxs] = ic_idx
    return assign


def _build_figure(
    conservation: np.ndarray,
    assignment: np.ndarray,
    x_values: np.ndarray,
    x_label: str,
) -> go.Figure:
    n_ic = int(assignment.max()) + 1 if assignment.size and assignment.max() >= 0 else 0

    fig = go.Figure()
    # Neutral bars first (no-sector positions)
    mask = assignment == -1
    if mask.any():
        fig.add_bar(
            x=x_values[mask], y=conservation[mask],
            marker_color=NEUTRAL_BAR,
            name="(no sector)",
            hovertemplate=(
                f"{x_label}=%{{x}}<br>conservation=%{{y:.3f}}"
                "<extra>(no sector)</extra>"
            ),
        )
    for ic_idx in range(n_ic):
        mask = assignment == ic_idx
        if not mask.any():
            continue
        fig.add_bar(
            x=x_values[mask], y=conservation[mask],
            marker_color=sector_color(ic_idx),
            name=f"IC {ic_idx + 1}",
            hovertemplate=(
                f"{x_label}=%{{x}}<br>conservation=%{{y:.3f}}"
                f"<extra>IC {ic_idx + 1}</extra>"
            ),
        )

    fig.update_layout(
        barmode="overlay",
        plot_bgcolor=PALETTE_BG,
        paper_bgcolor=PALETTE_BG,
        font_color=PALETTE_INK,
        xaxis=dict(title=x_label, showgrid=False),
        yaxis=dict(title="conservation (Dᵢ)", gridcolor="#E5E5E5"),
        legend=dict(orientation="h", y=-0.18),
        height=520,
        margin=dict(l=40, r=20, t=20, b=60),
        bargap=0,
    )
    return fig


def _run() -> None:
    keep_state_alive()
    inject_styles()
    st.markdown('<div class="sca-tagline">MYSCA · CONSERVATION</div>',
                unsafe_allow_html=True)

    bundle = resolve_bundle(require_kind="sca_core")
    if bundle is None:
        return

    try:
        artifacts = _load_artifacts(
            str(bundle.root),
            str(bundle.preprocessing_dir) if bundle.preprocessing_dir else None,
        )
    except Exception as e:
        st.error(
            "Failed to load conservation / IC positions: "
            f"`{type(e).__name__}: {e}`"
        )
        return

    conservation = artifacts["conservation"]
    if conservation is None:
        st.error("This bundle has no `conservation` field on its SCAResults.")
        return

    retained_positions = artifacts["retained_positions"]
    coord_options = ["processed-MSA column"]
    if retained_positions is not None:
        coord_options.append("original-MSA column")

    c1, c2 = st.columns([2, 1])
    with c1:
        coord = st.radio(
            "X coordinate", coord_options, horizontal=True,
            key="cv_coord_mode",
        )
    with c2:
        show_topn = st.toggle(
            f"Show only top-N positions", value=False, key="cv_topn_toggle",
        )
        top_n = st.number_input(
            "N", min_value=5, max_value=200, value=TOP_N_DEFAULT,
            step=5, disabled=not show_topn, key="cv_topn_value",
        )

    n_pos = len(conservation)
    assignment = _sector_assignment(n_pos, artifacts["ic_positions"])

    if coord == "original-MSA column":
        x_values = retained_positions
        x_label = "original-MSA column"
    else:
        x_values = np.arange(n_pos)
        x_label = "processed-MSA column"

    if show_topn:
        order = np.argsort(-conservation)[:int(top_n)]
        # Sort by x for readability
        order = order[np.argsort(x_values[order])]
        x_plot = x_values[order]
        y_plot = conservation[order]
        a_plot = assignment[order]
    else:
        x_plot, y_plot, a_plot = x_values, conservation, assignment

    fig = _build_figure(y_plot, a_plot, x_plot, x_label)
    st.plotly_chart(fig, use_container_width=True, key="cv_chart")

    n_in_sectors = int((assignment != -1).sum())
    st.caption(
        f"{n_pos} positions · {n_in_sectors} assigned to sectors · "
        f"mean Dᵢ = {conservation.mean():.3f} · "
        f"max Dᵢ = {conservation.max():.3f}"
    )


_run()
