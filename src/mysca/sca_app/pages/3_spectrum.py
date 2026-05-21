"""Spectrum — eigenvalue scree vs bootstrap null.

Replicates the figure sca-core writes to
``images/sca_matrix_spectrum_vs_null.png`` but interactively. Plots
``SCAResults.evals_sca`` (sorted descending) as a scatter, overlays the
bootstrap-null spectrum ``evals_shuff`` as a mean-±-σ band, marks
``cutoff`` and the ``kstar`` count, and lets the user zoom into the
top-K eigenvalues.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from mysca.results import SCAResults
from mysca.sca_app._chrome import resolve_bundle
from mysca.sca_app._structure_view import (
    PALETTE_BG,
    PALETTE_GREEN,
    PALETTE_INK,
    PALETTE_RED,
    PALETTE_TEAL,
)
from mysca.sca_app._styles import inject_styles, keep_state_alive


st.set_page_config(
    page_title="mysca — Spectrum",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def _load_spectrum(bundle_root: str) -> dict:
    sca = SCAResults.load(bundle_root)
    return {
        "evals_sca": (
            np.asarray(sca.evals_sca, dtype=np.float64)
            if sca.evals_sca is not None else None
        ),
        "evals_shuff": (
            np.asarray(sca.evals_shuff, dtype=np.float64)
            if sca.evals_shuff is not None else None
        ),
        "kstar": sca.kstar,
        "kstar_identified": sca.kstar_identified,
        "cutoff": sca.cutoff,
    }


def _build_figure(
    evals_sca: np.ndarray,
    evals_shuff: np.ndarray | None,
    cutoff: float | None,
    kstar: int | None,
    top_k: int | None,
) -> go.Figure:
    sorted_evals = np.sort(evals_sca)[::-1]
    n = len(sorted_evals)
    if top_k:
        sorted_evals = sorted_evals[:top_k]
    idx = np.arange(1, len(sorted_evals) + 1)

    fig = go.Figure()

    if evals_shuff is not None and evals_shuff.size:
        shuff_sorted = np.sort(evals_shuff, axis=1)[:, ::-1]
        if top_k:
            shuff_sorted = shuff_sorted[:, :top_k]
        mean = shuff_sorted.mean(axis=0)
        std = shuff_sorted.std(axis=0)
        upper = mean + std
        lower = mean - std
        x_band = np.concatenate([idx, idx[::-1]])
        y_band = np.concatenate([upper, lower[::-1]])
        fig.add_trace(go.Scatter(
            x=x_band, y=y_band,
            fill="toself",
            fillcolor="rgba(116, 177, 193, 0.30)",
            line=dict(color="rgba(116, 177, 193, 0)"),
            name="bootstrap null (mean ± σ)",
            hoverinfo="skip",
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=idx, y=mean,
            mode="lines",
            line=dict(color=PALETTE_TEAL, dash="dot", width=1.5),
            name="null mean",
            hovertemplate="rank=%{x}<br>λ_null=%{y:.3f}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=idx, y=sorted_evals,
        mode="markers+lines",
        marker=dict(color=PALETTE_INK, size=7),
        line=dict(color=PALETTE_INK, width=1.0),
        name="λ (SCA)",
        hovertemplate="rank=%{x}<br>λ=%{y:.3f}<extra></extra>",
    ))

    if cutoff is not None:
        fig.add_hline(
            y=float(cutoff), line=dict(color=PALETTE_RED, dash="dash"),
            annotation_text=f"cutoff = {float(cutoff):.3f}",
            annotation_position="top right",
            annotation_font_color=PALETTE_RED,
        )
    if kstar is not None and kstar > 0:
        fig.add_vline(
            x=int(kstar), line=dict(color=PALETTE_GREEN, dash="dash"),
            annotation_text=f"k* = {int(kstar)}",
            annotation_position="bottom right",
            annotation_font_color=PALETTE_GREEN,
        )

    fig.update_layout(
        plot_bgcolor=PALETTE_BG,
        paper_bgcolor=PALETTE_BG,
        font_color=PALETTE_INK,
        xaxis=dict(title="eigenvalue rank", showgrid=False),
        yaxis=dict(title="eigenvalue λ", gridcolor="#E5E5E5"),
        legend=dict(orientation="h", y=-0.20),
        height=560,
        margin=dict(l=40, r=20, t=20, b=70),
    )
    return fig


def _run() -> None:
    keep_state_alive()
    inject_styles()
    st.markdown('<div class="sca-tagline">MYSCA · SPECTRUM</div>',
                unsafe_allow_html=True)

    bundle = resolve_bundle(require_kind="sca_core")
    if bundle is None:
        return

    spectrum = _load_spectrum(str(bundle.root))
    if spectrum["evals_sca"] is None:
        st.error("This bundle has no `evals_sca` field on its SCAResults.")
        return

    n = len(spectrum["evals_sca"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("L (positions)", n)
    c2.metric("k★", spectrum["kstar"] or "—")
    c3.metric("k★ (raw)", spectrum["kstar_identified"] or "—")
    c4.metric("cutoff",
              f"{spectrum['cutoff']:.3f}" if spectrum["cutoff"] is not None else "—")

    zoom = st.toggle("Zoom to top eigenvalues", value=True, key="sp_zoom_toggle")
    top_k = None
    if zoom:
        top_k = st.slider(
            "Show top-K", min_value=5, max_value=min(n, 60),
            value=min(n, 25), step=1, key="sp_top_k",
        )

    fig = _build_figure(
        spectrum["evals_sca"],
        spectrum["evals_shuff"],
        spectrum["cutoff"],
        spectrum["kstar"],
        top_k,
    )
    st.plotly_chart(fig, use_container_width=True, key="sp_chart")

    if spectrum["evals_shuff"] is None:
        st.caption(
            "No bootstrap null on disk (`evals_shuff.npy`). Run sca-core "
            "with `-nb / --n_boot > 0` to populate it."
        )


_run()
