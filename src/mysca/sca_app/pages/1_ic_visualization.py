"""IC visualized — sequence projection onto two ICs, colored by biology.

Plots every retained sequence at its (Uᵖ_i, Uᵖ_j) coordinates with
optional marginal histograms along each axis. Colors come from one of
four buckets, chosen via a two-step picker (category → level):

  - **Phylogeny** — taxonomic ranks (kingdom → genus) pulled from
    UniProt via accession-keyed batch fetch (cached on disk).
  - **Function** — UniProt EC class, keywords, protein names.
  - **Sequence-derived** — organism / species mnemonic / ungapped
    length, derived locally from the bundle.
  - **Custom metadata** — anything in a user-uploaded TSV/CSV merged
    onto ``seq_id``, or columns from a bundle's ``sequence_metadata.tsv``.

The plot is a fixed 900×900 with marginal histograms and the legend at
the bottom, so adding more categories doesn't squeeze the data area.
"""

from __future__ import annotations

import io
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from mysca.sca_app._chrome import resolve_bundle
from mysca.sca_app._results_io import ResultsBundle
from mysca.sca_app._seq_table import (
    UP_PREFIX,
    build_sequence_table,
    categorize_color_columns,
    has_uniprot_enrichment,
    is_numeric_column,
    list_up_columns,
    merge_user_metadata,
    uniprot_cache_path,
)
from mysca.sca_app._structure_view import (
    CONSERVATION_HIGH,
    CONSERVATION_LOW,
    EXTENDED_PALETTE,
    PALETTE_BG,
    PALETTE_INK,
    SECTOR_PALETTE,
)
from mysca.sca_app._styles import inject_styles, keep_state_alive
from mysca.sca_app._uniprot import (
    enrich_dataframe as enrich_with_uniprot,
    extract_uniprot_accessions,
    has_cached_metadata,
)

UNIFORM = "(uniform)"
MAX_CATEGORIES = 20
# Plot area is left square; the figure adds extra width for the legend
# margin only when Plotly's built-in legend / colorbar is in use.
# Categorical mode renders a custom HTML legend in a sibling Streamlit
# column instead, so the figure itself drops the right-margin reserve
# and uses the full PLOT_SIZE for the data area.
PLOT_SIZE = 1080
LEGEND_MARGIN_PX = 180
USER_METADATA_STATE_KEY = "iv_user_metadata"


st.set_page_config(
    page_title="mysca — IC Visualized",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def _cached_sequence_table(
    bundle_root: str, cache_mtime: float | None,
) -> pd.DataFrame:
    from mysca.sca_app._results_io import discover_results
    return build_sequence_table(discover_results(bundle_root))


def _uniprot_cache_mtime(bundle: ResultsBundle) -> float | None:
    p = uniprot_cache_path(bundle)
    return p.stat().st_mtime if p.is_file() else None


def _ic_label(col: str) -> str:
    return f"IC {int(col[len(UP_PREFIX):]) + 1}"


def _bin_categorical(s: pd.Series, max_n: int) -> pd.Series:
    s = s.astype("string").fillna("(missing)")
    top = s.value_counts().head(max_n).index
    return s.where(s.isin(top), other="(other)")


def _wrap_label(
    label: str, max_chars: int = 22, max_lines: int = 2,
) -> str:
    """Break a long legend label into at most ``max_lines`` lines.

    Plotly renders ``<br>`` in trace names as a line break inside the
    legend entry, which keeps long labels readable without overflowing
    the legend column.

    Crucially, the number of lines per entry is **capped** so a
    20-entry legend of long keywords doesn't stack taller than the
    figure (which would cause Plotly to compress the plot area to fit).
    Content past the cap is truncated with ``…``.

    Break points are taken at the rightmost space or ``;`` within each
    line's window (so semicolon-separated keyword strings break
    naturally); falls back to hard wrap when no break point fits.
    """
    s = str(label)
    if len(s) <= max_chars:
        return s

    lines: list[str] = []
    remaining = s
    while remaining and len(lines) < max_lines:
        if len(remaining) <= max_chars:
            lines.append(remaining)
            remaining = ""
            break
        chunk = remaining[:max_chars]
        # Prefer the latest soft break point in the window.
        break_idx = max(chunk.rfind(" "), chunk.rfind(";"))
        if break_idx > 0:
            split = break_idx + 1
            lines.append(remaining[:split].rstrip())
            remaining = remaining[split:].lstrip()
        else:
            lines.append(remaining[:max_chars])
            remaining = remaining[max_chars:]

    if remaining:
        # More content than fits in max_lines; truncate the last line.
        last = lines[-1]
        if len(last) > max_chars - 1:
            last = last[:max_chars - 1]
        lines[-1] = last + "…"

    return "<br>".join(lines)


def _palette_for(n_bins: int) -> list[str]:
    """Pick the palette that fits ``n_bins`` without repeating colors.

    Up to 5 bins use the in-app SECTOR_PALETTE so colors echo the
    structure-view IC highlights. Past 5, fall back to the extended
    20-color palette of harmonic tints/shades. Past 20, we'd repeat —
    but :func:`_bin_categorical` keeps the cap at 20 plus a single
    ``(other)`` bin, so 21 is the worst case.
    """
    if n_bins <= len(SECTOR_PALETTE):
        return list(SECTOR_PALETTE)
    return list(EXTENDED_PALETTE)


# Per-line character width for legend labels. Combined with the
# 2-line cap in `_wrap_label`, this bounds each legend entry to
# ~44 chars, so a 20-entry categorical legend tops out at ~40 lines
# of text — comfortably under the 980-px figure height even with the
# longest keyword strings, so the plot area is never compressed.
LEGEND_LABEL_WRAP_CHARS = 22


def _build_figure(
    df: pd.DataFrame, x_col: str, y_col: str, color_col: str | None,
    color_label: str | None,
):
    """Return ``(fig, legend_items)``.

    For categorical colors the Plotly legend is suppressed and the
    caller renders a custom HTML legend instead (Plotly's SVG legend
    text doesn't support hover tooltips, so we can't surface the
    untruncated label without an HTML one). ``legend_items`` is the
    per-bin metadata that custom renderer needs:
    ``[{full, label, color, count}, ...]``. Empty for numeric or
    uniform-color modes.
    """
    common = dict(
        x=x_col, y=y_col,
        labels={x_col: _ic_label(x_col), y_col: _ic_label(y_col)},
        hover_data={"seq_id": True, x_col: ":.3f", y_col: ":.3f"},
        marginal_x="histogram",
        marginal_y="histogram",
        height=PLOT_SIZE,
        # Total figure width = plot area + legend margin. The plot area
        # stays PLOT_SIZE wide regardless of how many legend entries
        # appear; extra entries just stack inside the reserved margin.
        width=PLOT_SIZE + LEGEND_MARGIN_PX,
    )
    legend_items: list[dict] = []
    if color_col is None:
        fig = px.scatter(df, **common)
        fig.update_traces(selector=dict(type="scatter"),
                          marker=dict(color=PALETTE_INK))
    elif is_numeric_column(df, color_col):
        plot_df = df.dropna(subset=[color_col])
        labels = dict(common.pop("labels"))
        labels[color_col] = color_label or color_col
        fig = px.scatter(
            plot_df,
            color=color_col,
            color_continuous_scale=[CONSERVATION_LOW, CONSERVATION_HIGH],
            labels=labels,
            **common,
        )
    else:
        binned_raw = _bin_categorical(df[color_col], MAX_CATEGORIES)
        counts = binned_raw.value_counts()
        ordered_raw = list(counts.index)
        # Wrap long category labels with <br> so they break onto a
        # second line inside the legend column rather than pushing past
        # the legend margin and squeezing the plot.
        label_map = {
            v: _wrap_label(v, LEGEND_LABEL_WRAP_CHARS) for v in ordered_raw
        }
        binned = binned_raw.map(label_map)
        ordered = [label_map[v] for v in ordered_raw]
        palette = _palette_for(len(ordered_raw))
        sub = df.assign(_color=binned, _full_label=binned_raw.astype(str))
        labels = dict(common.pop("labels"))
        labels["_color"] = color_label or color_col
        labels["_full_label"] = color_label or color_col
        fig = px.scatter(
            sub,
            x=x_col, y=y_col,
            color="_color",
            category_orders={"_color": ordered},
            color_discrete_sequence=palette,
            labels=labels,
            # Include the *full* category in the point hover so users
            # see the un-truncated value when hovering over a colored
            # marker, mirroring the hover tooltip on the custom legend.
            hover_data={
                "seq_id": True, x_col: ":.3f", y_col: ":.3f",
                "_full_label": True, "_color": False,
            },
            marginal_x="histogram",
            marginal_y="histogram",
            height=PLOT_SIZE,
            # Custom legend is rendered alongside the plot; the figure
            # itself doesn't need the legend margin, so the plot uses
            # the full PLOT_SIZE width.
            width=PLOT_SIZE,
        )
        # Hide the built-in legend; the caller renders a custom HTML
        # legend with per-entry hover tooltips for the full label.
        fig.update_layout(showlegend=False)
        legend_items = [
            {
                "full": str(raw),
                "label": label_map[raw],
                "color": palette[i % len(palette)],
                "count": int(counts.iloc[i]),
            }
            for i, raw in enumerate(ordered_raw)
        ]
    return fig, legend_items


def _render_html_legend(items: list[dict], title: str) -> str:
    """Standalone HTML document for an iframe-rendered legend.

    Each item gets ``title="<full label>"`` so the browser fires a
    native tooltip with the un-truncated text on hover. We deliver
    this through ``st.components.v1.html`` (which renders inside an
    iframe with no sanitization) rather than ``st.markdown`` —
    Streamlit's markdown sanitizer strips ``title`` attributes, which
    is why hover wasn't firing previously.
    """
    import html as _html
    item_html: list[str] = []
    for item in items:
        full = _html.escape(item["full"], quote=True)
        # `<br>` is already baked into `item["label"]`; pass through.
        item_html.append(
            f'<div class="sca-legend-item" title="{full}">'
            f'<span class="sca-legend-swatch" '
            f'style="background:{item["color"]}"></span>'
            f'<span class="sca-legend-text">'
            f'{item["label"]}'
            f' <span class="sca-legend-count">· {item["count"]}</span>'
            f'</span></div>'
        )
    title_html = _html.escape(title)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
  body {{
    margin: 0; padding: 4px 0 0 8px;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    background: #FBF3EF; color: #000;
  }}
  .sca-legend-title {{
    font-weight: 700; font-size: 0.82rem;
    margin: 0 0 8px 0;
  }}
  .sca-legend-item {{
    display: flex; align-items: flex-start; gap: 8px;
    margin-bottom: 6px; cursor: help;
    font-size: 0.78rem; line-height: 1.2;
  }}
  .sca-legend-swatch {{
    width: 12px; height: 12px; border-radius: 2px;
    flex-shrink: 0; margin-top: 3px;
  }}
  .sca-legend-text {{ white-space: normal; word-break: break-word; }}
  .sca-legend-count {{ color: #6B6155; font-weight: 600; }}
</style></head>
<body>
  <div class="sca-legend-title">{title_html}</div>
  {''.join(item_html)}
</body></html>"""


def _legend_height(items: list[dict]) -> int:
    """Pixel height for the legend iframe; ~44 px for wrapped items,
    ~28 px for single-liners, plus 40 px header/padding.
    """
    base = 40
    per_item_total = sum(
        44 if "<br>" in item["label"] else 28 for item in items
    )
    return base + per_item_total


def _render_uniprot_panel(bundle: ResultsBundle, df: pd.DataFrame) -> None:
    accessions = [a for a in extract_uniprot_accessions(df["seq_id"]).values() if a]
    n_acc = len(accessions)
    cache = uniprot_cache_path(bundle)
    cached = has_cached_metadata(cache)
    enriched = has_uniprot_enrichment(df)

    if n_acc == 0:
        st.caption(
            "ℹ️ UniProt enrichment is unavailable — none of the seq_ids "
            "carry a UniProt accession."
        )
        return

    c1, c2 = st.columns([3, 1])
    with c1:
        if cached and enriched:
            st.caption(
                f"🧬 UniProt metadata cached at `{cache}` "
                f"({n_acc} accessions); phylogeny + function color "
                "options unlocked."
            )
        else:
            st.caption(
                f"🧬 No UniProt cache yet. Fetch metadata for {n_acc} "
                "accessions to color by phylogeny / function."
            )
    with c2:
        label = "Refresh UniProt" if cached else "Fetch UniProt metadata"
        if st.button(label, key="iv_fetch_uniprot", use_container_width=True):
            try:
                with st.spinner(
                    f"Fetching {n_acc} UniProt records "
                    f"(~{1 + n_acc // 100} batches)…"
                ):
                    enrich_with_uniprot(
                        df, cache_path=cache, force_refresh=cached,
                    )
                st.toast("UniProt metadata refreshed", icon="✅")
                _cached_sequence_table.clear()
                st.rerun()
            except urllib.error.HTTPError as e:
                st.error(
                    f"UniProt request failed with HTTP {e.code}. Retry "
                    "with a different network."
                )
            except urllib.error.URLError as e:
                st.error(f"Network error while contacting UniProt: {e.reason}")
            except Exception as e:  # noqa: BLE001
                st.error(
                    f"UniProt enrichment failed: `{type(e).__name__}: {e}`"
                )


def _render_user_metadata_panel() -> None:
    """File-upload widget for a user's own metadata TSV/CSV.

    Parsed table lives in ``st.session_state[USER_METADATA_STATE_KEY]``;
    merged into the per-sequence DataFrame on every render until the
    user clicks *Clear*. Requires a ``seq_id`` column; any other
    columns become color options automatically.
    """
    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption(
            "📄 Upload a TSV/CSV keyed on `seq_id` (other columns become "
            "color options). Pairs with or replaces UniProt enrichment."
        )
        existing = st.session_state.get(USER_METADATA_STATE_KEY)
        if existing is not None:
            n_rows, n_cols = existing.shape
            cols = [c for c in existing.columns if c != "seq_id"]
            st.caption(
                f"✅ Custom metadata loaded: {n_rows} rows × {n_cols} "
                f"columns ({', '.join(cols[:6])}"
                f"{'…' if len(cols) > 6 else ''})."
            )
    with c2:
        uploaded = st.file_uploader(
            "Upload metadata",
            type=["tsv", "csv", "txt"],
            key="iv_metadata_upload",
            label_visibility="collapsed",
        )
        if uploaded is not None and st.session_state.get(
            "iv_uploaded_filename"
        ) != uploaded.name:
            try:
                raw = uploaded.read()
                # Sniff delimiter from extension; fall back to tab.
                if uploaded.name.lower().endswith(".csv"):
                    sep = ","
                else:
                    sep = "\t"
                user_df = pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str)
                if "seq_id" not in user_df.columns:
                    st.error(
                        "Uploaded file is missing a `seq_id` column."
                    )
                else:
                    st.session_state[USER_METADATA_STATE_KEY] = user_df
                    st.session_state["iv_uploaded_filename"] = uploaded.name
                    st.toast("Custom metadata loaded", icon="✅")
                    _cached_sequence_table.clear()
                    st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Failed to parse upload: `{type(e).__name__}: {e}`")
        if st.session_state.get(USER_METADATA_STATE_KEY) is not None:
            if st.button("Clear metadata",
                         key="iv_clear_metadata",
                         use_container_width=True):
                st.session_state.pop(USER_METADATA_STATE_KEY, None)
                st.session_state.pop("iv_uploaded_filename", None)
                _cached_sequence_table.clear()
                st.rerun()


def _color_choice_widget(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Two-step picker: category → level. Returns (column, display label)."""
    cats = categorize_color_columns(df)
    cat_options = [UNIFORM] + list(cats.keys())

    if "iv_color_category" in st.session_state \
            and st.session_state["iv_color_category"] not in cat_options:
        st.session_state.pop("iv_color_category", None)

    c1, c2 = st.columns(2)
    with c1:
        category = st.selectbox(
            "Color by",
            cat_options,
            index=cat_options.index(UNIFORM) if not cats else 1,
            key="iv_color_category",
        )
    if category == UNIFORM or category not in cats:
        with c2:
            st.selectbox("Level", ["—"], disabled=True, key="iv_color_level")
        return None, None

    items = cats[category]
    labels = [lbl for _, lbl in items]
    state_key = f"iv_level_{category}"
    with c2:
        level_label = st.selectbox(
            "Level",
            labels,
            index=0,
            key=state_key,
        )
    chosen_col = next(c for c, lbl in items if lbl == level_label)
    return chosen_col, level_label


def _run() -> None:
    keep_state_alive()
    inject_styles()
    st.markdown('<div class="sca-tagline">MYSCA · IC VISUALIZED</div>',
                unsafe_allow_html=True)

    bundle = resolve_bundle(
        require_kind="sca_core", require_preprocessing=True,
    )
    if bundle is None:
        return

    try:
        with st.spinner("Building per-sequence Uᵖ table…"):
            df = _cached_sequence_table(
                str(bundle.root), _uniprot_cache_mtime(bundle),
            )
    except Exception as e:
        st.error(
            "Failed to build the sequence-projection table: "
            f"`{type(e).__name__}: {e}`"
        )
        return

    _render_uniprot_panel(bundle, df)
    _render_user_metadata_panel()

    user_df = st.session_state.get(USER_METADATA_STATE_KEY)
    if user_df is not None:
        try:
            df = merge_user_metadata(df, user_df)
        except ValueError as e:
            st.error(f"User metadata merge failed: {e}")

    up_cols = list_up_columns(df)
    if len(up_cols) < 2:
        st.error(
            f"Need at least two IC columns to draw a scatter; got `{up_cols}`."
        )
        return

    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        x_col = st.selectbox(
            "X axis", up_cols, index=0,
            format_func=_ic_label, key="iv_x_col",
        )
    with c2:
        y_col = st.selectbox(
            "Y axis", up_cols,
            index=1 if len(up_cols) > 1 else 0,
            format_func=_ic_label, key="iv_y_col",
        )
    with c3:
        color_col, color_label = _color_choice_widget(df)

    fig, legend_items = _build_figure(df, x_col, y_col, color_col, color_label)
    # Right margin is only needed when Plotly's colorbar (numeric mode)
    # uses it — categorical mode renders a custom HTML legend in its
    # own Streamlit column, so the figure can use the full PLOT_SIZE.
    margin_r = 20 if legend_items else LEGEND_MARGIN_PX
    fig.update_layout(
        plot_bgcolor=PALETTE_BG,
        paper_bgcolor=PALETTE_BG,
        font_color=PALETTE_INK,
        margin=dict(l=20, r=margin_r, t=20, b=20),
        legend=dict(
            orientation="v",
            yanchor="top", y=1,
            xanchor="left", x=1.02,
            title=dict(text=color_label or ""),
        ),
    )
    fig.update_traces(
        marker=dict(size=7, line=dict(width=0.4, color=PALETTE_BG)),
        selector=dict(type="scatter"),
    )

    if legend_items:
        # Custom HTML legend rendered as an iframe (via components.html)
        # so the browser-native `title=` hover tooltips fire reliably —
        # st.markdown sanitizes them away.
        c_plot, c_legend = st.columns([6, 1])
        with c_plot:
            st.plotly_chart(fig, use_container_width=False, key="iv_chart")
        with c_legend:
            components.html(
                _render_html_legend(legend_items, color_label or ""),
                height=_legend_height(legend_items),
                scrolling=True,
            )
    else:
        st.plotly_chart(fig, use_container_width=False, key="iv_chart")

    n_unique = df[color_col].nunique(dropna=False) if color_col else None
    st.caption(
        f"{len(df)} sequences · {len(up_cols)} IC dimensions"
        + (
            f" · color: {color_label} ({n_unique} unique)"
            if color_col is not None else ""
        )
    )

    with st.expander("Preview table", expanded=False):
        preview_cols = ["seq_id", x_col, y_col]
        if color_col is not None and color_col not in preview_cols:
            preview_cols.append(color_col)
        st.dataframe(df[preview_cols].head(50), hide_index=True,
                     use_container_width=True)


_run()
