"""Page-level styling + widget-state persistence helpers.

Streamlit injects ``st.markdown(CSS)`` into the rendered DOM only while
the calling page is the *current* page. Switching pages rips that style
block out, so every page calls :func:`inject_styles`.

Streamlit also garbage-collects widget ``session_state`` entries when
their widget is no longer rendered (e.g. on page navigation). To keep
inputs alive across page switches, every page touches the keys it owns
via :func:`keep_state_alive` *before* its widgets are recreated.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

PERSISTED_WIDGET_KEYS: frozenset[str] = frozenset({
    "vr_bundle_path",
    "vr_view",
    # Structure view (home)
    "ss_pdb_source", "ss_uniprot_acc",
    "ss_pdb_path", "ss_chain",
    "ss_highlight_style", "ss_backbone_mode", "ss_backbone_style",
    # IC visualized
    "iv_x_col", "iv_y_col",
    "iv_color_category", "iv_color_level",
    # Conservation
    "cv_coord_mode", "cv_topn_toggle", "cv_topn_value",
    # Spectrum
    "sp_zoom_toggle", "sp_top_k",
})

PERSISTED_WIDGET_PREFIXES: tuple[str, ...] = (
    "ss_ic_on_", "ss_ic_color_",
    "iv_level_",  # per-category "Level" selectboxes on the IC page
)

CSS = """
<style>
  /* Page palette: black, cream, mustard, light green, teal, red-orange. */
  [data-testid="stAppViewContainer"],
  [data-testid="stHeader"],
  [data-testid="stSidebar"],
  .main, .stApp, body {
    background-color: #FBF3EF !important;
  }
  .block-container { padding-top: 4rem !important; padding-bottom: 1rem !important; }
  h1, h2, h3 { margin-top: 0.2rem !important; margin-bottom: 0.4rem !important; }
  [data-testid="stVerticalBlock"] { gap: 0.5rem !important; }

  .sca-tagline {
    color: #E04C24;
    font-weight: 700;
    font-size: 0.95rem;
    margin: 0 0 0.8rem 0;
    letter-spacing: 0.04rem;
  }

  div[data-testid="stButton"] > button {
    background-color: #74B1C1 !important;
    color: #000000 !important;
    border: 1px solid #74B1C1 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
  }
  div[data-testid="stButton"] > button:hover {
    background-color: #B2D26E !important;
    border-color: #B2D26E !important;
  }
  /* Tertiary buttons (e.g. select-all / deselect-all) render as small
     unobtrusive text links — the global rule above is too loud for them. */
  div[data-testid="stButton"] > button[kind="tertiary"] {
    background-color: transparent !important;
    color: #74B1C1 !important;
    border: none !important;
    padding: 0.1rem 0.4rem !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
  }
  div[data-testid="stButton"] > button[kind="tertiary"]:hover {
    background-color: transparent !important;
    color: #E04C24 !important;
    text-decoration: underline !important;
  }
</style>
"""

_RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Nimbus Sans",
                        "Liberation Sans", "Arial", "DejaVu Sans"],
}


def inject_styles() -> None:
    """Inject the global CSS and matplotlib rcParams.

    Call this near the top of every Streamlit page. The CSS is required
    on each page because Streamlit re-renders pages independently.
    """
    st.markdown(CSS, unsafe_allow_html=True)
    plt.rcParams.update(_RC_PARAMS)


def keep_state_alive() -> None:
    """Re-bind input-widget ``session_state`` keys so Streamlit doesn't
    garbage-collect them on page navigation.

    Call this *at the top* of every page, before any widget is created.
    Only keys in :data:`PERSISTED_WIDGET_KEYS` or matching
    :data:`PERSISTED_WIDGET_PREFIXES` are touched. Buttons must be
    excluded — touching a button's key raises on its next render.
    """
    for k in list(st.session_state.keys()):
        is_input = (
            k in PERSISTED_WIDGET_KEYS
            or any(k.startswith(p) for p in PERSISTED_WIDGET_PREFIXES)
        )
        if not is_input:
            continue
        try:
            st.session_state[k] = st.session_state[k]
        except Exception:
            continue
