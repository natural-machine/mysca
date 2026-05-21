"""Shared page chrome: sidebar bundle picker + bundle resolution.

Every page calls :func:`resolve_bundle` near the top to render the
global bundle-path input in the sidebar and get a :class:`ResultsBundle`
back. Returns ``None`` when the user hasn't set a valid path yet; the
page should bail with an informative message and stop rendering.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from mysca.sca_app._launcher import BUNDLE_ENV_VAR
from mysca.sca_app._results_io import ResultsBundle, discover_results

BUNDLE_KEY = "vr_bundle_path"


def _seed_bundle_path() -> None:
    if BUNDLE_KEY not in st.session_state:
        st.session_state[BUNDLE_KEY] = os.environ.get(BUNDLE_ENV_VAR, "")


def render_sidebar_bundle_picker() -> str:
    """Render the bundle-path input in the sidebar; return the current value."""
    _seed_bundle_path()
    with st.sidebar:
        st.text_input(
            "Bundle path",
            key=BUNDLE_KEY,
            help=(
                "Path to an sca-core (or sca-preprocess / sca-project / "
                "sca-structure) output directory. Pre-fill on the CLI via "
                "`sca-app --bundle <path>`."
            ),
        )
    return st.session_state[BUNDLE_KEY]


def resolve_bundle(
    *, require_kind: str | None = None,
    require_preprocessing: bool = False,
) -> ResultsBundle | None:
    """Render the picker and discover the bundle, with friendly bailouts.

    Parameters
    ----------
    require_kind
        If set, displays an error when the discovered bundle's ``kind``
        doesn't match (e.g. ``"sca_core"``).
    require_preprocessing
        If True, displays an error when no matching preprocessing dir
        was resolved alongside an sca-core bundle.

    Returns ``None`` after rendering the relevant message; callers
    should ``return`` immediately in that case.
    """
    path = render_sidebar_bundle_picker()
    if not path or not Path(path).is_dir():
        st.info("Set a valid bundle path in the sidebar to begin.")
        return None
    bundle = discover_results(path)
    if require_kind and bundle.kind != require_kind:
        st.error(
            f"This page needs a `{require_kind}` bundle; got "
            f"`kind={bundle.kind}` at `{bundle.root}`."
        )
        return None
    if require_preprocessing and bundle.preprocessing_dir is None:
        st.error(
            "No sca-preprocess directory found alongside this sca-core "
            "bundle. Set `scarun_args.json`'s `indir` to a valid path "
            "or place the preprocessing dir as a sibling."
        )
        return None
    return bundle
