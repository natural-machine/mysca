"""Run summary — what's inside the current bundle.

Top-level inventory + the CLI args sca-preprocess and sca-core were
invoked with, plus a "files present" table. Useful for sanity-checking
that a bundle has everything the other pages need.
"""

from __future__ import annotations

import json

import streamlit as st

from mysca.sca_app._chrome import resolve_bundle
from mysca.sca_app._results_io import ResultsBundle
from mysca.sca_app._styles import inject_styles, keep_state_alive

KIND_LABELS = {
    "sca_core": "sca-core output",
    "preprocess": "sca-preprocess output",
    "project": "sca-project output",
    "structure": "sca-structure output",
    "unknown": "(no recognised mysca output detected)",
}


st.set_page_config(
    page_title="mysca — Run Summary",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _format_scalar(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") or "0"
    return str(v)


def _render_headline_metrics(
    bundle: ResultsBundle, sca_args: dict,
) -> None:
    """The 'in big' metrics. Pulls live values from the saved SCAResults
    rather than just the CLI args, so post-hoc kstar overrides show up.
    """
    from mysca.results import SCAResults
    sca = SCAResults.load(bundle.root)

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Significant eigenvalues (k★)",
        _format_scalar(sca.kstar),
        help=(
            "Number of top eigenvalues of the SCA covariance matrix "
            "judged significant by the bootstrap null. The corresponding "
            "eigenvectors define the directions ICA decomposes into "
            "sectors. Set automatically by `--n_boot`; can be overridden "
            "with `-k / --kstar`."
        ),
    )
    c2.metric(
        "ICs computed (n_components)",
        _format_scalar(sca.n_components),
        help=(
            "Number of independent components ICA produces. Always ≥ k★. "
            "Defaults to k★ unless `--n_components` raised it (e.g. "
            "`--n_components all` runs ICA on every eigenvector). Each IC "
            "is a candidate sector."
        ),
    )
    c3.metric(
        "Bootstrap iterations (n_boot)",
        _format_scalar(sca_args.get("n_boot")),
        help=(
            "How many shuffles of the MSA fed the null spectrum used to "
            "pick the eigenvalue significance cutoff. More = more stable "
            "k★ estimate, but linearly more compute. `n_boot=0` reuses "
            "an existing bootstrap; `n_boot=-1` skips it entirely."
        ),
    )


def _preproc_args(bundle: ResultsBundle) -> dict | None:
    if bundle.preprocessing_dir is None:
        return None
    p = bundle.preprocessing_dir / "preprocessing_args.json"
    if not p.is_file():
        return None
    with open(p) as f:
        return json.load(f)


def _render_inventory(bundle: ResultsBundle) -> None:
    rows = [
        ("scarun_results.npz", bundle.scarun_results_npz),
        ("scarun_args.json", bundle.scarun_args_json),
        ("sca_eigendecomp.npz", bundle.eigendecomp_npz),
        ("sca_results/", bundle.sca_results_dir),
        ("ic_positions/", bundle.ic_positions_dir),
        ("ic_residues_per_seq.npz", bundle.ic_residues_per_seq_npz),
        ("ic_loadings_per_seq.npz", bundle.ic_loadings_per_seq_npz),
        ("component_coverage_per_seq.npz", bundle.component_coverage_per_seq_npz),
        ("seq_projections.tsv", bundle.seq_projections_tsv),
        ("sequence_metadata.tsv", bundle.sequence_metadata_tsv),
        ("images/", bundle.images_dir),
        ("preprocessing dir", bundle.preprocessing_dir),
        ("projection.json", bundle.projection_json),
        ("structure_projection.json", bundle.structure_results_json),
    ]
    st.dataframe(
        {
            "artifact": [r[0] for r in rows],
            "present": ["yes" if r[1] is not None else "no" for r in rows],
            "path": [str(r[1]) if r[1] is not None else "" for r in rows],
        },
        hide_index=True, use_container_width=True,
    )


def _run() -> None:
    keep_state_alive()
    inject_styles()
    st.markdown('<div class="sca-tagline">MYSCA · RUN SUMMARY</div>',
                unsafe_allow_html=True)

    bundle = resolve_bundle()
    if bundle is None:
        return

    st.caption(
        f"Detected: **{KIND_LABELS[bundle.kind]}** at `{bundle.root}`"
    )

    sca_args = bundle.load_scarun_args()
    if sca_args:
        _render_headline_metrics(bundle, sca_args)

    pre_args = _preproc_args(bundle)
    if pre_args:
        with st.expander("sca-preprocess args", expanded=False):
            st.json(pre_args)

    if sca_args:
        with st.expander("sca-core args (full)", expanded=False):
            st.json(sca_args)

    st.markdown("**Files in this bundle**")
    _render_inventory(bundle)


_run()
