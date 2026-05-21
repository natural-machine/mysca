"""mysca — Structure View (home page).

Default landing page: 3D viewer of the reference protein with per-IC
sector highlights and optional conservation shading. The bundle path
lives in the shared sidebar (rendered via ``_chrome.resolve_bundle``)
so every page sees the same value.

PDB source order: pre-computed ``structure_projection.json`` in the
bundle (if present), then live AlphaFold fetch keyed off the reference
UniProt accession, then manual path/upload. Live projection is
computed via ``mysca.structure.project_pdb`` and cached per-PDB.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from mysca.results import SCAResults
from mysca.sca_app._chrome import resolve_bundle
from mysca.sca_app._pdb_io import (
    extract_uniprot_accession,
    fetch_alphafold_pdb,
    list_chains,
    parse_pdb_text,
)
from mysca.sca_app._results_io import ResultsBundle
from mysca.sca_app._structure_view import (
    CONSERVATION_HIGH,
    CONSERVATION_LOW,
    build_view,
    sector_color,
    to_inline_html,
)
from mysca.sca_app._styles import inject_styles, keep_state_alive
from mysca.structure.projection import project_pdb

VIEWER_HEIGHT = 580
PDB_CACHE_SUBDIR = ".pdb_cache"


st.set_page_config(
    page_title="mysca — Structure View",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _reference_id(bundle: ResultsBundle) -> str | None:
    if bundle.preprocessing_dir is None:
        return None
    args_path = bundle.preprocessing_dir / "preprocessing_args.json"
    if not args_path.is_file():
        return None
    with open(args_path) as f:
        args = json.load(f)
    return args.get("reference_id")


def _read_artifacts_from_json(
    bundle: ResultsBundle, pdb_residue_ids: list[int],
) -> dict | None:
    data = bundle.load_structure_projection()
    if not data:
        return None
    entry = data[0] if isinstance(data, list) else data
    sp = entry.get("sequence_projection") or {}
    return {
        "ic_pdb_residues": [
            [int(r) for r in xs] for xs in entry.get("ic_pdb_residues", [])
        ],
        "residue_by_processed_col": list(sp.get("residue_by_processed_col") or []),
        "input_residue_indices": list(sp.get("input_residue_indices") or []),
        "pdb_residue_ids": list(pdb_residue_ids),
    }


@st.cache_data(show_spinner=False)
def _compute_artifacts_live(
    pdb_text: str, chain: str, bundle_root: str,
    preproc_dir: str, seq_id: str | None,
) -> dict:
    """Live ``project_pdb`` → cached dict carrying the IC residues plus
    enough projection-mapping fields to drive conservation shading."""
    pdb = parse_pdb_text(pdb_text, chain=chain)
    projection = project_pdb(
        pdb,
        sca_result_dir=bundle_root,
        preproc_result_dir=preproc_dir,
        seq_id=seq_id,
    )
    sp = projection.sequence_projection
    return {
        "ic_pdb_residues": [
            [int(r) for r in members]
            for members in projection.ic_pdb_residues
        ],
        "residue_by_processed_col": list(sp.residue_by_processed_col),
        "input_residue_indices": list(sp.input_residue_indices),
        "pdb_residue_ids": list(pdb.residue_ids),
    }


@st.cache_data(show_spinner=False)
def _load_conservation(bundle_root: str) -> np.ndarray | None:
    sca = SCAResults.load(bundle_root)
    if sca.conservation is None:
        return None
    return np.asarray(sca.conservation, dtype=np.float64)


def _conservation_by_pdb_residue(
    artifacts: dict, conservation: np.ndarray,
) -> dict[int, float]:
    """proc_col → raw_idx → input_idx → pdb_residue_num."""
    by_resnum: dict[int, float] = {}
    pdb_ids = artifacts["pdb_residue_ids"]
    input_idxs = artifacts["input_residue_indices"]
    res_by_col = artifacts["residue_by_processed_col"]
    if len(res_by_col) != len(conservation):
        return by_resnum
    for proc_col, raw_idx in enumerate(res_by_col):
        if raw_idx is None:
            continue
        try:
            input_idx = input_idxs[int(raw_idx)]
            resnum = int(pdb_ids[int(input_idx)])
        except (IndexError, TypeError):
            continue
        by_resnum[resnum] = float(conservation[proc_col])
    return by_resnum


def _acquire_pdb(bundle: ResultsBundle, reference_id: str | None) -> str | None:
    accession = (
        extract_uniprot_accession(reference_id) if reference_id else None
    )
    sources = ["AlphaFold (auto-fetch)", "Local path", "Upload"]
    default_idx = 0 if accession else 1
    source = st.radio(
        "PDB source", sources, index=default_idx, horizontal=True,
        key="ss_pdb_source",
    )

    if source == "AlphaFold (auto-fetch)":
        if not accession:
            st.warning(
                "Could not extract a UniProt accession from "
                f"`{reference_id}`. Switch to *Local path* or *Upload*."
            )
            return None
        acc_in = st.text_input(
            "UniProt accession", value=accession, key="ss_uniprot_acc",
            help=(
                "Pre-filled from the bundle's reference_id; edit to override. "
                "The latest AlphaFold model version is auto-resolved via the "
                "AlphaFold prediction API."
            ),
        )
        if not acc_in.strip():
            return None
        cache_dir = bundle.root / PDB_CACHE_SUBDIR
        try:
            with st.spinner(f"Fetching AlphaFold model for {acc_in}…"):
                pdb_path = fetch_alphafold_pdb(
                    acc_in.strip(), cache_dir=cache_dir,
                )
        except urllib.error.HTTPError as e:
            st.error(
                f"AlphaFold has no model for `{acc_in}` (HTTP {e.code}). "
                "Try a different accession or switch to manual upload."
            )
            return None
        except urllib.error.URLError as e:
            st.error(f"Network error while fetching AlphaFold: {e.reason}")
            return None
        except ValueError as e:
            st.error(f"AlphaFold response could not be parsed: {e}")
            return None
        return pdb_path.read_text()

    if source == "Local path":
        path_str = st.text_input(
            "PDB path", key="ss_pdb_path",
            placeholder="/abs/path/to/structure.pdb",
        )
        if not path_str.strip():
            return None
        p = Path(path_str.strip()).expanduser()
        if not p.is_file():
            st.error(f"Not a file: `{p}`")
            return None
        return p.read_text()

    f = st.file_uploader("Upload PDB", type=["pdb", "ent"], key="ss_pdb_upload")
    if f is None:
        return None
    return f.read().decode("utf-8")


def _ic_controls(
    ic_residues: dict[int, list[int]],
) -> tuple[set[int], dict[int, str], str, str, str]:
    backbone_style = st.radio(
        "Backbone style",
        ["cartoon", "spheres"],
        index=0,
        key="ss_backbone_style",
        help=(
            "*cartoon*: smooth ribbon trace of the chain. *spheres*: every "
            "atom as a space-filling sphere (CPK view) — IC residues "
            "recolor in place instead of getting an overlay."
        ),
    )
    backbone_mode = st.radio(
        "Backbone shading",
        ["solid", "by conservation"],
        index=0,
        key="ss_backbone_mode",
        help=(
            "*solid*: flat gray backbone. *by conservation*: cream → black "
            "ramp using `SCAResults.conservation` (10–90th percentile "
            "clipped for visibility) mapped onto PDB residues."
        ),
    )
    # Reserve a fixed-height row so the gradient legend doesn't change
    # the vertical spacing between the two radios.
    if backbone_mode == "by conservation":
        legend_inner = (
            f"<span style='background:{CONSERVATION_LOW}; padding:0 8px;'"
            f">low</span>"
            f"<span style='background:linear-gradient(to right,"
            f"{CONSERVATION_LOW},{CONSERVATION_HIGH}); padding:0 32px;'>"
            f"</span>"
            f"<span style='background:{CONSERVATION_HIGH}; color:white;"
            f" padding:0 8px;'>high</span>"
        )
    else:
        legend_inner = ""
    st.markdown(
        f"<div style='font-size:0.8rem; height:1.4rem; line-height:1.4rem; "
        f"margin:-0.2rem 0 0.4rem 0;'>{legend_inner}</div>",
        unsafe_allow_html=True,
    )
    # Highlight-style is meaningful only when the backbone is a cartoon
    # — spheres backbone recolors IC residues in place.
    if backbone_style == "cartoon":
        style = st.radio(
            "Highlight style",
            ["spheres", "sticks", "cartoon"],
            index=0,
            key="ss_highlight_style",
        )
    else:
        style = "spheres"

    head_l, head_r = st.columns([2, 1])
    with head_l:
        st.markdown("**Sectors**")
    with head_r:
        if st.button("clear", key="ss_deselect_all", type="tertiary"):
            for ic_idx in ic_residues:
                st.session_state[f"ss_ic_on_{ic_idx}"] = False
            st.rerun()

    enabled: set[int] = set()
    colors: dict[int, str] = {}
    for ic_idx in sorted(ic_residues.keys()):
        c_on, c_color = st.columns([3, 1])
        with c_on:
            on = st.checkbox(
                f"IC {ic_idx + 1}  ({len(ic_residues[ic_idx])} res)",
                value=True,
                key=f"ss_ic_on_{ic_idx}",
            )
        with c_color:
            color = st.color_picker(
                f"color_{ic_idx}",
                value=sector_color(ic_idx),
                key=f"ss_ic_color_{ic_idx}",
                label_visibility="collapsed",
            )
        if on:
            enabled.add(ic_idx)
        colors[ic_idx] = color
    return enabled, colors, style, backbone_mode, backbone_style


def _residue_table(
    ic_residues: dict[int, list[int]],
    enabled: set[int],
    colors: dict[int, str],
) -> None:
    rows = []
    for ic_idx in sorted(ic_residues.keys()):
        if ic_idx not in enabled:
            continue
        rows.append({
            "IC": f"IC {ic_idx + 1}",
            "color": colors[ic_idx],
            "n_residues": len(ic_residues[ic_idx]),
            "PDB residues": ", ".join(str(r) for r in ic_residues[ic_idx]),
        })
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)


def _run() -> None:
    keep_state_alive()
    inject_styles()
    st.markdown('<div class="sca-tagline">MYSCA · STRUCTURE VIEW</div>',
                unsafe_allow_html=True)

    bundle = resolve_bundle(
        require_kind="sca_core", require_preprocessing=True,
    )
    if bundle is None:
        return

    reference_id = _reference_id(bundle)
    st.caption(f"Reference: `{reference_id or '(unknown)'}`")

    pdb_text = _acquire_pdb(bundle, reference_id)
    if pdb_text is None:
        return

    probe = parse_pdb_text(pdb_text)
    chains = list_chains(probe)
    default_chain = probe.chain_id
    chain = st.selectbox(
        "Chain", chains,
        index=chains.index(default_chain) if default_chain in chains else 0,
        key="ss_chain",
    )

    artifacts = _read_artifacts_from_json(bundle, probe.residue_ids)
    if artifacts is None:
        try:
            with st.spinner("Mapping IC residues onto the PDB…"):
                artifacts = _compute_artifacts_live(
                    pdb_text, chain, str(bundle.root),
                    str(bundle.preprocessing_dir), reference_id,
                )
        except Exception as e:
            st.error(
                f"Failed to project IC residues onto the PDB: "
                f"`{type(e).__name__}: {e}`"
            )
            return

    ic_residues = {
        i: list(residues)
        for i, residues in enumerate(artifacts["ic_pdb_residues"])
    }
    if not any(ic_residues.values()):
        st.warning(
            "Every IC came back with zero residues — likely a PDB/sequence "
            "mismatch. Try a different chain or PDB."
        )
        return

    col_viewer, col_ic = st.columns([4, 1])
    with col_ic:
        (enabled, colors, style, backbone_mode,
         backbone_style) = _ic_controls(ic_residues)
    with col_viewer:
        backbone_conservation: dict[int, float] | None = None
        if backbone_mode == "by conservation":
            conservation = _load_conservation(str(bundle.root))
            if conservation is None:
                st.warning(
                    "No `conservation` array in the SCA bundle; falling "
                    "back to solid backbone."
                )
            else:
                backbone_conservation = _conservation_by_pdb_residue(
                    artifacts, conservation,
                )
                if not backbone_conservation:
                    st.warning(
                        "Could not map conservation onto any PDB residue. "
                        "Falling back to solid backbone."
                    )
                    backbone_conservation = None
        view = build_view(
            pdb_text,
            chain_id=chain,
            ic_residues={i: ic_residues[i] for i in enabled},
            colors=colors,
            height=VIEWER_HEIGHT,
            style=style,
            backbone_style=backbone_style,
            backbone_conservation=backbone_conservation,
        )
        components.html(to_inline_html(view), height=VIEWER_HEIGHT + 20)

    with st.expander("Residue lists", expanded=False):
        _residue_table(ic_residues, enabled, colors)


_run()
