"""Discover and read files from an mysca results bundle.

A "bundle" is whichever output directory the user points the app at:

  - ``sca-core`` output (preferred entry point — has ``scarun_results.npz``
    and/or ``scarun_args.json``). The matching preprocessing directory is
    recovered from ``scarun_args.json["indir"]`` when it points at an
    existing path, otherwise the app falls back to a sibling
    ``preprocess*`` directory if one is present.
  - ``sca-preprocess`` output (has ``preprocessing_results.npz``). Loaded
    on its own when no sca-core results sit alongside.
  - ``sca-project`` output (has ``projection.json``).
  - ``sca-structure`` output (has ``structure_projection.json``).

The dataclass groups the discovered file paths and exposes lazy loaders
so pages don't need to know the on-disk layout. The actual heavy
deserialization happens in :class:`mysca.results.SCAResults` and
:class:`mysca.results.PreprocessingResults`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mysca.results import (
    COMPONENT_COVERAGE_PER_SEQ_FNAME,
    IC_LOADINGS_PER_SEQ_FNAME,
    IC_RESIDUES_PER_SEQ_FNAME,
    PREPROCESSING_RESULTS_FNAME,
    SCARUN_ARGS_FNAME,
    SCARUN_EIGENDECOMP_FNAME,
    SCARUN_RESULTS_FNAME,
    SEQUENCE_METADATA_FNAME,
    PreprocessingResults,
    SCAResults,
)

BundleKind = Literal[
    "sca_core", "preprocess", "project", "structure", "unknown",
]

PROJECTION_RESULTS_FNAME = "projection.json"
PROJECTION_ARGS_FNAME = "projection_args.json"
STRUCTURE_RESULTS_FNAME = "structure_projection.json"
STRUCTURE_ARGS_FNAME = "structure_args.json"


@dataclass
class ResultsBundle:
    """File-path map for a single results directory.

    Use :func:`discover_results` to construct one. Members are absolute
    paths; missing pieces are ``None`` — pages decide what's required.
    Heavy artifacts are loaded on demand via :meth:`load_sca` /
    :meth:`load_preprocessing`.
    """

    root: Path
    kind: BundleKind = "unknown"

    # sca-core artifacts
    scarun_results_npz: Path | None = None
    scarun_args_json: Path | None = None
    eigendecomp_npz: Path | None = None
    sca_results_dir: Path | None = None
    ic_positions_dir: Path | None = None
    ic_residues_per_seq_npz: Path | None = None
    ic_loadings_per_seq_npz: Path | None = None
    component_coverage_per_seq_npz: Path | None = None
    seq_projections_tsv: Path | None = None
    sequence_metadata_tsv: Path | None = None
    images_dir: Path | None = None

    # sca-preprocess (either co-located or pointed at by sca-core args)
    preprocessing_dir: Path | None = None

    # sca-project / sca-structure (co-located outputs)
    projection_json: Path | None = None
    projection_args_json: Path | None = None
    structure_results_json: Path | None = None
    structure_args_json: Path | None = None

    _scarun_args_cache: dict | None = field(default=None, repr=False)
    _projection_cache: dict | None = field(default=None, repr=False)
    _structure_cache: list | None = field(default=None, repr=False)

    def load_sca(self) -> SCAResults | None:
        """Load the full :class:`SCAResults` for this bundle.

        Returns ``None`` when the bundle has no sca-core artifacts.
        Heavy file IO; cache the result at the call site if needed.
        """
        if not (self.scarun_results_npz or self.scarun_args_json):
            return None
        return SCAResults.load(self.root)

    def load_preprocessing(self) -> PreprocessingResults | None:
        """Load the :class:`PreprocessingResults` for this bundle, if any."""
        if self.preprocessing_dir is None:
            return None
        return PreprocessingResults.load(self.preprocessing_dir)

    def load_scarun_args(self) -> dict | None:
        """Return the sca-core CLI args dict (cached)."""
        if self._scarun_args_cache is None and self.scarun_args_json:
            with open(self.scarun_args_json) as f:
                self._scarun_args_cache = json.load(f)
        return self._scarun_args_cache

    def load_projection(self) -> dict | None:
        """Return the parsed ``projection.json`` (cached)."""
        if self._projection_cache is None and self.projection_json:
            with open(self.projection_json) as f:
                self._projection_cache = json.load(f)
        return self._projection_cache

    def load_structure_projection(self) -> list | None:
        """Return the parsed ``structure_projection.json`` (cached)."""
        if self._structure_cache is None and self.structure_results_json:
            with open(self.structure_results_json) as f:
                self._structure_cache = json.load(f)
        return self._structure_cache


def _classify(root: Path) -> BundleKind:
    if (root / SCARUN_RESULTS_FNAME).is_file() \
            or (root / SCARUN_ARGS_FNAME).is_file():
        return "sca_core"
    if (root / PREPROCESSING_RESULTS_FNAME).is_file():
        return "preprocess"
    if (root / PROJECTION_RESULTS_FNAME).is_file():
        return "project"
    if (root / STRUCTURE_RESULTS_FNAME).is_file():
        return "structure"
    return "unknown"


def _resolve_preprocessing_dir(root: Path, scarun_args: dict | None) -> Path | None:
    """Recover the matching preprocessing dir for an sca-core bundle.

    Tries, in order: the ``indir`` recorded in ``scarun_args.json``;
    sibling directories whose name starts with ``preprocess``; the root
    itself (in case preprocessing was co-located).
    """
    if scarun_args:
        indir = scarun_args.get("indir")
        if indir:
            p = Path(indir)
            if not p.is_absolute():
                p = (root / p).resolve()
            if (p / PREPROCESSING_RESULTS_FNAME).is_file():
                return p
    if (root / PREPROCESSING_RESULTS_FNAME).is_file():
        return root
    parent = root.parent
    if parent.is_dir():
        for sib in sorted(parent.iterdir()):
            if not sib.is_dir() or not sib.name.startswith("preprocess"):
                continue
            if (sib / PREPROCESSING_RESULTS_FNAME).is_file():
                return sib
    return None


def discover_results(root: str | Path) -> ResultsBundle:
    """Build a :class:`ResultsBundle` by scanning ``root``.

    Tolerates partial bundles: any missing file lands as ``None`` on the
    returned dataclass. Callers branch on :attr:`ResultsBundle.kind` or
    on the presence of specific path attributes.
    """
    root = Path(root)
    bundle = ResultsBundle(root=root, kind=_classify(root))

    def _pick(name: str) -> Path | None:
        p = root / name
        return p if p.is_file() else None

    def _pickdir(name: str) -> Path | None:
        p = root / name
        return p if p.is_dir() else None

    bundle.scarun_results_npz = _pick(SCARUN_RESULTS_FNAME)
    bundle.scarun_args_json = _pick(SCARUN_ARGS_FNAME)
    bundle.eigendecomp_npz = _pick(SCARUN_EIGENDECOMP_FNAME)
    bundle.sca_results_dir = _pickdir("sca_results")
    bundle.ic_positions_dir = _pickdir("ic_positions")
    bundle.ic_residues_per_seq_npz = _pick(IC_RESIDUES_PER_SEQ_FNAME)
    bundle.ic_loadings_per_seq_npz = _pick(IC_LOADINGS_PER_SEQ_FNAME)
    bundle.component_coverage_per_seq_npz = _pick(COMPONENT_COVERAGE_PER_SEQ_FNAME)
    bundle.seq_projections_tsv = _pick("seq_projections.tsv")
    bundle.sequence_metadata_tsv = _pick(SEQUENCE_METADATA_FNAME)
    bundle.images_dir = _pickdir("images")

    bundle.projection_json = _pick(PROJECTION_RESULTS_FNAME)
    bundle.projection_args_json = _pick(PROJECTION_ARGS_FNAME)
    bundle.structure_results_json = _pick(STRUCTURE_RESULTS_FNAME)
    bundle.structure_args_json = _pick(STRUCTURE_ARGS_FNAME)

    if bundle.kind == "sca_core":
        bundle.preprocessing_dir = _resolve_preprocessing_dir(
            root, bundle.load_scarun_args(),
        )
    elif bundle.kind == "preprocess":
        bundle.preprocessing_dir = root

    return bundle


def looks_like_results_root(path: str | Path) -> bool:
    """Cheap heuristic: directory contains any recognised mysca output file."""
    p = Path(path)
    if not p.is_dir():
        return False
    return _classify(p) != "unknown"


def find_results_roots(search_root: str | Path, max_depth: int = 3) -> list[Path]:
    """Walk ``search_root`` up to ``max_depth`` levels and return directories
    that pass :func:`looks_like_results_root`."""
    root = Path(search_root)
    if not root.is_dir():
        return []
    found: list[Path] = []
    if looks_like_results_root(root):
        found.append(root)
    for depth in range(1, max_depth + 1):
        for path in root.glob("/".join(["*"] * depth)):
            if path.is_dir() and looks_like_results_root(path):
                found.append(path)
    seen: set[Path] = set()
    out: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out
