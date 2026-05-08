"""Constants and shared palette helpers."""

from __future__ import annotations

import json
import os

AA_STD20 = "ACDEFGHIKLMNPQRSTVWY"

DEFAULT_BACKGROUND_FREQ = {
    'A': 0.073, 'C': 0.025, 'D': 0.050, 'E': 0.061,
    'F': 0.042, 'G': 0.072, 'H': 0.023, 'I': 0.053,
    'K': 0.064, 'L': 0.089, 'M': 0.023, 'N': 0.043,
    'P': 0.052, 'Q': 0.040, 'R': 0.052, 'S': 0.073,
    'T': 0.056, 'V': 0.063, 'W': 0.013, 'Y': 0.033,
}

SECTOR_COLORS = [
    "#e377c2",
    "#f62727",
    "#8c564b",
    "#1f77b4",
    "#ff00fb",
    "#2ca02c",
    "#60ffda",
    "#bcbd22",
    "#c52626",
    "#17becf",
    "#ff7f0e",
    "#DBF3AF",
    "#393b79",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
    "#31a354",
    "#756bb1",
    "#636363",
]


def resolve_sector_colors(spec: str | None) -> list[str] | None:
    """Resolve a ``--sector_colors`` CLI value into a palette.

    Returns ``None`` when no per-sector coloring is requested
    (``spec="none"``); otherwise a list of hex color strings.

    Accepted forms (checked in order):

    * ``"none"`` → ``None`` (suppress per-sector coloring).
    * ``"default"`` or ``None`` → :data:`SECTOR_COLORS`.
    * Comma-separated list of matplotlib-parseable color tokens, e.g.
      ``"#e377c2,#f62727,red"``. Whitespace around tokens is stripped.
    * Path to an existing file. ``.json`` is parsed as a JSON array of
      strings; any other suffix is read as one color per non-blank
      line (no comment syntax — every non-blank line must be a valid
      color, since ``#`` starts hex colors).
    * Name of a registered matplotlib colormap (e.g. ``"tab10"``,
      ``"Set1"``, ``"viridis"``). Qualitative palettes (any
      ``ListedColormap`` with ``N <= 32``) are taken verbatim;
      everything else is sampled at ``len(SECTOR_COLORS)`` evenly
      spaced points.

    Raises ``ValueError`` for malformed input or unrecognised cmap
    names. matplotlib is imported lazily so callers that never pass a
    cmap name (notably sca-pymol) do not pay for the import.
    """
    if spec is None or spec == "default":
        return list(SECTOR_COLORS)
    if spec == "none":
        return None
    if "," in spec:
        tokens = [t.strip() for t in spec.split(",")]
        tokens = [t for t in tokens if t]
        if not tokens:
            raise ValueError(
                f"--sector_colors={spec!r} parsed to an empty list."
            )
        _validate_color_tokens(tokens, source=f"--sector_colors={spec!r}")
        return [_to_hex(t) for t in tokens]
    if os.path.isfile(spec):
        return _load_colors_from_file(spec)
    return _sample_cmap(spec)


def _validate_color_tokens(tokens: list[str], *, source: str) -> None:
    from matplotlib import colors as mcolors  # local: matplotlib is heavy
    bad = [t for t in tokens if not mcolors.is_color_like(t)]
    if bad:
        raise ValueError(
            f"{source} contains unrecognised color tokens: {bad}. "
            "Each token must be a matplotlib-parseable color (hex like "
            "'#e377c2', a named color like 'red', or an 'rgb(...)' "
            "string)."
        )


def _to_hex(token: str) -> str:
    from matplotlib import colors as mcolors
    return mcolors.to_hex(token)


def _load_colors_from_file(path: str) -> list[str]:
    if path.lower().endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list) or not all(
            isinstance(x, str) for x in data
        ):
            raise ValueError(
                f"--sector_colors={path!r}: JSON file must contain a "
                "flat array of color strings."
            )
        tokens = data
    else:
        with open(path) as f:
            tokens = [line.strip() for line in f if line.strip()]
    if not tokens:
        raise ValueError(
            f"--sector_colors={path!r} contained no color entries."
        )
    _validate_color_tokens(tokens, source=f"--sector_colors={path!r}")
    return [_to_hex(t) for t in tokens]


def _sample_cmap(name: str) -> list[str]:
    from matplotlib import colormaps, colors as mcolors
    try:
        cmap = colormaps[name]
    except KeyError as exc:
        raise ValueError(
            f"--sector_colors={name!r}: not a path, comma-list, "
            "'default', 'none', or a registered matplotlib colormap. "
            "Use a hex list (e.g. '#ff0000,#00ff00'), a file path, or "
            "a cmap name like 'tab10' / 'Set1'."
        ) from exc
    if isinstance(cmap, mcolors.ListedColormap) and cmap.N <= 32:
        rgba = list(cmap.colors)
    else:
        n = len(SECTOR_COLORS)
        rgba = [cmap(i / max(n - 1, 1)) for i in range(n)]
    return [mcolors.to_hex(c) for c in rgba]
