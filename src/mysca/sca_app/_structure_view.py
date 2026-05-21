"""py3Dmol structure-rendering helpers for the structure-view page.

Builds a cartoon view of a single chain with two optional overlays:

  - per-residue conservation shading on the cartoon backbone (light →
    dark cream-to-black ramp), and
  - per-IC residue sets drawn as spheres / sticks / cartoon recolors.

When no conservation values are supplied the backbone gets a flat gray
so the sector highlights read against context.
"""

from __future__ import annotations

import math

import py3Dmol

DEFAULT_BG_COLOR = "#D9D9D9"

# App palette (cohesive across pages). Two background-ish anchors plus
# four accent colors used as the default sector palette in the order
# below. Wraps to the front past the last entry.
PALETTE_BG = "#FBF3EF"
PALETTE_INK = "#000000"
PALETTE_TEAL = "#74B1C1"
PALETTE_GREEN = "#B2D26E"
PALETTE_YELLOW = "#DFBA51"
PALETTE_RED = "#E04C24"

SECTOR_PALETTE = [
    PALETTE_TEAL,
    PALETTE_GREEN,
    PALETTE_YELLOW,
    PALETTE_RED,
    PALETTE_INK,
]

# Extended palette for categorical plots with more than 5 bins (e.g.
# IC scatter colored by `uniprot_family` with 95 unique values). Builds
# out from the 5 anchor colors with darker / lighter / harmonic variants
# so the palette stays visually cohesive with the rest of the app.
# Hand-tuned so consecutive entries don't read as the same color.
EXTENDED_PALETTE = [
    PALETTE_TEAL,    # #74B1C1
    PALETTE_GREEN,   # #B2D26E
    PALETTE_YELLOW,  # #DFBA51
    PALETTE_RED,     # #E04C24
    PALETTE_INK,     # #000000
    "#3D7B8E",       # dark teal
    "#7BA73B",       # dark green
    "#A88B2E",       # dark mustard
    "#A03414",       # dark red
    "#4A6371",       # slate
    "#A3CFD9",       # light teal
    "#CFE39E",       # light green
    "#EBD180",       # light mustard
    "#EE876A",       # coral
    "#7A5D87",       # muted plum
    "#5C7A6F",       # pine
    "#9CA858",       # olive
    "#8B6F47",       # warm brown
    "#D49B95",       # rose
    "#6B6155",       # warm gray
]

CONSERVATION_LOW = PALETTE_BG
CONSERVATION_HIGH = PALETTE_INK


def sector_color(ic_idx: int) -> str:
    """Default color for IC ``ic_idx`` (0-based).

    Indexes into :data:`EXTENDED_PALETTE` (20 cohesive colors) so bundles
    with more than 5 ICs don't reuse hues. The first 5 entries are the
    same as :data:`SECTOR_PALETTE`, so the visual identity stays
    consistent with the rest of the app for small IC counts.
    """
    return EXTENDED_PALETTE[ic_idx % len(EXTENDED_PALETTE)]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def conservation_color(
    value: float | None,
    vmin: float,
    vmax: float,
    *,
    low_hex: str = CONSERVATION_LOW,
    high_hex: str = CONSERVATION_HIGH,
) -> str:
    """Linear interpolation between ``low_hex`` and ``high_hex``.

    ``value=None`` or ``NaN`` returns the default backbone gray, so
    residues with no SCA coverage (gaps, dropped columns) stay neutral
    rather than landing at ``low_hex``.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return DEFAULT_BG_COLOR
    if vmax > vmin:
        t = (value - vmin) / (vmax - vmin)
    else:
        t = 0.5
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = _hex_to_rgb(low_hex)
    r2, g2, b2 = _hex_to_rgb(high_hex)
    return _rgb_to_hex((
        round(r1 + (r2 - r1) * t),
        round(g1 + (g2 - g1) * t),
        round(b1 + (b2 - b1) * t),
    ))


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolated quantile of a *pre-sorted* float list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    rank = p * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def conservation_value_range(
    values: list[float] | dict[int, float],
    percentile_clip: float = 0.10,
) -> tuple[float, float]:
    """Compute the ramp endpoints used for conservation shading.

    Defaults to clipping the bottom and top 10% of values so the bulk
    of the distribution spans the full cream→black ramp — this gives a
    much more dramatic visual than naive ``(min, max)`` when a few
    outlier positions compress the rest of the range.
    """
    if isinstance(values, dict):
        vals = list(values.values())
    else:
        vals = list(values)
    vals = sorted(v for v in vals if v is not None and not math.isnan(v))
    if not vals:
        return 0.0, 1.0
    clip = max(0.0, min(0.49, percentile_clip))
    return _percentile(vals, clip), _percentile(vals, 1 - clip)


def build_view(
    pdb_str: str,
    chain_id: str,
    ic_residues: dict[int, list[int]],
    colors: dict[int, str],
    *,
    width: int = 760,
    height: int = 560,
    style: str = "spheres",
    backbone_style: str = "cartoon",
    backbone_conservation: dict[int, float] | None = None,
    conservation_range: tuple[float, float] | None = None,
    conservation_percentile_clip: float = 0.10,
) -> py3Dmol.view:
    """Render ``pdb_str`` with per-IC residue highlights.

    Parameters
    ----------
    pdb_str
        Full PDB file content.
    chain_id
        Chain to render. Residues on other chains are not drawn.
    ic_residues
        ``{ic_index: [pdb_residue_number, ...]}`` for the user-enabled
        sectors only.
    colors
        ``{ic_index: hex_color}`` aligned with ``ic_residues`` keys.
    style
        How highlighted residues are drawn on top of the cartoon
        backbone. Ignored when ``backbone_style="spheres"`` — IC
        residues are then re-colored in place rather than overlaid.
    backbone_style
        ``"cartoon"`` (default) renders the chain as a cartoon trace
        with IC residues overlaid in ``style``. ``"spheres"`` renders
        every atom as a van-der-Waals sphere (CPK / space-filling
        view) and re-colors IC residues directly.
    backbone_conservation
        Optional ``{pdb_residue_number: conservation_value}`` map. When
        provided, the backbone is shaded via :func:`conservation_color`
        instead of getting a flat color. PDB residues absent from the
        map keep the default gray.
    conservation_range
        Optional explicit ``(vmin, vmax)`` for the gradient; derived
        from the values in ``backbone_conservation`` if omitted.
    """
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_str, "pdb")

    is_spheres = backbone_style == "spheres"
    base_style_key = "sphere" if is_spheres else "cartoon"

    # The cartoon backbone uses 0.9 opacity to look airy; spheres are
    # already volumetric, so 1.0 keeps them crisp.
    base_opacity = 1.0 if is_spheres else 0.9

    def _set(selector, color, opacity=base_opacity):
        view.setStyle(
            selector, {base_style_key: {"color": color, "opacity": opacity}},
        )

    _set({"chain": chain_id}, DEFAULT_BG_COLOR)
    if backbone_conservation:
        if conservation_range is None:
            vmin, vmax = conservation_value_range(
                backbone_conservation,
                percentile_clip=conservation_percentile_clip,
            )
        else:
            vmin, vmax = conservation_range
        for resnum, val in backbone_conservation.items():
            _set(
                {"chain": chain_id, "resi": int(resnum)},
                conservation_color(val, vmin, vmax),
            )

    for ic_idx, residues in ic_residues.items():
        if not residues:
            continue
        color = colors.get(ic_idx, PALETTE_RED)
        sel = {"chain": chain_id, "resi": [int(r) for r in residues]}
        if is_spheres:
            # Recolor the existing sphere atoms — no overlay needed
            # since the chain is already drawn as spheres everywhere.
            _set(sel, color, opacity=1.0)
        elif style == "cartoon":
            view.setStyle(sel, {"cartoon": {"color": color}})
        elif style == "sticks":
            view.addStyle(sel, {"stick": {"color": color, "radius": 0.25}})
        else:
            view.addStyle(
                sel,
                {"sphere": {"color": color, "radius": 1.4, "opacity": 0.95}},
            )

    view.zoomTo({"chain": chain_id})
    return view


def to_inline_html(view: py3Dmol.view) -> str:
    """Return inline HTML for ``components.html(...)``."""
    return view.write_html()
