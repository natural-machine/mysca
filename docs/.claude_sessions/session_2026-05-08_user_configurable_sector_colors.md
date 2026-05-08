# 2026-05-08 — user-configurable sector palette

## Summary

Replaced the hard-coded sector palette (`mysca.constants.SECTOR_COLORS`)
with a user-configurable `--sector_colors SPEC` flag exposed by the
three CLIs that render sectors: `sca-core`, `sca-pymol`, and
`sca-plots`. SPEC accepts five forms — the literal string `"default"`
(built-in 20-color palette), `"none"` (skip per-sector coloring; not
allowed by `sca-pymol`), a comma-separated list of hex / named colors,
a path to a `.json` array or one-color-per-line text file, or the name
of a registered matplotlib colormap (e.g. `tab10`, `Set1`, `viridis`).

Replaces the previous `--sector_cmap {default,none}` flag on
`sca-core`. The two original choices have direct equivalents
(`--sector_colors default` / `--sector_colors none`) but the flag name
itself is breaking — there's no shim or alias.

## Why

User wanted a way to override the built-in palette without editing
source. The previous `--sector_cmap` only toggled "default vs nothing".
Three input formats give an ergonomic ladder: comma-list for one-off
runs, file for shared / curated palettes, and matplotlib cmap names
for canonical palettes (which is what users coming from matplotlib
naturally reach for).

`--sector_cmap` was kept narrow on purpose at the time, but renaming
to `--sector_colors` makes the new semantics clearer and the old
choice-based name was misleading once arbitrary input was allowed.
`sca-pymol` and `sca-plots` previously had no override at all — they
imported `SECTOR_COLORS` directly from `mysca.constants` — so adding
the flag there is purely additive.

## What changed

### Resolver (new)

`src/mysca/constants.py`: added `resolve_sector_colors(spec) -> list[str] | None`
next to the existing `SECTOR_COLORS` constant.

- `"default"` / `None` → returns a copy of `SECTOR_COLORS` (defensive
  copy so a caller mutating the result doesn't poison the module-level
  list).
- `"none"` → returns `None` (suppress per-sector coloring).
- Spec containing `,` → split + strip → comma-list, every token
  validated via `matplotlib.colors.is_color_like` and normalized to
  hex via `to_hex`.
- Spec that resolves to an existing file → `.json` parsed as a flat
  array of strings; any other suffix read as one color per non-blank
  line. **No comment syntax** — `#` starts hex colors so any leading-
  `#` line must be a real color.
- Else treated as a registered matplotlib colormap name. If the cmap
  is `ListedColormap` with `N <= 32` (covers `tab10`/`Set1`/`tab20`/
  `Paired` etc.) we take its colors verbatim; otherwise sample
  `len(SECTOR_COLORS)` evenly-spaced points (covers `viridis`,
  `plasma`, etc., which are technically `ListedColormap` of N=256).
- Unknown cmap names raise `ValueError` with a hint.

matplotlib is imported lazily inside the validator / cmap branches so
`sca-pymol` (which deliberately avoids matplotlib) doesn't pay for the
import on the `default` / comma-list / file paths.

### sca-core (`run_sca.py`)

- Removed `--sector_cmap` argument and the inline `{"default":
  SECTOR_COLORS, "none": None}` lookup.
- Added `--sector_colors SPEC` (default `"default"`).
- `sector_color_set` is now `resolve_sector_colors(args.sector_colors)`;
  passed unchanged into the existing `sector_color_set` plumbing.
- Updated the module docstring's COMMAND LINE ARGUMENTS section.

### sca-pymol (`run_pymol.py`)

- Added `--sector_colors SPEC` (default `"default"`).
- `parse_args` rejects `--sector_colors none` with a clear error since
  PyMOL rendering needs a palette.
- Replaced the module constant `DEFAULT_SECTOR_COLORS = SECTOR_COLORS`
  (now unused) with a per-run `palette =
  resolve_sector_colors(args.sector_colors)`.
- Updated the module docstring.

### sca-plots (`run_plots.py`)

- Added `--sector_colors SPEC` (default `"default"`).
- Plumbed through `_replay_scacore(..., sector_colors=...)`; the call
  site that used to import `SECTOR_COLORS` locally now uses the
  resolved palette.
- Updated the module docstring.

### Docs

- `docs/cli_reference.md`: updated all three sector-color rows
  (sca-core, sca-pymol, sca-plots) and the matching usage-block flag
  lists.
- `README.md`: added a `--sector_colors` bullet to the sca-core "Extra
  arguments" list, alongside the existing `--seq_proj_color_by`
  bullet. The sca-pymol and sca-plots README sections defer to
  `cli_reference.md` for flag details, so no inline addition there.

### Tests

`tests/test_sector_colors.py` (new) — 19 tests:

- Resolver: default/none/None, comma-list happy + error paths, text
  file + error paths, JSON file + error paths, ListedColormap branch
  (`tab10`), continuous-cmap branch (`viridis`), unknown cmap name.
- CLI: `parse_args` for all three entrypoints accepts the default and
  an override; `sca-core` rejects the now-removed `--sector_cmap`;
  `sca-pymol` rejects `--sector_colors none` at parse time.

Also re-ran `tests/test_entrypoint_{pymol,plots,scarun}.py` (156
tests, 2 skipped) — all green.

## Lingering / not done

- **No backward-compatibility alias for `--sector_cmap`.** A user with
  a saved invocation using `--sector_cmap default` will see argparse
  reject the flag. We discussed keeping both flags with `--sector_colors`
  winning, but the user picked the cleaner "replace" option. `0.1.x`
  pre-1.0, low blast radius.
- **No CLI `--sector_colors` for `sca-project` / `sca-structure`.**
  Those CLIs don't render sectors directly; they emit the data that
  `sca-pymol` / `sca-plots` consume. The user's three picks (sca-core,
  sca-pymol, sca-plots) cover every render path.
- **`docs/.claude_sessions/codebase_overview.md` still describes
  `SECTOR_COLORS` as "color palette for sector visualization."** Still
  technically accurate (the constant is now the *default* palette).
  That file is a frozen overview snapshot, not a maintained doc, so I
  didn't touch it.

## State before this session

```bash
git checkout 32b438b   # "Per-input-sequence retention statistics"
```

(addison-dev tip immediately before this session's edits.)
