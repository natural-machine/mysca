# Plan — `sca-pymol` rewrite to consume `sca-structure` output

Planning doc drafted 2026-04-23, to be executed in a future session.
Not yet started.

## Motivation

The current [run_pymol.py](../../src/mysca/run_pymol.py) consumes
`statsectors_seq.npz` (an sca-core artifact) and treats the raw
residue index directly as a PDB residue number via an implicit
``residue_number = 1 + raw_index`` fudge
([run_pymol.py:222](../../src/mysca/run_pymol.py#L222),
[run_pymol.py:356](../../src/mysca/run_pymol.py#L356)). This works
only when the PDB chain is contiguously numbered starting at 1, which
is an unreliable assumption for PDBs pulled from RCSB.

`sca-structure` now produces authoritative per-structure JSON with
``ic_pdb_residues[ic] = [pdb_resi, ...]`` — PDB residue numbers
derived from ``PDBStructure.residue_ids`` rather than inferred. The
rewrite replaces the raw-index path with a direct read of
``sca-structure`` output.

## Scope split (land in order)

Commit 1: input-path rewrite only. Keeps all PyMOL rendering logic
(styles, views, animate, multisector, features, reference align)
unchanged. Changes what the CLI reads and how residues are resolved.

Commit 2 (optional follow-up): refactor rendering (the 180 lines of
duplicated logic in `plot_scaffold_by_sectors` and
`plot_scaffold_with_multiple_sectors`).

Only commit 1 is planned here.

## Public CLI (commit 1)

```bash
sca-pymol --structure <structure_out_dir> [--structure-id ID] \
    -o <outdir> [rendering options...]
```

- `--structure DIR` — directory produced by `sca-structure`. The CLI
  reads `structure_projection.json` to discover per-structure IC
  residue lists. When multiple structures are present (`--seq_map`
  mode), require `--structure-id` to pick one, or default to all and
  emit one render per structure.
- `--structure-id ID` — subset selector within a batch; optional when
  the json contains a single entry.
- **Deprecated / removed**:
  - `-s/--scaffold`, `--pdb_dir`, `--modes` — all replaced by the
    `structure_projection.json` + referenced PDB path combination.

Everything else stays (`--groups`, `--multisector`, `-r/--reference`,
`--features`, `--views`, `--animate`, `--nframes`, `--duration`,
`--show_molybdenum`, `-o`, `-v`).

Open question: **should the legacy flags be kept as a deprecated
fallback for one release**, or removed immediately? I'd recommend
removing immediately — the current tool has no automated tests, so
there's no known external consumer depending on it, and the
``1 + raw_index`` bug means its output was already suspect for any
PDB not using 1-indexed contiguous residues.

## Input contract

`structure_projection.json` from `sca-structure` is a list of dicts.
Each entry has shape:

```json
{
    "structure_id": "...",
    "chain_id": "A",
    "sequence_projection": { "seq_id": "...", ... },
    "ic_pdb_residues": [[10, 12, 15], [...], ...]
}
```

The CLI also needs:
1. **The PDB file path.** Not currently in
   `structure_projection.json`. Either:
   - (a) Extend `sca-structure` to record `pdb_path` in the per-
     structure dict. Small addition to `PdbProjection.to_dict()`.
   - (b) Require the CLI caller to pass `--pdb_dir` separately
     (regression on the "one input" goal).

   Prefer **(a)**. Requires a one-line change to `PdbProjection` +
   one-line change to `projection.to_dict()`.

2. **Scores** for transparency scaling. `ic_loadings` in
   `sequence_projection.ic_loadings` is the successor of
   `sector_{i}_scores_{scaffold}`.

## Rendering mapping (no logic change)

Replace every occurrence of:

```python
# old
groupkey = f"sector_{gidx}_pdbpos_{scaffold}"
group = mode_data[groupkey]           # raw indices
res_idxs = 1 + group                   # implicit offset
```

with:

```python
# new
pdb_resids = projection["ic_pdb_residues"][gidx]
res_idxs = pdb_resids                  # authoritative PDB numbers
```

and for scores:

```python
# old
scorekey = f"sector_{gidx}_scores_{scaffold}"
scores = mode_data[scorekey]

# new
scores = projection["sequence_projection"]["ic_loadings"][gidx]
```

Every other line of the existing `plot_scaffold_by_sectors` and
`plot_scaffold_with_multiple_sectors` can stay as-is.

## Tests

`tests/test_entrypoint_pymol.py` (new):

- **Fixture**: the existing `prep_and_sca_dirs` fixture (via
  `tests/test_structure.py`) + a synthetic PDB built with
  `_write_minimal_pdb` whose sequence matches a retained training
  sequence. Run `sca-structure -s <pdb>` to produce a
  `structure_projection.json`.
- **Assertion**: drive `sca-pymol --structure <out> -o <plots>` and
  assert the expected PNGs land under `<plots>/`. We do NOT compare
  image pixels; we just confirm the CLI completes and emits files.
- **PyMOL skip**: gate the test on `pymol` being importable (like
  `needs_mafft` / `needs_hmmer`). The default test env does not have
  `pymol-open-source` installed, so the test skips cleanly.

No unit tests for the PyMOL rendering functions themselves —
integration-only.

## Docstring + docs updates

- Update [docs/cli_reference.md](../../docs/cli_reference.md)
  `## sca-pymol` section: new required `--structure` flag, dropped
  flags, new Output list, new example.
- Update the top docstring of `run_pymol.py`.
- Keep the `## sca-pymol` requirement on `pymol-open-source`.

## Commit plan

Single commit:

- `src/mysca/run_pymol.py`: new `parse_args` + `main`;
  `_load_structure_projection()` helper; rendering functions updated
  to take a `projection` dict instead of `scaffold + mode_data`.
- `src/mysca/structure/projection.py`: add `pdb_path` to
  `PdbProjection` (forward it from the CLI so the downstream render
  step can find the file).
- `src/mysca/run_structure.py`: write `pdb_path` into
  `PdbProjection.to_dict()` output.
- `tests/test_entrypoint_pymol.py`: new.
- `docs/cli_reference.md`: update `## sca-pymol` section.

## Post-rewrite follow-ups (out of scope)

- Refactor `plot_scaffold_by_sectors` / `plot_scaffold_with_multiple_sectors`
  (significant duplication; roughly 180 lines).
- Add a per-structure animation that auto-chooses the "most
  important" IC based on `ic_loadings` magnitude.
- Support `--features` sourced from the structure-projection output
  rather than a separate JSON.
