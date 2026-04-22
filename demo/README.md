# mysca demos

This directory contains a worked SH3-family example that exercises the full
mysca pipeline from two different starting points. Running
`./run_demo_SH3.sh` executes both paths; each writes into its own output
subdirectory under `SH3/out/`:

- `SH3/out/from_msa/` — pipeline starting from a pre-aligned MSA.
- `SH3/out/from_raw/` — pipeline starting from raw (unaligned) sequences.

## Path A — starting from an aligned MSA

Input: `SH3/data/msas/SH3_demo_MSA_1.afa`, a pre-aligned SH3 MSA in FASTA
format. The scripts run `sca-preprocess` followed by `sca-core`, writing
everything under `SH3/out/from_msa/`.

## Path B — starting from raw sequences

Input: `SH3/data/seqs/PF00018_raw.fasta`, a set of 55 unaligned SH3 homolog
sequences. This file was produced from the Pfam PF00018 (SH3_1) seed
alignment as follows: the seed alignment was retrieved from Pfam in
Stockholm format and saved at `SH3/data/msas/PF00018.alignment.seed`. Gap
characters (both `-` and `.`) were then removed from every sequence so that
each record reflects only the residues actually present in the original
protein, and the resulting records were rewritten in FASTA format with the
Pfam sequence accessions preserved as record IDs. The checked-in
`PF00018_raw.fasta` is the output of that conversion; the Stockholm file
remains untouched alongside it for reference.

From that raw FASTA the demo runs `sca-prealign` (which wraps MAFFT in its
default configuration to produce an alignment), then the standard
`sca-preprocess` and `sca-core` stages, writing everything under
`SH3/out/from_raw/`.

## Running

From this `demo/` directory:

```bash
./run_demo_SH3.sh
```

Individual stages can also be invoked directly from `SH3/scripts/` for
iteration. The `SH3/out/` tree is gitignored, so re-running the demo is
safe and leaves no tracked artifacts.

## Prerequisites

Path B (raw sequences) invokes `sca-prealign`, which in turn requires
`mafft` to be available on `PATH`. Install it however you prefer — for
example `conda install -c bioconda mafft`.
