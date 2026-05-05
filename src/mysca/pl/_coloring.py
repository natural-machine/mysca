"""Helpers for resolving per-sequence color values from metadata."""


def resolve_color_values(metadata_df, seq_ids, column):
    """Return a 1-D array of `metadata_df[column]` aligned to `seq_ids`.

    `metadata_df` must carry a `seq_id` column (validated upstream by
    the loader). Sequences in `seq_ids` that are absent from the
    metadata land as NaN (numeric column) or None (object column);
    `plot_seq_projection_2d` renders both as the "NA" bucket.
    """
    md_indexed = metadata_df.set_index("seq_id", drop=False)
    series = md_indexed[column].reindex(seq_ids)
    return series.to_numpy()
