"""Plotting helpers for the mysca pipeline.

All plotters take already-computed inputs and write PNGs into a given
imgdir. They are intentionally I/O helpers, not analysis — the actual
computation lives in preprocess / core / run_sca. Every function closes
its figure on the way out, so they are safe to call in a loop.
"""

from mysca.pl.plotting import (
    plot_conservation,
    plot_conservation_positional,
    plot_conservation_top,
    plot_covariance_matrix,
    plot_data_2d,
    plot_data_3d,
    plot_dendrogram,
    plot_filter_distributions,
    plot_filter_history,
    plot_prealign_filter_history,
    plot_sca_matrix,
    plot_sca_matrix_sector_subset,
    plot_sca_spectrum,
    plot_sca_spectrum_vs_null,
    plot_sequence_similarity,
    plot_t_distributions,
)

__all__ = [
    "plot_conservation",
    "plot_conservation_positional",
    "plot_conservation_top",
    "plot_covariance_matrix",
    "plot_data_2d",
    "plot_data_3d",
    "plot_dendrogram",
    "plot_filter_distributions",
    "plot_filter_history",
    "plot_prealign_filter_history",
    "plot_sca_matrix",
    "plot_sca_matrix_sector_subset",
    "plot_sca_spectrum",
    "plot_sca_spectrum_vs_null",
    "plot_sequence_similarity",
    "plot_t_distributions",
]
