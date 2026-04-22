"""Guards against silent failures in pl.plotting helpers.

plot_data_2d and plot_data_3d skip when the requested axis index exceeds
the data's trailing dimension (e.g. asking for axis 6 on a 3-column IC
matrix). They now emit a debug log line instead of returning silently, so
a user who sees fewer plots than expected can find the reason in the log.
"""

import logging
import os

import numpy as np
import pytest

from mysca.pl.plotting import plot_data_2d, plot_data_3d


@pytest.fixture
def empty_imgdir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def capture_plotting_debug():
    """Capture DEBUG records from mysca.pl.plotting via a local handler.

    Avoids pytest's caplog because the mysca package logger is configured
    with ``propagate=False`` in earlier tests (via configure_logging), so
    records never reach the root logger caplog attaches to.
    """
    target = logging.getLogger("mysca.pl.plotting")
    records = []

    class _Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collector(level=logging.DEBUG)
    old_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        target.removeHandler(handler)
        target.setLevel(old_level)


def test_plot_data_2d_out_of_bounds_logs_and_skips(
    empty_imgdir, capture_plotting_debug,
):
    data = np.random.default_rng(0).standard_normal((10, 3))  # only 3 columns
    groups = [np.array([0, 1]), np.array([2, 3])]
    plot_data_2d("ic", (4, 5), "all", groups, data, empty_imgdir)
    assert os.listdir(empty_imgdir) == []
    assert any(
        "plot_data_2d: skipping" in r.getMessage()
        for r in capture_plotting_debug
    )


def test_plot_data_3d_out_of_bounds_logs_and_skips(
    empty_imgdir, capture_plotting_debug,
):
    data = np.random.default_rng(0).standard_normal((10, 3))
    groups = [np.array([0, 1]), np.array([2, 3])]
    plot_data_3d("ev", (0, 1, 5), "all", groups, data, empty_imgdir)
    assert os.listdir(empty_imgdir) == []
    assert any(
        "plot_data_3d: skipping" in r.getMessage()
        for r in capture_plotting_debug
    )


def test_plot_data_2d_in_bounds_emits_file(empty_imgdir):
    """Sanity: when axes are in range, a plot file is produced."""
    data = np.random.default_rng(0).standard_normal((10, 4))
    groups = [np.array([0, 1]), np.array([2, 3])]
    plot_data_2d("ic", (0, 1), "all", groups, data, empty_imgdir)
    files = os.listdir(empty_imgdir)
    assert any(f.endswith(".png") for f in files), files


def test_plot_data_2d_returns_axes_and_skips_save(empty_imgdir):
    """With save=False, the axes are returned and no file is written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.random.default_rng(0).standard_normal((10, 4))
    groups = [np.array([0, 1])]
    ax = plot_data_2d(
        "ic", (0, 1), "all", groups, data, empty_imgdir, save=False,
    )
    assert ax is not None, "plot_data_2d should return the Axes"
    assert os.listdir(empty_imgdir) == [], "save=False should not write"
    assert ax.get_figure() is not None
    plt.close(ax.get_figure())


def test_plot_data_2d_custom_filename(empty_imgdir):
    """An explicit filename overrides the default data-derived name."""
    data = np.random.default_rng(0).standard_normal((10, 4))
    groups = [np.array([0, 1])]
    plot_data_2d(
        "ic", (0, 1), "all", groups, data, empty_imgdir,
        filename="custom_name.png",
    )
    assert os.listdir(empty_imgdir) == ["custom_name.png"]


def test_plot_data_3d_returns_axes_and_skips_save(empty_imgdir):
    data = np.random.default_rng(0).standard_normal((10, 4))
    groups = [np.array([0, 1])]
    ax = plot_data_3d(
        "ev", (0, 1, 2), "all", groups, data, empty_imgdir, save=False,
    )
    assert ax is not None
    assert os.listdir(empty_imgdir) == []
    import matplotlib.pyplot as plt
    plt.close(ax.get_figure())


def test_plot_t_distributions_max_plots_caps_subplots(empty_imgdir):
    """max_plots=N produces a figure with exactly N subplots."""
    from mysca.pl import plot_t_distributions
    rng = np.random.default_rng(0)
    v = rng.standard_normal((50, 5))
    t_dists_info = [
        {"df": 10.0, "loc": 0.0, "scale": 1.0, "cutoff": 2.0}
        for _ in range(5)
    ]
    fig = plot_t_distributions(
        v, t_dists_info, empty_imgdir, max_plots=2, save=False,
    )
    # With 2 subplots, fig.axes has length 2.
    assert len(fig.axes) == 2
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_t_distributions_max_plots_none_is_all(empty_imgdir):
    from mysca.pl import plot_t_distributions
    v = np.random.default_rng(0).standard_normal((50, 3))
    t_dists_info = [
        {"df": 10.0, "loc": 0.0, "scale": 1.0, "cutoff": 2.0}
        for _ in range(3)
    ]
    fig = plot_t_distributions(v, t_dists_info, empty_imgdir, save=False)
    assert len(fig.axes) == 3
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_t_distributions_max_plots_zero_returns_none(empty_imgdir):
    from mysca.pl import plot_t_distributions
    v = np.random.default_rng(0).standard_normal((50, 3))
    t_dists_info = [
        {"df": 10.0, "loc": 0.0, "scale": 1.0, "cutoff": 2.0}
        for _ in range(3)
    ]
    result = plot_t_distributions(
        v, t_dists_info, empty_imgdir, max_plots=0, save=False,
    )
    assert result is None
