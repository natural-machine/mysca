"""Downstream analysis: plotting and data-export for SCA results.

This module is a scratch-space for post-hoc analysis built on top of
loaded ``SCAResults`` (+ ``PreprocessingResults``) bundles. Add new
plotting helpers and data-export helpers here; split into separate
modules (``plots.py``, ``exports.py``, ...) once it grows.

Conventions, mirroring the rest of ``mysca``:

* Functions take already-loaded results objects (``SCAResults`` /
  ``PreprocessingResults``) — not directories. Loading is the caller's
  responsibility (``SCAResults.load(dirpath)``).
* Data-export helpers write to an explicit output path passed by the
  caller; don't bake in directory layout.
* Plotting helpers take an ``ax`` when single-panel, or an ``imgdir``
  when they write files, and close any figures they create.
"""

import logging
from pathlib import Path

from mysca.results import PreprocessingResults, SCAResults

logger = logging.getLogger("mysca.analysis")
