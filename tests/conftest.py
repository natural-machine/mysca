"""Pytest Configuration File

"""

import pytest
import shutil
import warnings

DATDIR = "tests/_data"  # data directory for all tests.
TMPDIR = "tests/_tmp"  # output directory for all tests.

OPTIONAL_TOOLS = (
    ("mafft", "sca-prealign --align mafft / sca-project --aligner mafft_add"),
    ("hmmbuild", "sca-project --aligner hmmalign (hmmer)"),
    ("hmmalign", "sca-project --aligner hmmalign (hmmer)"),
    ("mmseqs", "sca-prealign --cluster mmseqs2"),
    ("clustalo", "sca-prealign --align clustalo"),
)

def remove_dir(dir:str):
    """Helper function to remove a directory recursively."""
    if not dir.startswith(TMPDIR):
        msg = f"Can only use function `remove_dir` from tests.conftest to \
        remove directories in the directory {TMPDIR}. Got: {dir}"
        raise RuntimeError(msg)
    shutil.rmtree(dir)

#####################
##  Configuration  ##
#####################

def pytest_addoption(parser):
    parser.addoption(
        "--benchmark", action="store_true", default=False, 
        help="run benchmarking tests"
    )
    parser.addoption(
        "--use_gpu", action="store_true", default=False, 
        help="run GPU specific tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "benchmark: mark test as benchmarking")
    config.addinivalue_line("markers", "use_gpu: mark test as GPU specific")

    missing = [(name, why) for name, why in OPTIONAL_TOOLS
               if shutil.which(name) is None]
    try:
        import pymol  # noqa: F401
        pymol_missing = False
    except ImportError:
        pymol_missing = True
    if pymol_missing:
        missing.append(("pymol-open-source", "sca-pymol rendering"))

    if missing:
        lines = [
            f"  - {name} (gates: {why})" for name, why in missing
        ]
        warnings.warn(
            "Optional mysca dependencies not found in this environment; "
            "tests that require them will be skipped:\n" + "\n".join(lines),
            UserWarning,
            stacklevel=2,
        )

def pytest_collection_modifyitems(config, items):
    benchmark_flag_given = False
    use_gpu_flag_given = False
    if config.getoption("--benchmark"):
        # --benchmark given in cli: do not skip benchmarking tests
        benchmark_flag_given = True
    if config.getoption("--use_gpu"):
        # --use_gpu given in cli: do not skip GPU tests
        use_gpu_flag_given = True
    skip_benchmark = pytest.mark.skip(reason="need --benchmark option to run")
    skip_use_gpu = pytest.mark.skip(reason="need --use_gpu option to run")
    for item in items:
        if "benchmark" in item.keywords and not benchmark_flag_given:
            item.add_marker(skip_benchmark)
        if "use_gpu" in item.keywords and not use_gpu_flag_given:
            item.add_marker(skip_use_gpu)
    