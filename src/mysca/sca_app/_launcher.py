"""CLI launcher: execs ``streamlit run`` on the page entrypoint.

Recognises ``--config <path>`` (settings JSON) and ``--bundle <path>``
(default bundle path); both pass through to the Streamlit child via
environment variables.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from mysca.sca_app.config import SETTINGS_ENV_VAR

BUNDLE_ENV_VAR = "MYSCA_BUNDLE_PATH"
ENTRYPOINT = Path(__file__).with_name("home.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="mysca Streamlit app")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to app settings JSON (overrides bundled defaults)",
    )
    parser.add_argument(
        "--bundle", type=str, default=None,
        help="Path to an SCA results bundle directory (pre-fills the page input)",
    )
    args, extra = parser.parse_known_args()
    if args.config:
        os.environ[SETTINGS_ENV_VAR] = str(Path(args.config).resolve())
    if args.bundle:
        os.environ[BUNDLE_ENV_VAR] = str(Path(args.bundle).resolve())
    try:
        import streamlit  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "sca-app requires streamlit. Install with "
            "`pip install 'mysca[app]'` or `pip install streamlit`."
        ) from e
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ENTRYPOINT)] + extra,
        check=True,
    )
