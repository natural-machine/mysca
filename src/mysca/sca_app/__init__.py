"""mysca Streamlit app package.

Importing this package is side-effect-free; Streamlit setup happens only
when ``home.py`` is loaded as the Streamlit entrypoint. The :func:`main`
launcher exec's ``streamlit run`` on ``home.py``.
"""

from mysca.sca_app._launcher import main  # noqa: F401
