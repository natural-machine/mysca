import logging

__version__ = "0.0.2"

logging.getLogger("mysca").addHandler(logging.NullHandler())

from mysca.results import PreprocessingResults, SCAResults
