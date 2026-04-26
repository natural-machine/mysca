import logging

__version__ = "0.1.1"

logging.getLogger("mysca").addHandler(logging.NullHandler())

from mysca.results import PreprocessingResults, SCAResults
