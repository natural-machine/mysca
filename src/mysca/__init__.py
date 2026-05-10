import logging

__version__ = "0.1.3"

logging.getLogger("mysca").addHandler(logging.NullHandler())

from mysca.results import PreprocessingResults, SCAResults
