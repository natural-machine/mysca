"""Out-of-sample sequence projection onto an existing SCA result.

Public surface:

    from mysca.project import (
        ALIGNERS, align_to_msa, register_aligner,
        SequenceProjection, ProjectionResult, project_sequences,
    )

Aligners are registered in ``mysca.project.alignment.ALIGNERS``. The
default is ``"mafft_add"`` (``mafft --add --keeplength``); ``hmmalign``
is reserved as a name but not yet implemented.
"""

from mysca.project.alignment import (
    ALIGNERS,
    align_to_msa,
    register_aligner,
)
from mysca.project.projection import (
    ProjectionResult,
    SequenceProjection,
    project_sequences,
)

__all__ = [
    "ALIGNERS",
    "align_to_msa",
    "register_aligner",
    "ProjectionResult",
    "SequenceProjection",
    "project_sequences",
]
