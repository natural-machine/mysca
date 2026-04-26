"""Out-of-sample sequence projection onto an existing SCA result.

Public surface:

    from mysca.project import (
        ALIGNERS, align_to_msa, register_aligner,
        SequenceProjection, ProjectionResult, project_sequences,
    )

Aligners are registered in ``mysca.project.alignment.ALIGNERS``. The
default is ``"mafft_add"`` (``mafft --add --keeplength``); ``"hmmalign"``
builds a profile HMM with every reference column as a match state
(``hmmbuild --hand --amino``) and aligns new sequences via
``hmmalign --outformat afa``, keeping only match columns.
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
