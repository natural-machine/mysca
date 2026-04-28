"""Shared accelerator-selection utilities for mysca CLIs.

`sca-core` and `sca-preprocess` both expose a `--accelerator {none, gpu}`
flag and a per-step method override (`--freq_method`, `--weight_method`).
The resolution rule is the same in both places:

    1. Explicit method (e.g. `--freq_method=numpy`) wins.
    2. Else if a deprecated alias is set (e.g. `--use_jax`), route via it
       with a DeprecationWarning.
    3. Else if `--accelerator=gpu`, route to the gpu variant of that
       step's kernel.
    4. Else fall back to the step's CPU default (e.g. ``numpy``,
       ``sparse``).

Lazy `import torch` lives here too so torch-less envs stay
import-safe.
"""

import logging
import warnings

logger = logging.getLogger("mysca._acceleration")


ACCELERATOR_CHOICES = ("none", "gpu")

PRECISION_CHOICES = ("fp64", "fp32", "fp16")
DEFAULT_PRECISION = "fp64"


def resolve_torch_dtype(precision):
    """Map a precision string to a torch floating-point dtype.

    fp16 is offered for raw throughput on accelerator hardware. It is
    numerically risky for the SCA covariance/eigendecomposition path
    (small eigenvalues can be lost to ~10⁻³ relative precision), and
    callers should treat fp16 results as a preview.

    Args:
        precision: one of PRECISION_CHOICES.

    Returns:
        A torch.dtype.

    Raises:
        ValueError: if `precision` is not a recognized choice.
    """
    import torch
    table = {
        "fp64": torch.float64,
        "fp32": torch.float32,
        "fp16": torch.float16,
    }
    try:
        return table[precision]
    except KeyError:
        raise ValueError(
            f"Unknown precision {precision!r}. Choices: {PRECISION_CHOICES}"
        )


def resolve_numpy_dtype(precision):
    """Map a precision string to a numpy floating-point dtype."""
    import numpy as np
    table = {
        "fp64": np.float64,
        "fp32": np.float32,
        "fp16": np.float16,
    }
    try:
        return table[precision]
    except KeyError:
        raise ValueError(
            f"Unknown precision {precision!r}. Choices: {PRECISION_CHOICES}"
        )


def resolve_method(
        *,
        method,
        accelerator,
        cpu_default,
        gpu_choice="gpu",
        deprecated_alias=False,
        deprecated_alias_name=None,
        deprecated_alias_target=None,
):
    """Resolve the final kernel method given user-facing flags.

    Args:
        method: explicit `--<step>_method` value, or None when the user
            didn't pass it.
        accelerator: `--accelerator` value (one of ACCELERATOR_CHOICES).
        cpu_default: kernel string when no accelerator and no explicit
            method are set (e.g. `"numpy"`, `"sparse"`).
        gpu_choice: kernel string when `accelerator="gpu"` is set.
        deprecated_alias: True if a deprecated alias flag was passed
            (e.g. `--use_jax`). Triggers a DeprecationWarning.
        deprecated_alias_name: human-readable name of the deprecated
            alias (e.g. `"use_jax"`). Used in the warning message.
        deprecated_alias_target: kernel string to route to when the
            alias is set (e.g. `"jax"`). Required if
            `deprecated_alias=True`.

    Returns:
        The resolved kernel string.
    """
    if method is not None:
        if deprecated_alias:
            warnings.warn(
                f"Both `{deprecated_alias_name}` and an explicit method "
                f"were set; `{deprecated_alias_name}` is deprecated and "
                "is being ignored.",
                DeprecationWarning, stacklevel=2,
            )
        return method
    if deprecated_alias:
        if deprecated_alias_target is None:
            raise ValueError(
                "deprecated_alias=True requires deprecated_alias_target."
            )
        warnings.warn(
            f"`{deprecated_alias_name}` is deprecated; pass "
            f"the equivalent --<step>_method instead. Routing to "
            f"{deprecated_alias_target!r} for now.",
            DeprecationWarning, stacklevel=2,
        )
        return deprecated_alias_target
    if accelerator == "gpu":
        return gpu_choice
    return cpu_default


def detect_device():
    """Return the first available torch device or torch.device('cpu').

    Lazy `import torch`: callers that never set `--accelerator gpu`
    won't import torch at all, keeping torch-less envs import-safe.
    """
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")
