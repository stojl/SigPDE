try:
    import torch
except ImportError as e:
    raise ImportError(
        "The 'torch' submodule requires PyTorch to be installed. "
        "Install it with 'pip install torch'."
    ) from e

from . import static_kernels as kernels

from .kernel import (
    SigPDE,
    RobustSigPDE
)

__all__ = ["SigPDE", "RobustSigPDE", "kernels"]