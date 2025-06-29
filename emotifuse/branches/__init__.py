"""Model branches encapsulating different modalities/features."""

from .wav2vec_branch import Wav2VecBranch  # noqa: F401
from .classical_branch import ClassicalBranch  # noqa: F401
from .spec_transformer_branch import SpecTransformerBranch  # noqa: F401

__all__ = [
    "Wav2VecBranch",
    "ClassicalBranch",
    "SpecTransformerBranch",
] 