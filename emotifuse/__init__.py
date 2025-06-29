"""Top-level package for EmotiFuse.

Exposes the main :class:`~emotifuse.model.EmotiFuse` model and convenience
functions for training and inference.
"""

from importlib import metadata as _metadata

try:
    __version__: str = _metadata.version(__name__)
except _metadata.PackageNotFoundError:  # pragma: no cover
    # When running from source without installation
    __version__ = "0.0.0.dev0"

del _metadata

# Public API ------------------------------------------------------------------
from .model import EmotiFuse  # noqa: E402 F401  (import after __version__)
from .infer import infer  # noqa: E402 F401

__all__ = [
    "__version__",
    "EmotiFuse",
    "infer",
] 