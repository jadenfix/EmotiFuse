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
def __getattr__(name):
    # Audio utilities do not require loading the model or its encoder stack.
    if name == "EmotiFuse":
        from .model import EmotiFuse
        return EmotiFuse
    if name == "infer":
        from .infer import infer
        return infer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "EmotiFuse",
    "infer",
] 