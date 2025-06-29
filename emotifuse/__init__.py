"""Python packaging namespace for the compiled `emotifuse` extension.

After installation, `import emotifuse` will load the native extension built via CMake/pybind11.
"""

from importlib import import_module as _imp

# Lazily import the compiled module so that missing binary raises an informative error.
try:
    _imp("emotifuse")  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "The EmotiFuse native extension could not be imported. "
        "Ensure that the package was built correctly and that your environment "
        "meets all build prerequisites (pybind11, onnxruntime)."
    ) from exc 