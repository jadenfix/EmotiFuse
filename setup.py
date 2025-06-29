import os
import platform
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    """Custom build_ext command that runs CMake."""

    def run(self):
        # Ensure build directory exists
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            "cmake",
            str(Path(__file__).parent.resolve()),
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        # Let users override the Python executable used by pybind11
        cmake_args += [f"-DPYTHON_EXECUTABLE={platform.python_executable}"] if hasattr(platform, "python_executable") else []

        subprocess.check_call(cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--config", "Release"], cwd=build_temp)

        # Copy the resulting extension to the final location expected by setuptools
        for ext in self.extensions:
            dest_path = Path(self.get_ext_fullpath(ext.name))
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            built_path = build_temp / dest_path.name
            self.copy_file(built_path, dest_path)

    def build_extensions(self):
        # We override run(), so we do not use build_extensions.
        pass


# Dummy extension just to hook into build_ext
ext_modules = [Extension("emotifuse", sources=[])]


setup(
    name="emotifuse",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="EmotiFuse: C++-backed speech emotion recognition (scaffold)",
    long_description="See README.md",
    packages=["emotifuse"],  # Empty namespace to ship the compiled module
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    install_requires=["onnxruntime>=1.17"],
    zip_safe=False,
    include_package_data=True,
) 