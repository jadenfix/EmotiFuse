import os
import sys
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
            f"-DPython_EXECUTABLE={sys.executable}",
        ]

        subprocess.check_call(cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--config", "Release"], cwd=build_temp)

        # Copy the resulting extension to the final location expected by setuptools
        for ext in self.extensions:
            dest_path = Path(self.get_ext_fullpath(ext.name))
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Find the built extension file (it may have a different Python version suffix)
            built_files = list(build_temp.glob("emotifuse*.so"))
            if not built_files:
                built_files = list(build_temp.glob("emotifuse*.pyd"))
            
            if built_files:
                built_path = built_files[0]
                self.copy_file(built_path, dest_path)
            else:
                raise RuntimeError(f"Could not find built extension in {build_temp}")

    def build_extensions(self):
        # We override run(), so we do not use build_extensions.
        pass


# Dummy extension just to hook into build_ext
ext_modules = [Extension("emotifuse", sources=[])]


setup(
    name="emotifuse",
    version="0.1.0",
    author="Jaden Fix",
    author_email="jadenfix123@gmail.com",
    description="EmotiFuse: C++-backed speech emotion recognition (scaffold)",
    long_description="See README.md",
    packages=["emotifuse"],
    package_dir={"emotifuse": "emotifuse_pkg"},  # Map emotifuse package to emotifuse_pkg directory
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    install_requires=["onnxruntime>=1.17"],
    zip_safe=False,
    include_package_data=True,
) 