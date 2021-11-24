import os
import re
import sys
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# The following two classes are adaptations of the Python example for pybind11:
# https://github.com/pybind/python_example/blob/master/setup.py


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir="."):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        path = self.get_ext_fullpath(ext.name)
        extdir = os.path.abspath(os.path.dirname(path))

        # Required for auto-detection of auxiliary "native" libs.
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        os.makedirs(self.build_temp, exist_ok=True)

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        cmake_args = [
            f"-DBUILD_PYTHON=ON",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        cmake_cmd = ["cmake", ext.sourcedir] + cmake_args
        build_cmd = ["cmake", "--build", "."]

        subprocess.check_call(cmake_cmd, cwd=self.build_temp)
        subprocess.check_call(build_cmd, cwd=self.build_temp)


# Hack to get the Jet version number without including the Python package.
with open("include/jet/Version.hpp", "r") as f:
    contents = f.read()
    major = re.search(r"MAJOR_VERSION\s*=\s*(\d+)", contents).group(1)
    minor = re.search(r"MINOR_VERSION\s*=\s*(\d+)", contents).group(1)
    patch = re.search(r"PATCH_VERSION\s*=\s*(\d+)", contents).group(1)
    version = f"{major}.{minor}.{patch}"

requirements = [
    # Necessary until https://github.com/numba/numba/issues/7176 is resolved.
    "numpy<1.21.0",
    "quantum-xir",
    "strawberryfields>=0.18.0",
    "thewalrus>=0.15.0",
]

info = {
    "cmdclass": {"build_ext": CMakeBuild},
    "description": (
        "Jet is an open-source library for quantum circuit simulation "
        "using tensor network contractions"
    ),
    "ext_modules": [CMakeExtension(name="jet.bindings")],
    "include_package_data": True,
    "install_requires": requirements,
    "license": "Apache License 2.0",
    "long_description": open("README.rst", encoding="utf-8").read(),
    "long_description_content_type": "text/x-rst",
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "name": "quantum-jet",
    "package_data": {"xir": ["ir.lark"]},
    "package_dir": {"": "python"},
    "packages": find_packages(where="python", exclude=["src", "tests"]),
    "provides": ["jet", "xir"],
    "url": "https://github.com/XanaduAI/jet",
    "version": version,
    "zip_safe": False,
}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
