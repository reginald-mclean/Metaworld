"""Sets up the project."""

import numpy as np
from Cython.Build import cythonize  # type: ignore
from Cython.Compiler import Options  # type: ignore
from setuptools import Extension, setup

Options.docstrings = True
Options.annotate = False

extensions = [
    Extension(
        "metaworld.utils.reward_utils",
        ["metaworld/utils/reward_utils.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="metaworld",
    ext_modules=cythonize(
        extensions, compiler_directives=({"language_level": 3, "profile": False})
    ),
)
