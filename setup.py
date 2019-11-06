from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "bayesfast.transforms._constraint",
        ["bayesfast/transforms/_constraint.pyx"],
        include_dirs=[np.get_include()],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        libraries=["m"],
    ),
    Extension(
        "bayesfast.utils._cubic",
        ["bayesfast/utils/_cubic.pyx"],
        include_dirs=[np.get_include()],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        # libraries=["m"],
    ),
    Extension(
        "bayesfast.modules._poly",
        ["bayesfast/modules/_poly.pyx"],
        # include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        # libraries=["m"],
    )
]

setup(
    name='bayesfast',
    version='0.1.0dev1',
    ext_modules=cythonize(ext_modules, language_level = "3"),
)
