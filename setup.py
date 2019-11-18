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
    author='He Jia and Uros Seljak',
    maintainer='He Jia',
    maintainer_email='he.jia.phy@gmail.com',
    description=('Next generation Bayesian analysis tools for efficient '
                 'posterior sampling and evidence estimation.'),
    url='https://github.com/HerculesJack/bayesfast',
    license='Apache License, Version 2.0',
    python_requires=">=3",
    install_requires=['cython', 'dask', 'numdifftools', 'numpy', 'scikit-learn',
                      'scipy'],
    ext_modules=cythonize(ext_modules, language_level="3"),
)
