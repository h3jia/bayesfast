from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
from extension_helpers import add_openmp_flags_if_available
import warnings


ext_modules = [
    Extension(
        "bayesfast.transforms._constraint",
        ["bayesfast/transforms/_constraint.pyx"],
        include_dirs=[np.get_include()],
        libraries=["m"],
    ),
    Extension(
        "bayesfast.utils._cubic",
        ["bayesfast/utils/_cubic.pyx"],
        include_dirs=[np.get_include()],
        # libraries=["m"],
    ),
    Extension(
        "bayesfast.utils._sobol",
        ["bayesfast/utils/_sobol.pyx"],
        include_dirs=[np.get_include()],
        libraries=["m"],
    ),
    Extension(
        "bayesfast.modules._poly",
        ["bayesfast/modules/_poly.pyx"],
        # include_dirs=[np.get_include()],
        # libraries=["m"],
    )
]


openmp_added = [add_openmp_flags_if_available(_) for _ in ext_modules]
if not all(openmp_added):
    warnings.warn('OpenMP check failed. Compiling without it for now.',
                  RuntimeWarning)


setup(
    name='bayesfast',
    version='0.1.0.dev2',
    author='He Jia and Uros Seljak',
    maintainer='He Jia',
    maintainer_email='he.jia.phy@gmail.com',
    description=('Next generation Bayesian analysis tools for efficient '
                 'posterior sampling and evidence estimation.'),
    url='https://github.com/HerculesJack/bayesfast',
    license='Apache License, Version 2.0',
    python_requires=">=3.6",
    install_requires=['cython', 'extension-helpers', 'numdifftools',
                      'multiprocess', 'matplotlib', 'numpy>=1.17',
                      'scikit-learn', 'scipy', 'threadpoolctl'],
    packages=find_packages(),
    ext_modules=cythonize(ext_modules, language_level="3"),
)
