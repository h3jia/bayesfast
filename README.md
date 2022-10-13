# BayesFast

![python package](https://github.com/h3jia/bayesfast/workflows/python%20package/badge.svg)
[![codecov](https://codecov.io/gh/h3jia/bayesfast/branch/master/graph/badge.svg)](https://codecov.io/gh/h3jia/bayesfast)
![PyPI](https://img.shields.io/pypi/v/bayesfast)
![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/bayesfast)
[![Documentation Status](https://readthedocs.org/projects/bayesfast/badge/?version=latest)](https://bayesfast.readthedocs.io/en/latest/?badge=latest)

BayesFast is a Python package for efficient Bayesian analysis developed by [He Jia](http://hejia.io)
and [Uros Seljak](https://physics.berkeley.edu/people/faculty/uros-seljak),
which can be orders of magnitude faster than traditional methods,
on both posterior sampling and evidence estimation.

For cosmologists, we have an add-on package [CosmoFast](https://github.com/h3jia/cosmofast),
which provides several frequently-used cosmological modules.

Both packages are in live development, so the API may be changed at any time.
**Note that some parts of the code are still experimental.**
If you find a bug or have useful suggestions, please feel free to open an issue / pull request,
or email [He Jia](mailto:he.jia.phy@gmail.com).
We also have a roadmap for features to implement in the future.
Your contributions would be greatly appreciated!

## Links

* Website: https://www.bayesfast.org/
* Documentation: https://bayesfast.readthedocs.io/en/latest/
* Source Code: https://github.com/h3jia/bayesfast
* Bug Reports: https://github.com/h3jia/bayesfast/issues

## What's New

We are upgrading BayesFast & CosmoFast to v0.2 with JAX, which would be faster, more accueate, and
much easier to use than the previous version!

## Installation

We plan to add pypi and conda-forge support later.
For now, please install BayesFast from source with:

```
git clone https://github.com/h3jia/bayesfast
cd bayesfast
pip install -e .
# you can drop the -e option if you don't want to use editable mode
# but note that pytest may not work correctly in this case
```

To check if BayesFast is built correctly, you can do:

```
pytest # for this you will need to have pytest installed
```

## Dependencies

BayesFast requires python>=3.7, cython, extension-helpers, jax>=0.3, matplotlib, multiprocess,
numdifftools, numpy>=1.17, scikit-learn, scipy>=1.0 and threadpoolctl.
Currently, it has been tested on Ubuntu and MacOS, with python 3.7-3.10.

## License

BayesFast is distributed under the Apache License, Version 2.0.

## Citing BayesFast

If you find BayesFast useful for your research, please consider citing our papers accordingly:

* He Jia and Uros Seljak,
*BayesFast: A Fast and Scalable Method for Cosmological Bayesian Inference*, to be submitted
(for posterior sampling)
* He Jia and Uros Seljak,
*Normalizing Constant Estimation with Gaussianized Bridge Sampling*,
[AABI 2019 Proceedings, PMLR 118:1-14](http://proceedings.mlr.press/v118/jia20a.html)
(for evidence estimation)
