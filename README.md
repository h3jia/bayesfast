# BayesFast

BayesFast is a Python package for efficient Bayesian analysis
developed by [He Jia](http://hejia.io) and
[Uros Seljak](https://physics.berkeley.edu/people/faculty/uros-seljak),
which can be orders of magnitude faster than traditional methods,
on both posterior sampling and evidence estimation.

For cosmologists, we have an add-on package
[CosmoFast](https://github.com/HerculesJack/cosmofast),
which provides several frequently-used cosmological modules.

Both packages are in live development, so the API may be changed at any time.
**Note that some parts of the code are still experimental.**
If you find a bug or have useful suggestions, please feel free to 
open an issue / pull request, or email [He Jia](mailto:he.jia.phy@gmail.com).
We also have a roadmap for features to implement in the future.
Your contributions would be greatly appreciated!

## Installation

We plan to add pypi and conda-forge support later.
For now, please install BayesFast from source with:

```
git clone https://github.com/HerculesJack/bayesfast
cd bayesfast
pip install -e . # please use editable mode for now
```

## Dependencies

BayesFast requires python>=3.6, cython, extension-helpers, numdifftools,
multiprocess, matplotlib, numpy>=1.17, scikit-learn, scipy and threadpoolctl.
Currently, it has been tested on Ubuntu and Mac, with python 3.6-3.8.

## License

BayesFast is distributed under the Apache License, Version 2.0.

## Citing BayesFast

If you find BayesFast useful for your research,
please consider citing our papers accordingly:

* He Jia and Uros Seljak,
*BayesFast: A Fast and Scalable Method for Cosmological Bayesian Inference*,
in prep (for posterior sampling)
* He Jia and Uros Seljak,
*Normalizing Constant Estimation with Gaussianized Bridge Sampling*,
[AABI 2019 Proceedings, PMLR 118:1-14](http://proceedings.mlr.press/v118/jia20a.html)
(for evidence estimation)
