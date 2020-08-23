import numpy as np
HAS_EMCEE = True
try:
    import emcee
except Exception:
    HAS_EMCEE = False

__all__ = ['EnsembleSampler']

# TODO: implement the wrapper of emcee sampler


class EnsembleSampler:
    def __init__(*args, **kwargs):
        raise NotImplementedError
