import numpy as np
HAS_EMCEE = True
try:
    import emcee
except:
    HAS_EMCEE = False

__all__ = ['EnsembleSampler']


class EnsembleSampler:
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
