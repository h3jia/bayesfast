import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial
from collections import namedtuple, OrderedDict
# from abc import ABC, abstractmethod
from ..utils.parallel import get_pool

__all__ = ['Density', 'ModuleCache']

ModuleCache = namedtuple('ModuleCache', ('args', 'kwargs', 'returns'))

# dill does not support ABC at the moment, https://github.com/uqfoundation/dill/pull/450
# once that PR is merged, we can add ABC back
# class Density(ABC):
class Density:

    def __init__(self, module_dict, surrogate_dict, use_jit=True):
        self._module_dict = OrderedDict(module_dict)
        self._surrogate_dict = OrderedDict(surrogate_dict)
        self._use_jit = bool(use_jit)
        # self.apply_jit()

    @property
    def module_dict(self):
        """
        A dict containing all the true models.
        """
        return self._module_dict

    @property
    def surrogate_dict(self):
        """
        A dict containing all the surrogate models.
        """
        return self._surrogate_dict

    @property
    def use_jit(self):
        """
        Whether to jit logq with jax.
        """
        return self._use_jit

    # @abstractmethod
    def forward(self, x):
        """
        Logarithm of the probability density function, and the cache dicts.
        """
        raise NotImplementedError
        # pass

    def logp_and_cache(self, x):
        x = np.atleast_1d(x)
        self._use_surrogate = False
        if x.ndim == 1:
            return self.forward(x)
        elif x.ndim == 2:
            with get_pool() as pool:
                foo = pool.map(self.forward, x)
            return np.asarray([_[0] for _ in foo]), [_[1] for _ in foo]
        else:
            raise NotImplementedError(f'x should be 1-dim or 2-dim, instead of {x.ndim}-dim.')

    def logp(self, x):
        """
        Logarithm of the probability density function.
        """
        return self.logp_and_cache(x)[0]

    __call__ = logp

    def cache(self, x):
        """
        Get the cache dict containing the true model evaluations.
        """
        return self.logp_and_cache(x)[1]

    def _logq(self, x):
        return self.forward(x)[0]

    def logq(self, x):
        """
        Logarithm of the probability density function using surrogate models.
        """
        x = np.atleast_1d(x)
        self._use_surrogate = True
        if x.ndim == 1:
            return self._logq(x)
        elif x.ndim == 2:
            with get_pool() as pool:
                # I don't really understand why I need a lambda here
                # But it seems that w/o it, I will get some maximum recursion depth error
                # Only if I use multiprocess pools; the serial map is fine
                return np.asarray(pool.map(lambda _: self._logq(_), x))
        else:
            raise NotImplementedError(f'x should be 1-dim or 2-dim, instead of {x.ndim}-dim.')

    def apply_jit(self):
        """
        Apply jit with jax.
        """
        # TODO: option for next level
        # self._use_surrogate = True
        self._logq = jit(self._logq)
        # self.forward = jit(self.forward)

    def module_hook(self, i, cache_dict):
        """
        Utility to switch between true and surrogate models.
        """
        # TODO: add gradient fitting
        if self._use_surrogate:
            return self.surrogate_dict[i]
        else:
            def module_wrapped(*args, **kwargs):
                foo = self.module_dict[i](*args, **kwargs)
                cache_dict[i] = ModuleCache(args, kwargs, foo)
                return foo
            return module_wrapped

    def fit(self, cache_dicts, add_keys=()):
        """
        Fit all the surrogate models.
        """
        add_dicts = [OrderedDict({ak: cd[ak] for ak in add_keys}) for cd in cache_dicts]
        for sk, s in self.surrogate_dict.items():
            c = [cd[sk] for cd in cache_dicts]
            s.fit(c, add_dicts)
        if self.use_jit:
            self.apply_jit()
