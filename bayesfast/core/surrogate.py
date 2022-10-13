import numpy as np
import jax.numpy as jnp
from jax import jit
from abc import ABC, abstractmethod

__all__ = ['Surrogate']


class Surrogate(ABC):

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Evaluate the surrogate model.
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def fit(self, cache_dicts, add_dicts=()):
        """
        Fit the surrogate model.
        """
        pass

    def apply_jit(self):
        """
        Apply jit with jax.
        """
        self.forward = jit(self.forward)

    @property
    @abstractmethod
    def n_param(self):
        """
        The number of free parameters. Required to determine the number of fitting samples.
        """
        pass
