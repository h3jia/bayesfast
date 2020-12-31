from ..core.module import ModuleBase
import numpy as np

__all__ = ['Sum']


class Sum(ModuleBase):
    """
    Computing the sum of input vars.
    """
    def __init__(self, input_vars, output_vars, delete_vars=(), b=None,
                 label=None):
        super().__init__(
            input_vars=input_vars, output_vars=output_vars,
            delete_vars=delete_vars, input_shapes=-1, output_shapes=None,
            input_scales=None, label=label)
        self.b = b

    _input_min_length = 2

    _input_max_length = np.inf

    _output_min_length = 1

    _output_max_length = 1

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        if b is None:
            pass
        else:
            try:
                b = np.array(b)
                assert b.ndim == 1 and b.size == len(self.input_vars)
            except Exception:
                raise ValueError('invalid value for b.')
        self._b = b

    def _fun(self, x):
        if self.b is None:
            return np.sum(x)
        elif isinstance(self.b, np.ndarray):
            return np.sum(self.b * x)
        else:
            raise RuntimeError('unexpected value {} for self.b.'.format(self.b))

    def _jax(self, x):
        if self.b is None:
            return np.ones((1, len(self.input_vars)))
        elif isinstance(self.b, np.ndarray):
            return self.b.copy()[np.newaxis]
        else:
            raise RuntimeError('unexpected value {} for self.b.'.format(self.b))

    def _fun_and_jac(self, x):
        if self.b is None:
            return np.sum(x), np.ones((1, len(self.input_vars)))
        elif isinstance(self.b, np.ndarray):
            return np.sum(self.b * x), self.b.copy()[np.newaxis]
        else:
            raise RuntimeError('unexpected value {} for self.b.'.format(self.b))
