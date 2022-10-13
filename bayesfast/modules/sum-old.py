from ..core.module import ModuleBase
import numpy as np

__all__ = ['Sum']


class Sum(ModuleBase):
    """
    Computing the sum of input vars.
    
    Parameters
    ----------
    input_vars : str or 1-d array_like of str
        Name(s) of input variable(s). Will first be concatenated as one single
        variable.
    output_vars : str or 1-d array_like of str
        Name of output variable. Should contain only 1 variable here.
    delete_vars : str or 1-d array_like of str, optional
        Name(s) of variable(s) to be deleted from the dict during runtime. Set
        to ``()`` by default.
    b : 1-d array_like of float, or None, optional
        If not None, should match the shape of (concatenated) ``input_vars``,
        and then the summation of ``b * x_input`` will be computed. Set to
        ``None`` by default.
    label : str or None, optional
        The label of the module used in ``print_summary``. Set to ``None`` by
        default.
    """
    def __init__(self, input_vars, output_vars, delete_vars=(), b=None,
                 label=None):
        super().__init__(
            input_vars=input_vars, output_vars=output_vars,
            delete_vars=delete_vars, input_shapes=-1, output_shapes=None,
            input_scales=None, label=label)
        self.b = b

    _input_min_length = 1

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
                b = np.atleast_1d(b)
                assert b.ndim == 1
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

    def _jac(self, x):
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
