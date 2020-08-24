import numpy as np
from collections import namedtuple, OrderedDict
from ..utils.collections import VariableDict, PropertyList
from ..utils import all_isinstance
from copy import deepcopy
import warnings
from .module import Module, Surrogate
from ..transforms._constraint import *

__all__ = ['Pipeline', 'Density', 'DensityLite']

# TODO: add call counter?
# TODO: review the behavior of out
# TODO: do we need logq information in fit?
# TODO: use jacobian information to fit
# TODO: return -inf when outside the bound
# TODO: implement decay and logp transform for VariableDict


DecayOptions = namedtuple('DecayOptions',
                          ('use_dacay', 'alpha', 'alpha_p','gamma'))


class _PipelineBase:
    """Utilities shared by `Pipeline`, `Density` and `DensityLite`."""
    @property
    def input_scales(self):
        return self._input_scales
    
    @input_scales.setter
    def input_scales(self, scales):
        if scales is None:
            self._input_scales = None
        else:
            self._input_scales = self._scale_check(scales)
            # self._input_scales.flags.writeable = False # TODO: PropertyArray?
    
    @staticmethod
    def _scale_check(scales):
        try:
            scales = np.ascontiguousarray(scales).astype(np.float)
            if scales.ndim == 1:
                scales = np.array((np.zeros_like(scales), scales)).T.copy()
            if not (scales.ndim == 2 and scales.shape[-1] == 2):
                raise ValueError('I do not know how to interpret the shape '
                                 'of input_scales.')
        except Exception:
            raise ValueError('Invalid value for input_scales.')
        return scales
    
    @property
    def hard_bounds(self):
        return self._hard_bounds
    
    @hard_bounds.setter
    def hard_bounds(self, bounds):
        if isinstance(bounds, bool):
            self._hard_bounds = bounds
        else:
            self._hard_bounds = self._bound_check(bounds)
            # self._hard_bounds.flags.writeable = False # TODO: PropertyArray?
    
    @staticmethod
    def _bound_check(bounds):
        try:
            bounds = np.atleast_1d(bounds).astype(bool).astype(np.uint8).copy()
            if bounds.ndim == 1:
                bounds = np.array((bounds, bounds)).T.copy()
            if not (bounds.ndim == 2 and bounds.shape[-1] == 2):
                raise ValueError(
                    'I do not know how to interpret the shape of hard_bounds.')
        except Exception:
            raise ValueError('Invalid value for hard_bounds')
        return bounds
    
    @property
    def copy_input(self):
        return self._copy_input
    
    @copy_input.setter
    def copy_input(self, copy):
        self._copy_input = bool(copy)
    
    @property
    def original_space(self):
        return self._original_space
    
    @original_space.setter
    def original_space(self, os):
        self._original_space = bool(os)
    
    def _constraint(self, x, out, f, f2, k):
        if self._input_scales is None:
            if k == 0:
                _out = x.copy()
            elif k == 1:
                _out = np.ones_like(x)
            elif k == 2:
                _out = np.zeros_like(x)
            else:
                raise RuntimeError('unexpected value for k.')
            if out is False:
                x = _out
                return
            elif out is True:
                return _out
            else:
                if not (isinstance(out, np.ndarray) and x.shape == out.shape):
                    raise ValueError('invalid value for out.')
                out = _out
                return
        else:
            _return = False
            x = np.ascontiguousarray(x)
            if out is False:
                out = x
            elif out is True:
                out = np.empty_like(x)
                _return = True
            else:
                if not (isinstance(out, np.ndarray) and x.shape == out.shape):
                    raise ValueError('invalid value for out.')
                out = np.ascontiguousarray(out)
            if isinstance(self._hard_bounds, bool):
                _hb = self._hard_bounds * np.ones((x.shape[-1], 2), np.uint8)
            else:
                _hb = self._hard_bounds
            if x.ndim == 1:
                f(x, self._input_scales, out, _hb, x.shape[0])
            elif x.ndim == 2:
                f2(x, self._input_scales, out, _hb, x.shape[1], x.shape[0])
            else:
                _shape = x.shape
                x = x.reshape((-1, _shape[-1]))
                out = out.reshape((-1, _shape[-1]))
                f2(x, self._input_scales, out, _hb, x.shape[1], x.shape[0])
                x = x.reshape(_shape)
                out = out.reshape(_shape)
            if _return:
                return out
    
    def from_original(self, x, out=True):
        return self._constraint(x, out, _from_original_f, _from_original_f2, 0)
    
    def from_original_grad(self, x, out=True):
        return self._constraint(x, out, _from_original_j, _from_original_j2, 1)
    
    def from_original_grad2(self, x, out=True):
        return self._constraint(
            x, out, _from_original_jj, _from_original_jj2, 2)
    
    def to_original(self, x, out=True):
        return self._constraint(x, out, _to_original_f, _to_original_f2, 0)
    
    def to_original_grad(self, x, out=True):
        return self._constraint(x, out, _to_original_j, _to_original_j2, 1)
    
    def to_original_grad2(self, x, out=True):
        return self._constraint(x, out, _to_original_jj, _to_original_jj2, 2)
    
    def print_summary(self):
        raise NotImplementedError
    
    def _check_os_us(self, original_space, use_surrogate):
        if original_space is None:
            original_space = self.original_space
        else:
            original_space = bool(original_space)
        if use_surrogate is None:
            use_surrogate = self.use_surrogate
        else:
            use_surrogate = bool(use_surrogate)
        return original_space, use_surrogate


class _DensityBase:
    """Utilities shared by `Density` and `DensityLite`."""
    def _get_diff(self, x=None, x_trans=None):
        # Returning log |dx / dx_trans|.
        if x is not None:
            return -np.sum(np.log(np.abs(self.from_original_grad(x))), axis=-1)
        elif x_trans is not None:
            return np.sum(np.log(np.abs(self.to_original_grad(x_trans))),
                          axis=-1)
        else:
            raise ValueError('x and x_trans cannot both be None.')
    
    def to_original_density(self, density, x_trans=None, x=None):
        diff = self._get_diff(x, x_trans)
        density = np.asarray(density)
        if density.size != diff.size:
            raise ValueError('the shape of density is inconsistent with the '
                             'shape of x_trans or x.')
        return density - diff
    
    def from_original_density(self, density, x=None, x_trans=None):
        diff = self._get_diff(x, x_trans)
        density = np.asarray(density)
        if density.size != diff.size:
            raise ValueError('the shape of density is inconsistent with the '
                             'shape of x or x_trans.')
        return density + diff


class Pipeline(_PipelineBase):
    """
    Constructing composite functions from basic `Module`(s).
    
    Parameters
    ----------
    module_list : Module or 1-d array_like of Module, optional
        List of `Module`(s) constituting the `Pipeline`. Set to `[]` by default.
    surrogate_list : Surrogate or 1-d array_like of Surrogate, optional
        List of surrogate modules. Set to `[]` by default.
    input_vars : str or 1-d array_like of str, optional
        Name(s) of input variable(s). Set to `['__var__']` by default.
    input_dims : 1-d array_like of int, or None, optional
        Used to divide and extract the variable(s) from the input. If 1-d
        array_like, should have the same shape as `input_vars`. If `None`, will
        be interpreted as there is only one input variable. Set to `None` by
        default.
    input_scales : None or array_like of float(s), optional
        Controlling the scaling of input variables. Set to `None` by default.
    hard_bounds : bool or array_like, optional
        Controlling whether `input_scales` should be interpreted as hard bounds,
        or just used to rescale the variables. If bool, will be applied to all
        the variables. If array_like, should have shape of `(input_size,)` or
        `(input_size, 2)`. Set to `False` by default.
    copy_input : bool, optional
        Whether to make a copy of the input before evaluating the Pipeline. Set
        to False by default.
    module_start : int or None, optional
        The index of the `Module` in `module_list` at which to start the
        evaluation. If `None`, will be interpreted as `0`, i.e. the first
        `Module`. Set to `None` by default.
    module_stop : int or None, optional
        The index of the `Module` in `module_list` after which to end the
        evaluation. If `None`, will be interpreted as `n_module - 1`, i.e. the
        last `Module`. Set to `None` by default.
    original_space : bool, optional
        Whether the input variables are in the original, untransformed space.
        Will be overwritten if the `original_space` argument of `fun`, `jac` and
        `fun_and_jac` is not None. Set to `True` by default.
    use_surrogate : bool, optional
        Whether to use surrogate modules during the evaluation. Will be
        overwritten if the `use_surrogate` argument of `fun`, `jac` and
        `fun_and_jac` is not None. Set to `False` by default.
    
    Notes
    -----
    See the tutorial for more information of usage.
    """
    def __init__(self, module_list=[], surrogate_list=[],
                 input_vars=['__var__'], input_dims=None, input_scales=None,
                 hard_bounds=True, copy_input=False, module_start=None,
                 module_stop=None, original_space=True, use_surrogate=False):
        self.module_list = module_list
        self.surrogate_list = surrogate_list
        self.input_vars = input_vars
        self.input_dims = input_dims
        self.input_scales = input_scales
        self.hard_bounds = hard_bounds
        self.copy_input = copy_input
        self.module_start = module_start
        self.module_stop = module_stop
        self.original_space = original_space
        self.use_surrogate = use_surrogate
    
    @property
    def module_list(self):
        return self._module_list
    
    @module_list.setter
    def module_list(self, ml):
        if isinstance(ml, Module):
            ml = [ml]
        if hasattr(ml, '__iter__'):
            self._module_list = PropertyList(ml, self._ml_check)
        else:
            raise ValueError('module_list should be a Module, or consist of '
                             'Module(s).')
    
    @staticmethod
    def _ml_check(ml):
        for i, m in enumerate(ml):
            if not isinstance(m, Module):
                raise ValueError(
                    'element #{} of module_list is not a Module.'.format(i))
        return ml
    
    @property
    def surrogate_list(self):
        return self._surrogate_list
    
    @surrogate_list.setter
    def surrogate_list(self, sl):
        if isinstance(sl, Surrogate):
            sl = [sl]
        if hasattr(sl, '__iter__'):
            self._surrogate_list = PropertyList(sl, self._sl_check)
        else:
            raise ValueError('surrogate_list should be a Surrogate, or consist '
                             'of Surrogate(s).')
    
    def _sl_check(self, sl):
        for i, s in enumerate(sl):
            if not isinstance(s, Surrogate):
                raise ValueError('element #{} of surrogate_list is not a '
                                 'Surrogate'.format(i))
        self._build_surrogate_recipe(sl)
        return sl
    
    def _build_surrogate_recipe(self, sl):
        # ((0, i_step_0, n_step_0), (1, i_step_1, n_step_1), ...)
        ns = len(sl)
        if ns > 0:
            self._surrogate_recipe = np.array(
                [[i, *s._scope] for i, s in enumerate(sl)])
            _recipe_sort = np.argsort(
                self._surrogate_recipe[:, 1] % self.n_module)
            self._surrogate_recipe = (
                self._surrogate_recipe[_recipe_sort].astype(np.int))
            for i in range(ns - 1):
                if (np.sum(self._surrogate_recipe[i, 1:]) >
                    self._surrogate_recipe[i + 1, 1]):
                    raise ValueError('the #{} surrogate model overlaps with '
                                     'the next one.'.format(i))
        else:
            self._surrogate_recipe = np.empty((ns, 3), dtype=np.int)
    
    def _get_start_stop(self):
        if self.module_start is None:
            start = 0
        else:
            start = self.module_start % self.n_module
        if self.module_stop is None:
            stop = self.n_module - 1
        else:
            stop = self.module_stop % self.n_module
        if start > stop:
            raise ValueError('start should be no larger than stop.')
        return start, stop
    
    def _options_check(self, start, stop):
        start = self._step_check(start, 'start')
        stop = self._step_check(stop, 'stop')
        if start > stop:
            raise ValueError('start should be no larger than stop.')
        return start, stop
    
    def _step_check(self, step, tag):
        if step is None:
            if tag == 'start':
                step = 0
            elif tag == 'stop':
                step = self.n_module - 1
            else:
                raise RuntimeError('unexpected value for tag.')
        else:
            try:
                step = int(step)
                step = step % self.n_module
            except Exception:
                raise ValueError('{} should be an int or None, instead '
                                 'of {}.'.format(tag, step))
        return step
    
    _var_check = Module._var_check
    
    @property
    def n_module(self):
        return len(self._module_list)
    
    @property
    def n_surrogate(self):
        return len(self._surrogate_list)
    
    @property
    def has_surrogate(self):
        return self.n_surrogate > 0
    
    @property
    def module_start(self):
        return self._module_start
    
    @module_start.setter
    def module_start(self, start):
        self._module_start = None if (start is None) else int(start)
    
    @property
    def module_stop(self):
        return self._module_stop
    
    @module_stop.setter
    def module_stop(self, stop):
        self._module_stop = None if (stop is None) else int(stop)
    
    @property
    def use_surrogate(self):
        return self._use_surrogate
    
    @use_surrogate.setter
    def use_surrogate(self, us):
        self._use_surrogate = bool(us)
    
    def fun(self, x, original_space=None, use_surrogate=None):
        original_space, use_surrogate = self._check_os_us(original_space,
                                                          use_surrogate)
        # vectorization using recursions
        # TODO: review this
        if isinstance(x, VariableDict):
            var_dict = deepcopy(x) if self.copy_input else x
        else:
            x = np.atleast_1d(x)
            if x.ndim == 1:
                if x.dtype.kind == 'f':
                    if self.copy_input:
                        x = x.copy()
                    if not original_space:
                        x = self.to_original(x)
                    var_dict = VariableDict()
                    if self._input_cum is None:
                        var_dict._fun[self._input_vars[0]] = x
                    else:
                        for i, n in enumerate(self._input_vars):
                            var_dict._fun[n] = x[
                                self._input_cum[i]:self._input_cum[i + 1]]
                elif x.dtype.kind == 'O':
                    return np.asarray([self.fun(_x, original_space,
                                                use_surrogate) for _x in x])
                else:
                    raise ValueError('invalid input for fun.')
            else:
                return np.asarray(
                    [self.fun(_x, original_space, use_surrogate) for _x in x])
        
        start, stop = self._get_start_stop()
        if use_surrogate and self.has_surrogate:
            si = np.searchsorted(self._surrogate_recipe[:, 1], start)
            if si == self.n_surrogate:
                use_surrogate = False
        i = start
        while i <= stop:
            try:
                if use_surrogate and self.has_surrogate:
                    if i < self._surrogate_recipe[si, 1]:
                        _module = self._module_list[i]
                        di = 1
                    elif i == self._surrogate_recipe[si, 1]:
                        _module = self._surrogate_list[
                            self._surrogate_recipe[si, 0]]
                        di = self._surrogate_recipe[si, 2]
                        if si == self.n_surrogate - 1:
                            use_surrogate = False
                        else:
                            si += 1
                    else:
                        raise RuntimeError('unexpected value for i and si.')
                else:
                    _module = self._module_list[i]
                    di = 1
                _input = [var_dict._fun[n] for n in _module._input_vars]
                _output = _module.fun(*_input)
                for j, n in enumerate(_module._output_vars):
                    var_dict._fun[n] = _output[j]
                for n in _module._delete_vars:
                    del var_dict._fun[n]
            except Exception:
                raise RuntimeError(
                    'pipeline fun evaluation failed at step #{}.'.format(i))
            i += di
        return var_dict
    
    __call__ = fun
    
    def jac(self, x, original_space=None, use_surrogate=None):
        _faj = self.fun_and_jac(x, original_space, use_surrogate)
        return _faj
    
    def fun_and_jac(self, x, original_space=None, use_surrogate=None):
        original_space, use_surrogate = self._check_os_us(original_space,
                                                          use_surrogate)
        # vectorization using recursions
        # TODO: review this
        if isinstance(x, VariableDict):
            var_dict = deepcopy(x) if self.copy_input else x
        else:
            x = np.atleast_1d(x)
            if x.ndim == 1:
                if x.dtype.kind == 'f':
                    if self.copy_input:
                        x = x.copy()
                    if not original_space:
                        j = np.diag(self.to_original_grad(x))
                        x = self.to_original(x)
                    else:
                        j = np.eye(x.shape[-1])
                    var_dict = VariableDict()
                    if self._input_cum is None:
                        var_dict._fun[self._input_vars[0]] = x
                        var_dict._jac[self._input_vars[0]] = j
                    else:
                        for i, n in enumerate(self._input_vars):
                            var_dict._fun[n] = x[
                                self._input_cum[i]:self._input_cum[i + 1]]
                            var_dict._jac[n] = j[
                                self._input_cum[i]:self._input_cum[i + 1]]
                elif x.dtype.kind == 'O':
                    return np.asarray([self.fun_and_jac(
                        _x, original_space, use_surrogate) for _x in x])
                else:
                    raise ValueError('invalid input for fun_and_jac.')
            else:
                return np.asarray([self.fun_and_jac(
                    _x, original_space, use_surrogate) for _x in x])
        
        start, stop = self._get_start_stop()
        if use_surrogate and self.has_surrogate:
            si = np.searchsorted(self._surrogate_recipe[:, 1], start)
            if si == self.n_surrogate:
                use_surrogate = False
        i = start
        while i <= stop:
            try:
                if use_surrogate and self.has_surrogate:
                    if i < self._surrogate_recipe[si, 1]:
                        _module = self._module_list[i]
                        di = 1
                    elif i == self._surrogate_recipe[si, 1]:
                        _module = self._surrogate_list[
                            self._surrogate_recipe[si, 0]]
                        di = self._surrogate_recipe[si, 2]
                        if si == self.n_surrogate - 1:
                            use_surrogate = False
                        else:
                            si += 1
                    else:
                        raise RuntimeError('unexpected value for i and si.')
                else:
                    _module = self._module_list[i]
                    di = 1
                _input = [var_dict._fun[n] for n in _module._input_vars]
                _input_jac =  np.concatenate(
                    [var_dict._jac[n] for n in _module._input_vars], axis=0)
                _output, _output_jac = _module.fun_and_jac(*_input)
                for j, n in enumerate(_module._output_vars):
                    var_dict._fun[n] = _output[j]
                    var_dict._jac[n] = np.dot(_output_jac[j], _input_jac)
                for n in _module._delete_vars:
                    del var_dict._fun[n], var_dict._jac[n]
            except Exception:
                raise RuntimeError(
                    'pipeline fun_and_jac evaluation failed at step '
                    '#{}.'.format(i))
            i += di
        return var_dict
    
    @property
    def input_vars(self):
        return self._input_vars
    
    @input_vars.setter
    def input_vars(self, names):
        self._input_vars = PropertyList(
            names, lambda x: self._var_check(x, 'input', False, 'raise'))
    
    @property
    def output_vars(self):
        return self._output_vars
    
    @output_vars.setter
    def output_vars(self, names):
        if names is None or isinstance(names, str):
            self._output_vars = names
        else:
            self._output_vars = PropertyList(
                names, lambda x: self._var_check(x, 'output', False, 'remove'))
    
    @property
    def input_dims(self):
        return self._input_dims
    
    @input_dims.setter
    def input_dims(self, dims):
        if dims is None:
            self._input_dims = None
            self._input_cum = None
        else:
            self._input_dims = self._dim_check(dims)
            # we do not allow directly modify the elements of input_dims here
            # as it cannot trigger the update of input_cum
            self._input_dims.flags.writeable = False # TODO: PropertyArray?
    
    def _dim_check(self, dims):
        try:
            dims = np.atleast_1d(dims).astype(np.int)
            assert np.all(dims > 0)
            assert dims.size > 0 and dims.ndim == 1
        except Exception:
            raise ValueError(
                'input_dims should be a 1-d array_like of positive int(s), or '
                'None, instead of {}.'.format(dims))
        self._input_cum = np.cumsum(np.insert(dims, 0, 0))
        return dims
    
    @property
    def input_size(self):
        return np.sum(self.input_dims) if self.input_dims is not None else None


class Density(Pipeline, _DensityBase):
    """
    Specialized `Pipeline` for probability densities.
    
    Parameters
    ----------
    density_name : str, optional
        The name of the variable that stands for the density. Set to `__var__`
        by default.
    decay_options : dict, optional
        Keyword arguments to be passed to `self.set_decay_options`. Set to `{}`
        by default.
    args : array_like, optional
        Additional arguments to be passed to `Pipeline.__init__`.
    kwargs : dict, optional
        Additional keyword arguments to be passed to `Pipeline.__init__`.
    
    Notes
    -----
    See the docstring of `Pipeline`. Here the `output_vars` should be a str,
    and will be set to `'__var__'` by default.
    """
    def __init__(self, density_name='__var__', decay_options={}, *args,
                 **kwargs):
        self.density_name = density_name
        super().__init__(*args, **kwargs)
        self.set_decay_options(**decay_options)
    
    @property
    def density_name(self):
        return self._density_name
    
    @density_name.setter
    def density_name(self, name):
        try:
            self._density_name = str(name)
        except Exception:
            raise ValueError('invalid value for density_name.')
    
    def logp(self, x, original_space=None, use_surrogate=None):
        x = np.asarray(x)
        if x.dtype.kind != 'f':
            raise NotImplementedError('currently x should be a numpy array of '
                                      'float.')
        original_space, use_surrogate = self._check_os_us(original_space,
                                                          use_surrogate)
        _fun = self.fun(x, original_space, use_surrogate)
        _logp = VariableDict.get(_fun, self.density_name, 'fun')[..., 0]
        if self._use_decay and use_surrogate:
            x_o = x if original_space else self.to_original(x)
            beta2 = np.einsum('...i,ij,...j', x_o - self._mu, self._hess,
                              x_o - self._mu)
            _logp -= self._gamma * np.clip(beta2 - self._alpha_2, 0, np.inf)
        if not original_space:
            _logp += self._get_diff(x_trans=x)
        return _logp
    
    __call__ = logp
    
    def grad(self, x, original_space=None, use_surrogate=None):
        x = np.asarray(x)
        if x.dtype.kind != 'f':
            raise NotImplementedError('currently x should be a numpy array of '
                                      'float.')
        original_space, use_surrogate = self._check_os_us(original_space,
                                                          use_surrogate)
        _jac = self.jac(x, original_space, use_surrogate)
        _grad = VariableDict.get(_jac, self.density_name, 'jac')[..., 0, :]
        if self._use_decay and use_surrogate:
            x_o = x if original_space else self.to_original(x)
            beta2 = np.einsum('...i,ij,...j', x_o - self._mu, self._hess,
                              x_o - self._mu)
            _grad -= (2 * self._gamma * np.dot(x_o - self._mu, self._hess) *
                      (beta2 > self._alpha_2)[..., np.newaxis])
        if not original_space:
            _tog = self.to_original_grad(x)
            _grad += self.to_original_grad2(x) / _tog
        return _grad
    
    def logp_and_grad(self, x, original_space=None, use_surrogate=None):
        x = np.asarray(x)
        if x.dtype.kind != 'f':
            raise NotImplementedError('currently x should be a numpy array of '
                                      'float.')
        original_space, use_surrogate = self._check_os_us(original_space,
                                                          use_surrogate)
        _fun_and_jac = self.fun_and_jac(x, original_space, use_surrogate)
        _logp = VariableDict.get(_fun_and_jac, self.density_name, 'fun')[..., 0]
        _grad = VariableDict.get(
            _fun_and_jac, self.density_name, 'jac')[..., 0, :]
        if self._use_decay and use_surrogate:
            x_o = x if original_space else self.to_original(x)
            beta2 = np.einsum('...i,ij,...j', x_o - self._mu, self._hess,
                              x_o - self._mu)
            _logp -= self._gamma * np.clip(beta2 - self._alpha_2, 0, np.inf)
            _grad -= (2 * self._gamma * np.dot(x_o - self._mu, self._hess) *
                      (beta2 > self._alpha_2)[..., np.newaxis])
        if not original_space:
            _logp += self._get_diff(x_trans=x)
            _tog = self.to_original_grad(x)
            _grad += self.to_original_grad2(x) / _tog
        return _logp, _grad
    
    @property
    def decay_options(self):
        return DecayOptions(self._use_decay, self._alpha, self._alpha_p,
                            self._gamma)
    
    def set_decay_options(self, use_decay=False, alpha=None, alpha_p=150.,
                          gamma=0.1):
        self._use_decay = bool(use_decay)
        if alpha is None:
            self._alpha = None
            self._alpha_2 = None
        else:
            try:
                alpha = float(alpha)
                assert alpha > 0
                self._alpha = alpha
                self._alpha_2 = alpha**2
            except Exception:
                raise ValueError('invalid value for alpha.')
        if alpha_p is None:
            if alpha is None:
                raise ValueError('alpha and alpha_p cannot both be None.')
            self._alpha_p = None
        else:
            try:
                alpha_p = float(alpha_p)
                assert alpha_p > 0
                self._alpha_p = alpha_p
            except Exception:
                raise ValueError('invalid value for alpha_p.')
        try:
            gamma = float(gamma)
            assert gamma > 0
            self._gamma = gamma
        except Exception:
            raise ValueError('invalid value for gamma.')
    
    def _set_decay(self, x):
        try:
            x = np.ascontiguousarray(x)
            assert x.ndim == 2
        except Exception:
            raise ValueError('invalid value for x.')
        self._mu = np.mean(x, axis=0)
        self._hess = np.linalg.inv(np.cov(x, rowvar=False))
        if self._alpha_p is not None:
            _beta = np.einsum('ij,jk,ik->i', x - self._mu, self._hess,
                              x - self._mu)**0.5
            if self._alpha_p < 100:
                self._alpha = np.percentile(_beta, self._alpha_p)
            else:
                self._alpha = np.max(_beta) * self._alpha_p / 100
            self._alpha_2 = self._alpha**2
    
    def fit(self, var_dicts):
        if not all_isinstance(var_dicts, VariableDict):
            raise ValueError('var_dicts should consist of VariableDict(s).')
        
        x = self._get_var(var_dicts, self._input_vars)
        if self._use_decay:
            self._set_decay(x)
        logp = self._get_logp(var_dicts)
        
        for i, su in enumerate(self._surrogate_list):
            x = self._get_var(var_dicts, su._input_vars)
            if su._input_scales is not None:
                x = (x - su._input_scales[:, 0]) / su._input_scales_diff
            y = self._get_var(var_dicts, su._output_vars)
            su.fit(x, y, logp)
    
    @classmethod
    def _get_var(cls, var_dicts, var_names):
        return np.array([np.concatenate([vd._fun[vn] for vn in var_names])
                         for vd in var_dicts])
    
    def _get_logp(self, var_dicts):
        return self._get_var(var_dicts, [self.density_name])[..., 0]


class DensityLite(_PipelineBase, _DensityBase):
    """
    Directly defines probability densities with logp, grad and/or logp_and_grad.
    
    Parameters
    ----------
    logp : callable or None, optional
        Callable returning the value of logp, or `None` if undefined.
    grad : callable or None, optional
        Callable returning the value of grad_logp, or `None` if undefined.
    logp_and_grad : callable or None, optional
        Callable returning the logp and grad_logp at the same time, or `None`
        if undefined.
    input_size : None or positive int, optional
        The size of input variables. Only used to generate starting points when
        no x_0 is given during sampling. Set to `None` by default.
    input_scales : None or array_like of float(s), optional
        Controlling the scaling of input variables. Set to `None` by default.
    hard_bounds : bool or array_like, optional
        Controlling whether `input_scales` should be interpreted as hard bounds,
        or just used to rescale the variables. If bool, will be applied to all
        the variables. If array_like, should have shape of `(input_size,)` or
        `(input_size, 2)`. Set to `False` by default.
    copy_input : bool, optional
        Whether to make a copy of the input before evaluating the Pipeline. Set
        to False by default.
    vectorized : bool, optional
        Whether the original definitions of `logp`, `grad` and `logp_and_grad`
        are vectorized. If not, a wrapper will be used to enable vectorization.
        Set to False by default.
    original_space : bool, optional
        Whether the input variables are in the original, untransformed space.
        Will be overwritten if the `original_space` argument of `logp`, `grad`
        and `logp_and_grad` is not None. Set to `True` by default.
    logp_args, grad_args, logp_and_grad_args : array_like, optional
        Additional arguments to be passed to `logp`, `grad` and `logp_and_grad`.
        Will be stored as tuples.
    logp_kwargs, grad_kwargs, logp_and_grad_kwargs : dict, optional
        Additional keyword arguments to be passed to `logp`, `grad` and
        `logp_and_grad`.
    """
    def __init__(self, logp=None, grad=None, logp_and_grad=None,
                 input_size=None, input_scales=None, hard_bounds=True,
                 copy_input=False, vectorized=False, original_space=True,
                 logp_args=(), logp_kwargs={}, grad_args=(), grad_kwargs={},
                 logp_and_grad_args=(), logp_and_grad_kwargs={}):
        self.logp = logp
        self.grad = grad
        self.logp_and_grad = logp_and_grad
        
        self.input_size = input_size
        self.input_scales = input_scales
        self.hard_bounds = hard_bounds
        self.copy_input = copy_input
        self.vectorized = vectorized
        self.original_space = original_space
        
        self.logp_args = logp_args
        self.logp_kwargs = logp_kwargs
        self.grad_args = grad_args
        self.grad_kwargs = grad_kwargs
        self.logp_and_grad_args = logp_and_grad_args
        self.logp_and_grad_kwargs = logp_and_grad_kwargs
    
    @property
    def logp(self):
        if self.has_logp:
            return self._logp_wrapped
        elif self.has_logp_and_grad:
            return lambda *args: self._logp_and_grad_wrapped(*args)[0]
        else:
            raise RuntimeError('No valid definition of logp is found.')
    
    @logp.setter
    def logp(self, lp):
        if callable(lp):
            self._logp = lp
        elif lp is None:
            self._logp = None
        else:
            raise ValueError('logp should be callable, or None if you want to '
                             'reset it.')
    
    __call__ = logp
    
    def _logp_wrapped(self, x, original_space=None, use_surrogate=None):
        x = np.atleast_1d(x)
        if self.copy_input:
            x = np.copy(x)
        if original_space is None:
            original_space = self.original_space
        else:
            original_space = bool(original_space)
        x_o = x if original_space else self.to_original(x)
        if x_o.ndim == 1 or self.vectorized:
            _logp = self._logp(x_o, *self.logp_args, **self.logp_kwargs)
        else:
            _logp = np.apply_along_axis(self._logp, -1, x_o, *self.logp_args,
                                        **self.logp_kwargs).astype(np.float)
        if not original_space:
            _logp += self._get_diff(x_trans=x)
        return _logp
    
    @property
    def has_logp(self):
        return self._logp is not None
    
    @property
    def grad(self):
        if self.has_grad:
            return self._grad_wrapped
        elif self.has_logp_and_grad:
            return lambda *args: self._logp_and_grad_wrapped(*args)[1]
        else:
            raise RuntimeError('No valid definition of grad is found.')
    
    @grad.setter
    def grad(self, gd):
        if callable(gd):
            self._grad = gd
        elif gd is None:
            self._grad = None
        else:
            raise ValueError('grad should be callable, or None if you want to '
                             'reset it.')
    
    def _grad_wrapped(self, x, original_space=None, use_surrogate=None):
        x = np.atleast_1d(x)
        if self.copy_input:
            x = np.copy(x)
        if original_space is None:
            original_space = self.original_space
        else:
            original_space = bool(original_space)
        x_o = x if original_space else self.to_original(x)
        if x_o.ndim == 1 or self.vectorized:
            _grad = self._grad(x_o, *self.grad_args, **self.grad_kwargs)
        else:
            _grad = np.apply_along_axis(self._grad, -1, x_o, *self.grad_args,
                                        **self.grad_kwargs).astype(np.float)
        if not original_space:
            _tog = self.to_original_grad(x)
            _grad *= _tog
            _grad += self.to_original_grad2(x) / _tog
        return _grad
    
    @property
    def has_grad(self):
        return self._grad is not None
    
    @property
    def logp_and_grad(self):
        if self.has_logp_and_grad:
            return self._logp_and_grad_wrapped
        elif self.has_logp and self.has_grad:
            return lambda *args, **kwargs: (self._logp_wrapped(*args, **kwargs),
                                            self._grad_wrapped(*args, **kwargs))
        else:
            raise ValueError('No valid definition of logp_and_grad is found.')
    
    @logp_and_grad.setter
    def logp_and_grad(self, lpgd):
        if callable(lpgd):
            self._logp_and_grad = lpgd
        elif lpgd is None:
            self._logp_and_grad = None
        else:
            raise ValueError('logp_and_grad should be callable, or None if you'
                             'want to reset it.')
    
    def _logp_and_grad_wrapped(self, x, original_space=None,
                               use_surrogate=None):
        x = np.atleast_1d(x)
        if self.copy_input:
            x = np.copy(x)
        if original_space is None:
            original_space = self.original_space
        else:
            original_space = bool(original_space)
        x_o = x if original_space else self.to_original(x)
        if x_o.ndim == 1 or self.vectorized:
            _logp, _grad = self._logp_and_grad(x_o, *self.logp_and_grad_args,
                                               **self.logp_and_grad_kwargs)
        else:
            # TODO: review this
            _lag = np.apply_along_axis(
                self._logp_and_grad, -1, x_o, *self.logp_and_grad_args,
                **self.logp_and_grad_kwargs)
            _logp = _lag[..., 0].astype(np.float)
            _grad = np.apply_along_axis(
                lambda x: list(x), -1, _lag[..., 1]).astype(np.float)
            # otherwise, it will be an object array
        if not original_space:
            _logp += self._get_diff(x_trans=x)
            _tog = self.to_original_grad(x)
            _grad *= _tog
            _grad += self.to_original_grad2(x) / _tog
        return _logp, _grad
    
    @property
    def has_logp_and_grad(self):
        return self._logp_and_grad is not None
    
    @property
    def input_size(self):
        return self._input_size
    
    @input_size.setter
    def input_size(self, size):
        if size is None:
            self._input_size = None
        else:
            try:
                size = int(size)
                assert size > 0
            except Exception:
                raise ValueError('input_size should be a positive int, or '
                                 'None, instead of {}.'.format(size))
            self._input_size = size
    
    @property
    def vectorized(self):
        return self._vectorized
    
    @vectorized.setter
    def vectorized(self, vec):
        self._vectorized = bool(vec)
    
    _args_setter = Module._args_setter
    
    _kwargs_setter = Module._kwargs_setter
    
    @property
    def logp_args(self):
        return self._logp_args
    
    @logp_args.setter
    def logp_args(self, args):
        self._logp_args = self._args_setter(args, 'logp')
    
    @property
    def logp_kwargs(self):
        return self._logp_kwargs
    
    @logp_kwargs.setter
    def logp_kwargs(self, kwargs):
        self._logp_kwargs = self._kwargs_setter(kwargs, 'logp')
    
    @property
    def grad_args(self):
        return self._grad_args
    
    @grad_args.setter
    def grad_args(self, args):
        self._grad_args = self._args_setter(args, 'grad')
    
    @property
    def grad_kwargs(self):
        return self._grad_kwargs
    
    @grad_kwargs.setter
    def grad_kwargs(self, kwargs):
        self._grad_kwargs = self._kwargs_setter(kwargs, 'grad')
    
    @property
    def logp_and_grad_args(self):
        return self._logp_and_grad_args
    
    @logp_and_grad_args.setter
    def logp_and_grad_args(self, args):
        self._logp_and_grad_args = self._args_setter(args, 'logp_and_grad')
    
    @property
    def logp_and_grad_kwargs(self):
        return self._logp_and_grad_kwargs
    
    @logp_and_grad_kwargs.setter
    def logp_and_grad_kwargs(self, kwargs):
        self._logp_and_grad_kwargs = self._kwargs_setter(kwargs,
                                                         'logp_and_grad')
