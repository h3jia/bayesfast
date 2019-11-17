import numpy as np
from collections import namedtuple, OrderedDict
from copy import deepcopy
import warnings
from .module import *
from ..transforms._constraint import *

__all__ = ['VariableDict', 'Pipeline', 'Density', 'DensityLite']

# TODO: finish DensityLite
# TODO: add call counter
# TODO: add checks in DensityLite
# TODO: use customized PropertyList to simplify property checks
# https://stackoverflow.com/a/39190103/12292488


class VariableDict:
    
    def __init__(self):
        self._fun = OrderedDict()
        self._jac = OrderedDict()
    
    def __getitem__(self, key):
        new_dict = VariableDict()
        if isinstance(key, str):
            try:
                fun = self._fun[key]
            except:
                fun = None
            try:
                jac = self._jac[key]
            except:
                jac = None
            if fun is None and jac is None:
                warnings.warn(
                    'you asked for the key "{}", but we found neither its '
                    'fun nor its jac.'.format(k), RuntimeWarning)
            return np.asarray((fun, jac, 0))[:-1]
        elif (isinstance(key, (list, tuple)) or 
              (isinstance(key, np.ndarray) and key.dtype.kind == 'U')):
            if isinstance(key, np.ndarray):
                key = key.flatten()
            for k in key:
                try:
                    new_dict._fun[k] = self._fun[k]
                except:
                    new_dict._fun[k] = None
                try:
                    new_dict._jac[k] = self._jac[k]
                except:
                    new_dict._jac[k] = None
                if new_dict._fun[k] is None and new_dict._jac[k] is None:
                    warnings.warn(
                        'you asked for the key "{}", but we found neither its '
                        'fun nor its jac.'.format(k), RuntimeWarning)
        else:
            raise ValueError('key should be a str, or a list/tuple/np.ndarray '
                             'of str.')
        return new_dict
    
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError('key should be a str.')
        try:
            value = (value[0], value[1])
            self._fun[key] = value[0]
            self._jac[key] = value[1]
        except:
            raise ValueError('failed to get the values for fun and jac.')
    
    @property
    def fun(self):
        return self._fun
    
    @property
    def jac(self):
        return self._jac


class Pipeline:
    
    def __init__(self, module_list=None, input_vars=['__var__'], var_dims=None,
                 surrogate_list=None, var_scales=None, hard_bounds=False):
        self.module_list = module_list
        self.input_vars = input_vars
        self.var_dims = var_dims
        self.surrogate_list = surrogate_list
        self.var_scales = var_scales
        self.hard_bounds = hard_bounds
        self._needs_input_check = False
        # self.lite_options()
    
    @property
    def module_list(self):
        self._needs_ml_check = True
        return self._module_list
    
    @module_list.setter
    def module_list(self, ml):
        if ml is not None:
            if isinstance(ml, Module):
                ml = [ml]
            if hasattr(ml, '__iter__'):
                self._module_list = list(ml)
                self._ml_check()
            else:
                raise ValueError(
                    'module_list should a Module or a list of Module(s), or '
                    'None if you want to reset it.')
        else:
            self._module_list = []
            self._needs_ml_check = False
    
    def _ml_check(self):
        for i, m in enumerate(self._module_list):
            if not isinstance(m, Module):
                raise ValueError('element #{} of self.module_list is not a '
                                 'Module.'.format(i))
        self._needs_ml_check = False
    
    @property
    def surrogate_list(self):
        self._needs_sl_check = True
        return self._surrogate_list
    
    @surrogate_list.setter
    def surrogate_list(self, sl):
        if sl is not None:
            if isinstance(sl, Surrogate):
                sl = [sl]
            if hasattr(sl, '__iter__'):
                self._surrogate_list = list(sl)
                self._sl_check()
            else:
                raise ValueError(
                    'surrogate_list should a Surrogate or a list of '
                    'Surrogate(s), or None if you want to leave it empty.')
        else:
            self._surrogate_list = []
            self._needs_sl_check = False
    
    def _sl_check(self):
        for i, s in enumerate(self._surrogate_list):
            if not isinstance(s, Surrogate):
                raise ValueError('element #{} of self.surrogate_list is not a'
                                 'Surrogate'.format(i))
        self._build_surrogate_recipe()
        self._needs_sl_check = False
    
    def _build_surrogate_recipe(self):
        if self.has_surrogate:
            self._surrogate_recipe = np.array(
                [[i, *s._scope] for i, s in enumerate(self._surrogate_list)])
            _recipe_sort = np.argsort(
                self._surrogate_recipe[:, 1] % self.n_module)
            self._surrogate_recipe = (
                self._surrogate_recipe[_recipe_sort].astype(np.int))
            for i in range(self.n_surrogate - 1):
                if (np.sum(self._surrogate_recipe[i, 1:]) > 
                    self._surrogate_recipe[i + 1, 1]):
                    raise ValueError('the #{} surrogate model overlaps with '
                                     'the next one.'.format(i))
        else:
            self._surrogate_recipe = np.empty((self.n_surrogate, 3),
                                              dtype=np.int)
    
    """
    def lite_options(self, use_surrogate=False, original_space=True, start=None,
                    stop=None, extract_vars=None, copy_x=False):
        keys = ['use_surrogate', 'original_sapce', 'start', 'stop',
                'extract_vars', 'copy_x']
        values = self._options_check(use_surrogate, original_space, start, stop,
                                    extract_vars, copy_x)
        self._lite_options = OrderedDict(zip(keys, values))
    """
    
    def _options_check(self, use_surrogate, original_space, start, stop,
                      extract_vars, copy_x):
        use_surrogate = bool(use_surrogate)
        original_space = bool(original_space)
        start = self._step_check(start, 'start')
        stop = self._step_check(stop, 'stop')
        if start > stop:
            raise ValueError('start should be no larger than stop.')
        if (extract_vars is None) or isinstance(extract_vars, str):
            pass
        else:
            extract_vars = self._vars_check(extract_vars, 'extract', False,
                                            'remove')
        copy_x = bool(copy_x)
        return (use_surrogate, original_space, start, stop, extract_vars,
                copy_x)
    
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
            except:
                raise ValueError('{} should be an int or None, instead '
                                 'of {}'.format(tag, step))
        return step
    
    _vars_check = Module._vars_check
    
    def fun(self, x, use_surrogate=False, original_space=True, start=None,
            stop=None, extract_vars=None, copy_x=False):
        if self._needs_ml_check:
            self._ml_check()
        if self._needs_sl_check:
            self._sl_check()
        if self._needs_input_check:
            self._input_check()
        if copy_x:
            x = deepcopy(x)
            copy_x = False
        conf = self._options_check(use_surrogate, original_space, start, stop, 
                                  extract_vars, copy_x)
        use_surrogate, original_space, start, stop, extract_vars, copy_x = conf
        
        if not isinstance(x, VariableDict):
            x = np.atleast_1d(x)
            if x.ndim == 1:
                if x.dtype.kind == 'f':
                    var_dict = None
                elif x.dtype.kind == 'O':
                    return np.asarray([self.fun(_x, *conf) for _x in x])
                else:
                    raise ValueError('invalid input for fun.')
            else:
                return np.asarray([self.fun(_x, *conf) for _x in x])
        else:
            var_dict = x
        
        """
        # TODO: there is probably a more graceful way to vectorize
        if isinstance(x, VariableDict):
            var_dict = x
        else:
            x = np.atleast_1d(x)
            if x.dtype.kind == 'f':
                if x.ndim == 1:
                    var_dict = None
                else:
                    x = np.ascontiguousarray(x)
                    x_f = x.reshape((size, -1))
                    shape = x.shape[:-1]
                    size = np.prod(shape)
                    if isinstance(extract_vars, str):
                        result_f = np.empty(size, dtype=np.float)
                    else:
                        result_f = np.empty(size, dtype='object')
                    for i in range(size):
                        result_f[i] = self.fun(x_f[i], *conf)
                    return result_f.reshape(shape)
            elif x.dtype.kind == 'O':
                x = np.ascontiguousarray(x)
                x_f = x.reshape(size)
                shape = x.shape
                size = np.prod(shape)
                if isinstance(extract_vars, str):
                    result_f = np.empty(size, dtype=np.float)
                else:
                    result_f = np.empty(size, dtype='object')
                for i in range(size):
                    result_f[i] = self.fun(x_f[i], *conf)
                return result_f.reshape(shape)
        """
        
        if var_dict is None:
            if not original_space:
                x = self.to_original(x, False)
            var_dict = VariableDict()
            for i, n in enumerate(self._input_vars):
                var_dict._fun[n] = x[self._input_cum[i]:self._input_cum[i + 1]]
        else:
            pass
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
                for j, n in enumerate(_module._copy_vars):
                    try:
                        nn = _module._paste_vars[j]
                    except:
                        raise ValueError(
                            'failed to get the name from paste_vars for '
                            'copy_vars #{}.'.format(j))
                    var_dict._fun[nn] = np.copy(var_dict._fun[n])
                for n in _module._delete_vars:
                    del var_dict._fun[n]
            except:
                raise RuntimeError(
                    'pipeline fun evaluation failed at step #{}.'.format(i))
            i += di
        if extract_vars is None:
            return var_dict
        elif isinstance(extract_vars, str):
            return var_dict._fun[extract_vars]
        else:
            return var_dict[extract_vars]
    
    __call__ = fun
    
    """
    def fun_lite(self, x):
        return self.fun(x, **self._lite_options)
    """
    
    def jac(self, x, use_surrogate=False, original_space=True, start=None,
            stop=None, extract_vars=None, copy_x=False):
        _faj = self.fun_and_jac(x, use_surrogate, original_space, start, stop,
                                extract_vars, copy_x)
        if isinstance(extract_vars, str):
            # return np.apply_along_axis(lambda xx: xx[1], -1, _faj)
            return _faj[1]
        else:
            return _faj
    
    """
    def jac_lite(self, x):
        return self.jac(x, **self._lite_options)
    """
    
    def fun_and_jac(self, x, use_surrogate=False, original_space=True,
                    start=None, stop=None, extract_vars=None, copy_x=False):
        if self._needs_ml_check:
            self._ml_check()
        if self._needs_sl_check:
            self._sl_check()
        if self._needs_input_check:
            self._input_check()
        if copy_x:
            x = deepcopy(x)
            copy_x = False
        conf = self._options_check(use_surrogate, original_space, start, stop, 
                                  extract_vars, copy_x)
        use_surrogate, original_space, start, stop, extract_vars, copy_x = conf
        
        """
        if not isinstance(x, VariableDict):
            x = np.atleast_1d(x)
            if x.ndim == 1:
                if x.dtype.kind == 'f':
                    var_dict = None
                elif x.dtype.kind == 'O':
                    if isinstance(extract_vars, str):
                        return np.asarray([(*self.fun_and_jac(_x, *conf), 0)
                                           for _x in x])[..., :-1]
                    else:
                        return np.asarray([self.fun_and_jac(_x, *conf) 
                                           for _x in x])
                else:
                    raise ValueError('invalid input for fun_and_jac.')
            else:
                if (isinstance(extract_vars, str) and x.dtype.kind == 'f' and 
                    x.ndim == 2):
                    return np.asarray([(*self.fun_and_jac(_x, *conf), 0)
                                       for _x in x])[..., :-1]
                else:
                    return np.asarray([self.fun_and_jac(_x, *conf) for _x 
                                       in x])
            # in some cases, the additional 0 is appended to make numpy create 
            # an object array correctly, 
            # see https://stackoverflow.com/q/51005699/12292488
        else:
            var_dict = x
        """
        
        # TODO: there is probably a more graceful way to vectorize
        if isinstance(x, VariableDict):
            var_dict = x
        else:
            x = np.atleast_1d(x)
            if x.dtype.kind == 'f' and x.ndim == 1:
                var_dict = None
            else:
                x = np.ascontiguousarray(x)
                if x.dtype.kind == 'f':
                    shape = x.shape[:-1]
                    size = np.prod(shape)
                    x_f = x.reshape((size, -1))
                elif x.dtype.kind == 'O':
                    shape = x.shape
                    size = np.prod(shape)
                    x_f = x.reshape(size)
                else:
                    ValueError('invalid input for fun_and_jac.')
                if isinstance(extract_vars, str):
                    _faj0 = self.fun_and_jac(x_f[0], *conf)
                    _fshape = _faj0[0].shape
                    _jshape = _faj0[1].shape
                    result_f = np.empty((size, *_fshape), dtype=np.float)
                    result_j = np.empty((size, *_jshape), dtype=np.float)
                    result_f[0] = _faj0[0]
                    result_j[0] = _faj0[1]
                    for i in range(1, size):
                        _faj = self.fun_and_jac(x_f[i], *conf)
                        result_f[i] = _faj[0]
                        result_j[i] = _faj[1]
                    return (result_f.reshape((*shape, *_fshape)), 
                            result_j.reshape((*shape, *_jshape)))
                else:
                    result = np.empty(size, dtype='object')
                    for i in range(size):
                        result[i] = self.fun_and_jac(x_f[i], *conf)
                    return result.reshape(shape)
        
        if var_dict is None:
            if not original_space:
                j = np.diag(self.to_original_grad(x, False))
                x = self.to_original(x, False)
            else:
                j = np.eye(self._input_size)
            var_dict = VariableDict()
            for i, n in enumerate(self._input_vars):
                var_dict._fun[n] = x[self._input_cum[i]:self._input_cum[i + 1]]
                var_dict._jac[n] = j[self._input_cum[i]:self._input_cum[i + 1]]
        else:
            pass
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
                for j, n in enumerate(_module._copy_vars):
                    try:
                        nn = _module._paste_vars[j]
                    except:
                        raise ValueError(
                            'failed to get the name from paste_vars for '
                            'copy_vars #{}.'.format(j))
                    var_dict[nn] = (np.copy(var_dict._fun[n]), 
                                    np.copy(var_dict._jac[n]))
                for n in _module._delete_vars:
                    del var_dict._fun[n], var_dict._jac[n]
            except:
                raise
                """raise RuntimeError(
                    'pipeline fun_and_jac evaluation failed at step '
                    '#{}.'.format(i))"""
            i += di
        if extract_vars is None:
            return var_dict
        else:
            return var_dict[extract_vars]
    
    """
    def fun_and_jac_lite(self, x):
        return self.fun_and_jac(x, **self._lite_options)
    """
    
    @property
    def has_true_fun(self):
        return all(m.has_fun or m.has_fun_and_jac for m in self._module_list)
    
    @property
    def has_true_jac(self):
        return all(m.has_jac or m.has_fun_and_jac for m in self._module_list)
    
    @property
    def has_true_fun_and_jac(self):
        return self.has_true_fun and self.has_true_jac
    
    @property
    def input_vars(self):
        self._needs_input_check = True
        return self._input_vars
    
    @input_vars.setter
    def input_vars(self, names):
        self._input_vars = self._vars_check(names, 'input', False, 'raise')
    
    @property
    def var_dims(self):
        self._needs_input_check = True
        return self._var_dims
    
    @var_dims.setter
    def var_dims(self, dims):
        try:
            dims = np.asarray(dims, dtype=np.int).reshape(-1)
            assert np.all(dims > 0)
            assert len(dims) > 0
        except:
            raise ValueError(
                'var_dims should be an array of positive int(s), instead of '
                '{}.'.format(dims))
        self._var_dims = dims
        self._input_cum = np.cumsum(np.insert(dims, 0, 0))
        self._input_size = np.sum(dims)
    
    @property
    def input_size(self):
        return np.sum(self._var_dims)
    
    @property
    def var_scales(self):
        self._needs_input_check = True
        return self._var_scales
    
    @var_scales.setter
    def var_scales(self, scales):
        if scales is None:
            pass
        else:
            try:
                scales = np.ascontiguousarray(scales)
                if scales.shape == (self.input_size,):
                    scales = np.array((np.zeros_like(scales), scales)).T.copy()
                if not scales.shape == (self.input_size, 2):
                    raise ValueError('I do not know how to interpret the shape '
                                     'of var_scales.')
            except:
                raise ValueError('Invalid value for var_scales.')
        self._var_scales = scales
    
    @property
    def hard_bounds(self):
        self._needs_input_check = True
        return self._hard_bounds
    
    @hard_bounds.setter
    def hard_bounds(self, bounds):
        if isinstance(bounds, bool):
            self._lower_bounds = np.full((self.input_size, 2), bounds, np.uint8)
        else:
            try:
                bounds = np.ascontiguousarray(bounds).astype(bool).astype(
                    np.uint8)
                if bounds.shape == (self.input_size,):
                    bounds = np.array((bounds, bounds)).T.copy()
                elif bounds.shape == (self.input_size, 2):
                    pass
                else:
                    raise ValueError('I do not know how to interpret the shape '
                                     'of hard_bounds.')
            except:
                raise ValueError('Invalid value for hard_bounds')
            self._lower_bounds = bounds
    
    def _input_check(self):
        self.input_vars = self._input_vars
        self.var_dims = self._var_dims
        self.var_scales = self._var_scales
        self.hard_bounds = self._hard_bounds
        self._needs_input_check = False
    
    @property
    def n_module(self):
        return len(self._module_list)
    
    @property
    def n_surrogate(self):
        return len(self._surrogate_list)
    
    @property
    def has_surrogate(self):
        return self.n_surrogate > 0
    
    def _constraint(self, x, out, f, f2):
        _return = False
        x = np.ascontiguousarray(x)
        if out is None:
            out = x
        elif out is False:
            out = np.empty_like(x)
            _return = True
        else:
            if not (isinstance(out, np.ndarray) and x.shape == out.shape):
                raise ValueError('invalid value for out.')
            out = np.ascontiguousarray(out)
        if x.ndim == 1:
            f(x, self._var_scales, out, self._hard_bounds, x.shape[0])
        elif x.ndim == 2:
            f2(x, self._var_scales, out, self._hard_bounds, x.shape[1],
               x.shape[0])
        else:
            raise NotImplementedError('x should be 1-d or 2-d for now.')
        if _return:
            return out
        
    def from_original(self, x, out=False):
        if self._var_scales is None:
            if out is None:
                return
            elif out is False:
                return x.copy()
            else:
                if not (isinstance(out, np.ndarray) and x.shape == out.shape):
                    raise ValueError('invalid value for out.')
                out = x.copy()
                return
        else:
            return self._constraint(x, out, _from_original_f, _from_original_f2)
    
    def from_original_grad(self, x, out=False):
        raise NotImplementedError
    
    def from_original_grad2(self, x, out=False):
        raise NotImplementedError
    
    def to_original(self, x, out=False):
        if self._var_scales is None:
            if out is None:
                return
            elif out is False:
                return x.copy()
            else:
                if not (isinstance(out, np.ndarray) and x.shape == out.shape):
                    raise ValueError('invalid value for out.')
                out = x.copy()
                return
        else:
            return self._constraint(x, out, _to_original_f, _to_original_f2)
    
    def to_original_grad(self, x, out=False):
        if self._var_scales is None:
            if out is None:
                x = np.ones_like(x)
                return
            elif out is False:
                return np.ones_like(x)
            else:
                if not (isinstance(out, np.ndarray) and x.shape == out.shape):
                    raise ValueError('invalid value for out.')
                out = np.ones_like(x)
                return
        else:
            return self._constraint(x, out, _to_original_j, _to_original_j2)
    
    def to_original_grad2(self, x, out=False):
        if self._var_scales is None:
            if out is None:
                x = np.zeros_like(x)
                return
            elif out is False:
                return np.zeros_like(x)
            else:
                if not (isinstance(out, np.ndarray) and x.shape == out.shape):
                    raise ValueError('invalid value for out.')
                out = np.zeros_like(x)
                return
        else:
            return self._constraint(x, out, _to_original_jj, _to_original_jj2)
    
    def print_summary(self):
        raise NotImplementedError
        

class Density(Pipeline):
    
    def __init__(self, density_name='__var__', *args, **kwargs):
        self.density_name = density_name
        super().__init__(*args, **kwargs)
        self._use_decay = False
    
    @property
    def density_name(self):
        return self._density_name
    
    @density_name.setter
    def density_name(self, name):
        if isinstance(name, str):
            self._density_name = name
        else:
            raise ValueError(
                'density_name should be a str, instead of {}'.format(name))
    
    def logp(self, x, use_surrogate=False, original_space=True, start=None,
             stop=None, copy_x=False):
        _logp = self.fun(x, use_surrogate, original_space, start, stop,
                         self._density_name, copy_x)[..., 0]
        if self._use_decay:
            x_o = x if original_space else self.to_original(x)
            beta2 = np.einsum('...i,ij,...j', x_o - self._mu, self._hess, 
                              x_o - self._mu)
            _logp += -self._gamma * np.clip(beta2 - self._alpha2, 0, np.inf)
        if not original_space:
            _logp += np.sum(np.log(np.abs(self.to_original_grad(x, False))))
        return _logp
    
    __call__ = logp
    
    """
    def logp_lite(self, x):
        conf = self._lite_options.copy()
        del conf['extract_vars']
        return self.logp(x, **conf)
    """
    
    def grad(self, x, use_surrogate=False, original_space=True, start=None,
             stop=None, copy_x=False):
        _grad = self.jac(x, use_surrogate, original_space, start, stop,
                         self._density_name, copy_x)[..., 0, :]
        if self._use_decay:
            x_o = x if original_space else self.to_original(x)
            beta2 = np.einsum('...i,ij,...j', x_o - self._mu, self._hess, 
                              x_o - self._mu)
            _grad += (-2 * self._gamma * np.dot(x_o - self._mu, self._hess) * 
                      (beta2 > self._alpha2)[..., np.newaxis])
        if not original_space:
            _grad += (self.to_original_grad2(x, False) / 
                      self.to_original_grad(x, False))
        return _grad
    
    """
    def grad_lite(self, x):
        conf = self._lite_options.copy()
        del conf['extract_vars']
        return self.grad(x, **conf)
    """
    
    def logp_and_grad(self, x, use_surrogate=False, original_space=True, 
                      start=None, stop=None, copy_x=False):
        _logp_and_grad = self.fun_and_jac(
            x, use_surrogate, original_space, start, stop, self._density_name, 
            copy_x)
        _logp = _logp_and_grad[0][..., 0]
        _grad = _logp_and_grad[1][..., 0, :]
        if self._use_decay:
            x_o = x if original_space else self.to_original(x)
            beta2 = np.einsum('...i,ij,...j', x_o - self._mu, self._hess, 
                              x_o - self._mu)
            _logp += -self._gamma * np.clip(beta2 - self._alpha2, 0, np.inf)
            _grad += (-2 * self._gamma * np.dot(x_o - self._mu, self._hess) * 
                      (beta2 > self._alpha2)[..., np.newaxis])
        if not original_space:
            _logp += np.sum(np.log(np.abs(self.to_original_grad(x, False))))
            _grad += (self.to_original_grad2(x, False) / 
                      self.to_original_grad(x, False))
        return _logp, _grad
    
    """
    def logp_and_grad_lite(self, x):
        conf = self._lite_options.copy()
        del conf['extract_vars']
        return self.logp_and_grad(x, **conf)
    """
    
    @property
    def alpha(self):
        try:
            return self._alpha
        except:
            return None
    
    @alpha.setter
    def alpha(self, a):
        try:
            a = float(a)
            assert a > 0
        except:
            raise ValueError('alpha should be a positive float.')
        self._alpha = a
        self._alpha2 = a**2
    
    @property
    def gamma(self):
        try:
            return self._gamma
        except:
            return None
    
    @gamma.setter
    def gamma(self, g):
        try:
            g = float(g)
            assert g > 0
        except:
            raise ValueError('gamma should be a positive float.')
        self._gamma = g
    
    @property
    def mu(self):
        try:
            return self._mu
        except:
            return None
    
    @property
    def hess(self):
        try:
            return self._hess
        except:
            return None
    
    @property
    def use_decay(self):
        return self._use_decay
    
    @use_decay.setter
    def use_decay(self, decay):
        self._use_decay = bool(decay)
    
    def set_decay(self, x, original_space=True, alpha=None, alpha_p=150,
                  gamma=None):
        try:
            x = np.ascontiguousarray(x)
            assert x.shape[-1] == self._input_size
        except:
            raise ValueError('invalid value for x.')
        x = x.reshape((-1, self._input_size))
        x_o = x if original_space else self.to_original(x)
        self._mu = np.mean(x_o, axis=0)
        self._hess = np.linalg.inv(np.cov(x_o, rowvar=False))
        if alpha is None:
            if (self.alpha is None) or (alpha_p is not None):
                try:
                    alpha_p = float(alpha_p)
                    assert alpha_p > 0
                except:
                    raise ValueError('alpha_p should be a positive float.')
                _beta = np.einsum('ij,jk,ik->i', x_o - self._mu, self._hess, 
                                  x_o - self._mu)**0.5
                if alpha_p < 100:
                    self.alpha = np.percentile(_beta, alpha_p)
                else:
                    self.alpha = np.max(_beta) * alpha_p / 100
            else:
                pass
        else:
            self.alpha = alpha
        if gamma is None:
            if self.gamma is None:
                self._gamma = 0.1
            else:
                pass
        else:
            self.gamma = gamma
        self._use_decay = True
    
    def fit(self, var_dicts, use_decay=True, decay_options={}, fit_options={}):
        if not (hasattr(var_dicts, '__iter__') and
                all(isinstance(vd, VariableDict) for vd in var_dicts)):
            raise ValueError('var_dicts should consist of VariableDict(s).')
        
        if not isinstance(decay_options, dict):
            raise ValueError('decay_options should be a dict.')
        
        if isinstance(fit_options, dict):
            fit_options = [fit_options for i in range(self.n_surrogate)]
        elif (hasattr(fit_options, '__iter__') and 
              all(isinstance(fi, dict) for fi in fit_options)):
            fit_options = list(fit_options)
            if len(fit_options) < self.n_surrogate:
                fit_options.extend([{} for i in range(self.n_surrogate - 
                                                      len(fit_options))])
        else:
            raise ValueError(
                'fit_options should be a dict or consist of dict(s).')
        
        if use_decay:
            x = self._fit_var(var_dicts, self._input_vars)
            self.set_decay(x, **decay_options)
        else:
            self._use_decay = False
        
        for i, su in enumerate(self._surrogate_list):
            x = self._fit_var(var_dicts, su._input_vars)
            y = self._fit_var(var_dicts, su._output_vars)
            su.fit(x, y, **fit_options[i])

    @classmethod
    def _fit_var(cls, var_dicts, var_names):
        return np.array([np.concatenate([vd._fun[vn] for vn in var_names]) 
                         for vd in var_dicts])


class DensityLite:
    '''NOT FINISHED YET'''
    def __init__(self, logp, grad, logp_and_grad, input_size, var_scales=None,
                 hard_bounds=False, logp_args=(), logp_kwargs={}, grad_args=(),
                 grad_kwargs={}, logp_and_grad_args=(),
                 logp_and_grad_kwargs={}):
        raise NotImplementedError
    
    @property
    def logp(self):
        if self.has_logp:
            return self._logp
        elif self.has_logp_and_grad:
            return lambda *args: self._logp_and_grad(*args)[0]
        else:
            raise RuntimeError('No valid definition of logp is found.')
    
    @logp.setter
    def logp(self, logp_):
        if callable(logp_):
            self._logp = logp_
        elif logp_ is None:
            self._logp = None
        else:
            raise ValueError('logp should be callable, or None if you want to '
                             'reset it.')
    
    @property
    def has_logp(self):
        return (self._logp is not None)
    
    @property
    def grad(self):
        if self.has_grad:
            return self._grad
        elif self.has_logp_and_grad:
            return lambda *args: self._logp_and_grad(*args)[1]
        else:
            raise RuntimeError('No valid definition of grad is found.')
    
    @grad.setter
    def grad(self, grad_):
        if callable(grad_):
            self._grad = grad_
        elif grad_ is None:
            self._grad = None
        else:
            raise ValueError('grad should be callable, or None if you want to '
                             'reset it.')
    
    @property
    def has_grad(self):
        return (self._grad is not None)
    
    @property
    def logp_and_grad(self):
        if self.has_logp_and_grad:
            return self._logp_and_grad
        elif self.has_logp and self.has_grad:
            return lambda *args: (self._logp(*args), self._grad(*args))
        else:
            raise ValueError('No valid definition of logp_and_grad is found.')
    
    @logp_and_grad.setter
    def logp_and_grad(self, logp_and_grad_):
        if callable(logp_and_grad_):
            self._logp_and_grad = logp_and_grad_
        elif logp_and_grad_ is None:
            self._logp_and_grad = None
        else:
            raise ValueError('logp_and_grad should be callable, or None if you'
                             'want to reset it.')
    
    @property
    def has_logp_and_grad(self):
        return (self._logp_and_grad is not None)
    
    @property
    def has_true_fun(self):
        return self.has_logp or self.has_logp_and_grad
    
    @property
    def has_true_jac(self):
        return self.has_grad or self.has_logp_and_grad
    
    @property
    def has_true_fun_and_jac(self):
        return self.has_true_fun and self.has_true_jac
    
"""
def DensityLite(logp=None, grad=None, logp_and_grad=None, dim=None, 
                logp_args=(), logp_kwargs={}, grad_args=(), 
                grad_kwargs={}, logp_and_grad_args=(), 
                logp_and_grad_kwargs={}, var_scales=None, 
                lower_bounds=False, upper_bounds=False):
    if callable(logp):
        _fun = lambda x, *args, **kwargs: logp(x, *args, **kwargs)
    else:
        _fun = None
    if callable(grad):
        _jac = lambda x, *args, **kwargs: [grad(x, *args, **kwargs)]
    else:
        _jac = None
    if callable(logp_and_grad):
        def _fun_and_jac(x, *args, **kwargs):
            ff, jj = logp_and_grad(x, *args, **kwargs)
            return ff, [jj]
    else:
        _fun_and_jac = None
    logp_module = Module(fun=_fun, jac=_jac, fun_and_jac=_fun_and_jac, 
                         input_vars=['__var__'], output_names=['__var__'], 
                         label='logp', concatenate_input=True, 
                         fun_args=logp_args, fun_kwargs=logp_kwargs, 
                         jac_args=grad_args, jac_kwargs=grad_kwargs, 
                         fun_and_jac_args=logp_and_grad_args, 
                         fun_and_jac_kwargs=logp_and_grad_kwargs)
    logp_density = Density(density_name='__var__', module_list=[logp_module], 
                           input_vars=['__var__'], var_dims=[dim], 
                           var_scales=var_scales, lower_bounds=lower_bounds, 
                           upper_bounds=upper_bounds)
    return logp_density
"""