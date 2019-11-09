import numpy as np
from collections import namedtuple
import warnings

__all__ = ['Module', 'Surrogate']

# TODO: use customized PropertyList to simplify property checks
# https://stackoverflow.com/a/39190103/12292488


class Module:

    def __init__(self, fun=None, jac=None, fun_and_jac=None,
                 input_vars=['__var__'], output_vars=['__var__'],
                 copy_vars=None, paste_vars=None, delete_vars=None,
                 recombine_input=False, recombine_output=False,
                 var_scales=None, label=None, fun_args=(), fun_kwargs={},
                 jac_args=(), jac_kwargs={}, fun_and_jac_args=(),
                 fun_and_jac_kwargs={}):
        self._fun_jac_init(fun, jac, fun_and_jac)
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.copy_vars = copy_vars
        self.paste_vars = paste_vars
        self.delete_vars = delete_vars
        self.recombine_input = recombine_input
        self.recombine_output = recombine_output
        self.var_scales = var_scales
        self.label = label
        
        self.fun_args = fun_args
        self.fun_kwargs = fun_kwargs
        self.jac_args = jac_args
        self.jac_kwargs = jac_kwargs
        self.fun_and_jac_args = fun_and_jac_args
        self.fun_and_jac_kwargs = fun_and_jac_kwargs
        
        self._needs_all_check = False
        self.reset_counter()
        
    def _fun_jac_init(self, fun, jac, fun_and_jac):
        self.fun = fun
        self.jac = jac
        self.fun_and_jac = fun_and_jac
    
    def _recombine(self, args, tag):
        if self._needs_all_check:
            self._all_check()
        if tag == 'input':
            strategy = self._recombine_input
            cum = self._input_cum
            dim = 1
            tag_1 = 'input parameters'
            tag_2 = 'self.recombine_input'
        elif tag == 'output_fun':
            strategy = self._recombine_output
            cum = self._output_cum
            dim = 1
            tag_1 = 'output of fun'
            tag_2 = 'self.recombine_output'
        elif tag == 'output_jac':
            strategy = self._recombine_output
            cum = self._output_cum
            dim = 2
            tag_1 = 'output of jac'
            tag_2 = 'self.recombine_output'
        else:
            raise RuntimeError('unexpected value for tag in self._recombine.')
        
        args = self._adjust_dim(args, dim, tag_1)
        if strategy is False:
            if tag == 'input' and self._var_scales is not None:
                strategy = np.array([a.shape[0] for a in args], dtype=np.int)
                cum = np.cumsum(np.insert(strategy, 0, 0))
            else:
                return args
        try:
            cargs = np.concatenate(args, axis=0)
        except:
            raise ValueError('failed to concatenate {}.'.format(tag_1))
        if tag == 'input' and self._var_scales is not None:
            try:
                cargs = (cargs - self._var_scales[:, 0]) / self._var_scales_diff
            except:
                raise ValueError('failed to rescale the input.')
        if strategy is True:
            return [cargs]
        elif isinstance(strategy, np.ndarray):
            try:
                return [cargs[cum[i]:cum[i + 1]] for i in range(strategy.size)]
            except:
                raise ValueError('failed to split {}.'.format(tag_1))
        else:
            raise RuntimeError('unexpected value for {}.'.format(tag_2))
    
    @classmethod
    def _adjust_dim(cls, args, dim, tag):
        if dim == 1:
            f = np.atleast_1d
        elif dim == 2:
            f = np.atleast_2d
        else:
            raise RuntimeError('unexpected value for dim in self._adjust_dim.')
        try:
            if (isinstance(args, (list, tuple)) or 
                (isinstance(args, np.ndarray) and args.dtype.kind == 'O')):
                args = [f(a) for a in args]
            else:
                args = [f(args)]
            assert all(a.ndim == dim for a in args)
            return args
        except:
            raise ValueError('invalid value for {}.'.format(tag))
    
    @property
    def fun(self):
        if self.has_fun:
            self._ncall_fun += 1
            return self._fun_wrapped
        elif self.has_fun_and_jac:
            self._ncall_fun_and_jac += 1
            return lambda *args: self._fun_and_jac_wrapped(*args)[0]
        else:
            raise RuntimeError('No valid definition of fun is found.')
            
    @fun.setter
    def fun(self, function):
        if callable(function):
            self._fun = function
        elif function is None:
            self._fun = None
        else:
            raise ValueError('fun should be callable, or None if you want to '
                             'reset it.')
    
    def _fun_wrapped(self, *args):
        args = self._recombine(args, 'input')
        fun_out = self._fun(*args, *self._fun_args, **self._fun_kwargs)
        return self._recombine(fun_out, 'output_fun')
    
    @property
    def has_fun(self):
        return (self._fun is not None)
    
    __call__ = fun
    
    @property
    def jac(self):
        if self.has_jac:
            self._ncall_jac += 1
            return self._jac_wrapped
        elif self.has_fun_and_jac:
            self._ncall_fun_and_jac += 1
            return lambda *args: self._fun_and_jac_wrapped(*args)[1]
        else:
            raise RuntimeError('No valid definition of jac is found.')
            
    @jac.setter
    def jac(self, jacobian):
        if callable(jacobian):
            self._jac = jacobian
        elif jacobian is None:
            self._jac = None
        else:
            raise ValueError('jac should be callable, or None if you want to '
                             'reset it.')
    
    def _jac_wrapped(self, *args):
        args = self._recombine(args, 'input')
        jac_out = self._jac(*args, *self._jac_args, **self._jac_kwargs)
        jac_out = self._recombine(jac_out, 'output_jac')
        return [j / self._var_scales_diff for j in jac_out]
    
    @property
    def has_jac(self):
        return (self._jac is not None)
    
    @property
    def fun_and_jac(self):
        if self.has_fun_and_jac:
            self._ncall_fun_and_jac += 1
            return self._fun_and_jac_wrapped
        elif self.has_fun and self.has_jac:
            self._ncall_fun += 1
            self._ncall_jac += 1
            return lambda *args: (self._fun_wrapped(*args), 
                                  self._jac_wrapped(*args))
        else:
            raise RuntimeError('No valid definition of fun_and_jac is found.')
            
    @fun_and_jac.setter
    def fun_and_jac(self, fun_jac):
        if callable(fun_jac):
            self._fun_and_jac = fun_jac
        elif fun_jac is None:
            self._fun_and_jac = None
        else:
            raise ValueError('fun_and_jac should be callable, or None if you '
                             'want to reset it.')
            
    def _fun_and_jac_wrapped(self, *args):
        args = self._recombine(args, 'input')
        fun_out, jac_out = self._fun_and_jac(
            *args, *self.fun_and_jac_args, **self.fun_and_jac_kwargs)
        fun_out = self._recombine(fun_out, 'output_fun')
        jac_out = self._recombine(jac_out, 'output_jac')
        return (fun_out, [j / self._var_scales_diff for j in jac_out])
    
    @property
    def has_fun_and_jac(self):
        return (self._fun_and_jac is not None)
    
    @property
    def ncall_fun(self):
        return self._ncall_fun
    
    @property
    def ncall_jac(self):
        return self._ncall_jac
    
    @property
    def ncall_fun_and_jac(self):
        return self._ncall_fun_and_jac
    
    @classmethod
    def _vars_check(cls, names, tag, allow_empty=False, handle_repeat='remove'):
        if allow_empty and names is None:
            return []
        if isinstance(names, str):
            names = [names]
        else:
            try:
                names = list(names)
                assert all(isinstance(nn, str) for nn in names)
                if not allow_empty:
                    assert len(names) > 0
            except:
                _none_msg = 'or None, ' if allow_empty else ''
                raise ValueError(
                    '{}_vars should be a str, or a list of str(s), {}instead '
                    'of {}'.format(tag, _none_msg, names))
            if handle_repeat == 'remove':
                names = list(set(names))
            elif handle_repeat == 'ignore':
                pass
            elif handle_repeat == 'warn':
                names = list(set(names))
                warnings.warn('repeated elements found in {}_vars'.format(tag),
                              RuntimeWarning)
            elif handle_repeat == 'raise':
                if len(names) != len(set(names)):
                    raise ValueError(
                        'some elements in {}_vars are not unique.'.format(tag))
            else:
                raise RuntimeError('unexpected value for handle_repeat.')
        return names
    
    @property
    def input_vars(self):
        self._needs_all_check = True
        return self._input_vars
    
    @input_vars.setter
    def input_vars(self, names):
        self._input_vars = self._vars_check(names, 'input', False, 'ignore')
        
    @property
    def output_vars(self):
        self._needs_all_check = True
        return self._output_vars
    
    @output_vars.setter
    def output_vars(self, names):
        self._output_vars = self._vars_check(names, 'output', False, 'remove')
    
    @property
    def copy_vars(self):
        self._needs_all_check = True
        return self._copy_vars
    
    @copy_vars.setter
    def copy_vars(self, names):
        self._copy_vars = self._vars_check(names, 'copy', True, 'ignore')
    
    @property
    def paste_vars(self):
        self._needs_all_check = True
        return self._paste_vars
    
    @paste_vars.setter
    def paste_vars(self, names):
        self._paste_vars = self._vars_check(names, 'paste', True, 'raise')
        
    @property
    def delete_vars(self):
        self._needs_all_check = True
        return self._delete_vars
    
    @delete_vars.setter
    def delete_vars(self, names):
        self._delete_vars = self._vars_check(names, 'delete', True, 'remove')
    
    def _all_check(self):
        self.input_vars = self._input_vars
        self.output_vars = self._output_vars
        self.copy_vars = self._copy_vars
        self.paste_vars = self._paste_vars
        self.delete_vars = self._delete_vars
        self.recombine_input = self._recombine_input
        self.recombine_output = self._recombine_output
        self.var_scales = self._var_scales
        self._needs_all_check = False
    
    @classmethod
    def _recombine_setter(cls, recombine, tag):
        if isinstance(recombine, bool):
            pass
        else:
            try:
                recombine = np.asarray(recombine, dtype=np.int).reshape(-1)
                assert np.all(recombine > 0)
            except:
                raise ValueError('invalid value for {}_recombine.'.format(tag))
        return recombine
    
    @property
    def recombine_input(self):
        self._needs_all_check = True
        return self._recombine_input
    
    @recombine_input.setter
    def recombine_input(self, recombine):
        recombine = self._recombine_setter(recombine, 'input')
        if isinstance(recombine, np.ndarray):
            self._input_cum = np.cumsum(np.insert(recombine, 0, 0))
        else:
            self._input_cum = None
        self._recombine_input = recombine
    
    @property
    def recombine_output(self):
        self._needs_all_check = True
        return self._recombine_output
    
    @recombine_output.setter
    def recombine_output(self, recombine):
        recombine = self._recombine_setter(recombine, 'output')
        if isinstance(recombine, np.ndarray):
            self._output_cum = np.cumsum(np.insert(recombine, 0, 0))
        else:
            self._output_cum = None
        self._recombine_output = recombine
    
    @property
    def var_scales(self):
        self._needs_all_check = True
        return self._var_scales
    
    @var_scales.setter
    def var_scales(self, scales):
        if scales is None:
            pass
        else:
            try:
                scales = np.ascontiguousarray(scales)
                if scales.ndim == 1:
                    scales = np.array((np.zeros_like(scales), scales)).T.copy()
                if not (scales.ndim == 2 and scales.shape[-1] == 2):
                    raise ValueError('I do not know how to interpret the shape '
                                     'of var_scales.')
            except:
                raise ValueError('Invalid value for var_scales.')
        self._var_scales = scales
        if scales is not None:
            self._var_scales_diff = scales[:, 1] - scales[:, 0]
        else:
            self._var_scales_diff = 1
    
    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, tag):
        if isinstance(tag, str) or tag is None:
            self._label = tag
        else:
            raise ValueError(
            'label should be a str or None, instead of {}.'.format(tag))
    
    @classmethod
    def _args_setter(cls, args, tag):
        if args is None:
            return ()
        else:
            try:
                return tuple(args)
            except:
                raise ValueError('{}_args should be a tuple.'.format(tag))
    
    @classmethod
    def _kwargs_setter(cls, kwargs, tag):
        if kwargs is None:
            return {}
        else:
            try:
                return dict(kwargs)
            except:
                raise ValueError('{}_kwargs should be a dict.'.format(tag))
        
    @property
    def fun_args(self):
        return self._fun_args
    
    @fun_args.setter
    def fun_args(self, args):
        self._fun_args = self._args_setter(args, 'fun')
            
    @property
    def fun_kwargs(self):
        return self._fun_kwargs
    
    @fun_kwargs.setter
    def fun_kwargs(self, kwargs):
        self._fun_kwargs = self._kwargs_setter(kwargs, 'fun')
            
    @property
    def jac_args(self):
        return self._jac_args
    
    @jac_args.setter
    def jac_args(self, args):
        self._jac_args = self._args_setter(args, 'jac')
            
    @property
    def jac_kwargs(self):
        return self._jac_kwargs
    
    @jac_kwargs.setter
    def jac_kwargs(self, kwargs):
        self._jac_kwargs = self._kwargs_setter(kwargs, 'jac')
            
    @property
    def fun_and_jac_args(self):
        return self._fun_and_jac_args
    
    @fun_and_jac_args.setter
    def fun_and_jac_args(self, args):
        self._fun_and_jac_args = self._args_setter(args, 'fun_and_jac')
            
    @property
    def fun_and_jac_kwargs(self):
        return self._fun_and_jac_kwargs
    
    @fun_and_jac_kwargs.setter
    def fun_and_jac_kwargs(self, kwargs):
        self._fun_and_jac_kwargs = self._kwargs_setter(kwargs, 'fun_and_jac')
            
    def reset_counter(self):
        self._ncall_fun = 0
        self._ncall_jac = 0
        self._ncall_fun_and_jac = 0
    
    def print_summary(self):
        raise NotImplementedError
        
        
SurrogateScope = namedtuple('SurrogateScope', ['start', 'extent'])


class Surrogate(Module):
    
    fixed_sizes = True
    
    def __init__(self, scope, input_size=None, output_size=None,
                 input_vars=['__var__'], output_vars=['__var__'],
                 copy_vars=None, paste_vars=None, delete_vars=None,
                 recombine_input=True, *args, **kwargs):
        self._initialized = False
        if input_size is None:
            try:
                assert not isinstance(recombine_input, bool)
                input_size = int(np.sum(recombine_input))
                assert input_size > 0
            except:
                raise ValueError(
                    'failed to infer input_size from recombine_input.')
        if output_size is None:
            try:
                assert not isinstance(recombine_output, bool)
                output_size = int(np.sum(recombine_output))
                assert output_size > 0
            except:
                raise ValueError(
                    'failed to infer output_size from recombine_output.')
        self.scope = scope
        self.input_size = input_size
        self.output_size = output_size
        super().__init__(input_vars=input_vars, output_vars=output_vars,
                         copy_vars=copy_vars, paste_vars=paste_vars,
                         delete_vars=delete_vars, 
                         recombine_input=recombine_input, *args, **kwargs)
        if not hasattr(self, '_fun'):
            self._fun = None
        if not hasattr(self, '_jac'):
            self._jac = None
        if not hasattr(self, '_fun_and_jac'):
            sele._fun_and_jac = None
        self._initialized = False
    
    def _fun_jac_init(self, fun, jac, fun_and_jac):
        pass
        
    @property
    def scope(self):
        return self._scope
    
    @scope.setter
    def scope(self, s):
        try:
            start, extent = s
            assert extent > 0
            self._scope = SurrogateScope(int(start), int(extent))
        except:
            raise ValueError('invalid value for scope.')
    
    @property
    def input_size(self):
        return self._input_size
    
    @input_size.setter
    def input_size(self, size):
        if self.fixed_sizes and self._initialized:
            raise RuntimeError('you set fixed_sizes as True, so input_size '
                               'cannot be modified after initialization.')
        else:
            try:
                size = int(size)
                assert size > 0
            except:
                raise ValueError('input_size should be a positive int.')
            self._input_size = size
    
    @property
    def output_size(self):
        return self._output_size
    
    @output_size.setter
    def output_size(self, size):
        if self.fixed_sizes and self._initialized:
            raise RuntimeError('you set fixed_sizes as True, so output_size '
                               'cannot be modified after initialization.')
        else:
            try:
                size = int(size)
                assert size > 0
            except:
                raise ValueError('output_size should be a positive int.')
            self._output_size = size
    
    var_scales = property(Module.var_scales.__get__)
    
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
        self._var_scales_diff = scales[:, 1] - scales[:, 0]
    
    def fit(self, *args, **kwargs):
        raise NotImplementedError('Abstract Method.')
