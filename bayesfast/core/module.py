import numpy as np
from collections import namedtuple
from ..utils.collections import PropertyList
from ..utils import all_isinstance
import warnings

__all__ = ['ModuleBase', 'Module', 'Surrogate']

# TODO: implement `Module.print_summary()`
# TODO: PropertyArray?
# TODO: check if Surrogate has been fitted?


class ModuleBase:
    """Base class for Module. To use it, please manually set its fun etc."""
    def __init__(self, input_vars=('__var__',), output_vars=('__var__',),
                 delete_vars=(), concat_input=False, concat_output=False,
                 input_scales=None, label=None, fun_args=(), fun_kwargs=None,
                 jac_args=(), jac_kwargs=None, fun_and_jac_args=(),
                 fun_and_jac_kwargs=None):
        if self.__class__.input_vars.fset is not None:
            self.input_vars = input_vars
        if self.__class__.output_vars.fset is not None:
            self.output_vars = output_vars

        self.delete_vars = delete_vars
        self.concat_input = concat_input
        self.concat_output = concat_output
        self.input_scales = input_scales
        self.label = label

        self.fun_args = fun_args
        self.fun_kwargs = fun_kwargs
        self.jac_args = jac_args
        self.jac_kwargs = jac_kwargs
        self.fun_and_jac_args = fun_and_jac_args
        self.fun_and_jac_kwargs = fun_and_jac_kwargs

        self.reset_counter()

    def _concat(self, args, tag):
        if tag == 'input':
            strategy = self._concat_input
            cum = self._input_cum
            dim = 1
            tag_1 = 'input variables'
            tag_2 = 'self.concat_input'
        elif tag == 'output_fun':
            strategy = self._concat_output
            cum = self._output_cum
            dim = 1
            tag_1 = 'output of fun'
            tag_2 = 'self.concat_output'
        elif tag == 'output_jac':
            strategy = self._concat_output
            cum = self._output_cum
            dim = 2
            tag_1 = 'output of jac'
            tag_2 = 'self.concat_output'
        else:
            raise RuntimeError('unexpected value for tag in self._concat.')

        args = self._adjust_dim(args, dim, tag_1)
        if strategy is False:
            if tag == 'input' and self._input_scales is not None:
                strategy = np.array([a.shape[0] for a in args], dtype=np.int)
                cum = np.cumsum(np.insert(strategy, 0, 0))
            else:
                return args
        try:
            cargs = np.concatenate(args, axis=0)
        except Exception:
            raise ValueError('failed to concatenate {}.'.format(tag_1))
        if tag == 'input' and self._input_scales is not None:
            try:
                cargs = ((cargs - self._input_scales[:, 0]) /
                         self._input_scales_diff)
            except Exception:
                raise ValueError('failed to rescale the input variables.')
        if strategy is True:
            return [cargs]
        elif isinstance(strategy, np.ndarray):
            try:
                return [cargs[cum[i]:cum[i + 1]] for i in range(strategy.size)]
            except Exception:
                raise ValueError('failed to split {}.'.format(tag_1))
        else:
            raise RuntimeError('unexpected value for {}.'.format(tag_2))

    @staticmethod
    def _adjust_dim(args, dim, tag):
        if dim == 1:
            f = np.atleast_1d
        elif dim == 2:
            f = np.atleast_2d
        else:
            raise RuntimeError('unexpected value for dim in self._adjust_dim.')
        try:
            # if args is a list/tuple/object-array
            # it's regarded as a collection of variables
            # otherwise, it's regarded as a single variable
            if (isinstance(args, (list, tuple)) or
                (isinstance(args, np.ndarray) and args.dtype.kind == 'O')):
                args = [f(a) for a in args]
            else:
                args = [f(args)]
            assert all(a.ndim == dim for a in args)
            return args
        except Exception:
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
        args = self._concat(args, 'input')
        fun_out = self._fun(*args, *self._fun_args, **self._fun_kwargs)
        return self._concat(fun_out, 'output_fun')

    @property
    def has_fun(self):
        try:
            return self._fun is not None
        except Exception:
            return False

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
        args = self._concat(args, 'input')
        jac_out = self._jac(*args, *self._jac_args, **self._jac_kwargs)
        jac_out = self._concat(jac_out, 'output_jac')
        return [j / self._input_scales_diff for j in jac_out]

    @property
    def has_jac(self):
        try:
            return self._jac is not None
        except Exception:
            return False

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
        args = self._concat(args, 'input')
        fun_out, jac_out = self._fun_and_jac(
            *args, *self.fun_and_jac_args, **self.fun_and_jac_kwargs)
        fun_out = self._concat(fun_out, 'output_fun')
        jac_out = self._concat(jac_out, 'output_jac')
        return (fun_out, [j / self._input_scales_diff for j in jac_out])

    @property
    def has_fun_and_jac(self):
        try:
            return self._fun_and_jac is not None
        except Exception:
            return False

    @property
    def ncall_fun(self):
        return self._ncall_fun

    @property
    def ncall_jac(self):
        return self._ncall_jac

    @property
    def ncall_fun_and_jac(self):
        return self._ncall_fun_and_jac

    @staticmethod
    def _var_check(names, tag, allow_empty=False, handle_repeat='remove'):
        if isinstance(names, str):
            names = [names]
        else:
            try:
                names = list(names)
                assert all_isinstance(names, str)
                if not allow_empty:
                    assert len(names) > 0
            except Exception:
                raise ValueError(
                    '{}_vars should be a str or an array_like of str, instead '
                    'of {}'.format(tag, names))

            # different strategies to handle the case of repeated elements
            if len(names) != len(set(names)):
                if handle_repeat == 'remove':
                    # trigger warning and remove repeated elements
                    names = list(set(names))
                    warnings.warn('removing repeated elements found in '
                                  '{}_vars'.format(tag), RuntimeWarning)
                elif handle_repeat == 'ignore':
                    # just ignore and pass
                    pass
                elif handle_repeat == 'warn':
                    # trigger warning but do not remove
                    warnings.warn('repeated elements found in '
                                  '{}_vars'.format(tag), RuntimeWarning)
                elif handle_repeat == 'raise':
                    # raise an exception
                    raise ValueError(
                        'some elements in {}_vars are not unique.'.format(tag))
                else:
                    raise RuntimeError('unexpected value for handle_repeat.')
        return names

    @property
    def input_vars(self):
        return self._input_vars

    @input_vars.setter
    def input_vars(self, names):
        self._input_vars = PropertyList(
            names, lambda x: self._var_check(x, 'input', False, 'ignore'))

    @property
    def output_vars(self):
        return self._output_vars

    @output_vars.setter
    def output_vars(self, names):
        self._output_vars = PropertyList(
            names, lambda x: self._var_check(x, 'output', False, 'raise'))

    @property
    def delete_vars(self):
        return self._delete_vars

    @delete_vars.setter
    def delete_vars(self, names):
        self._delete_vars = PropertyList(
            names, lambda x: self._var_check(x, 'delete', True, 'remove'))

    def _concat_check(self, concat, tag):
        try:
            concat = np.asarray(concat, dtype=np.int)
            assert np.all(concat > 0) and concat.ndim == 1
        except Exception:
            raise ValueError(
                '{}_concat should be a bool or an 1-d array_like of '
                'int, instead of {}'.format(tag, concat))
        if tag == 'input':
            self._input_cum = np.cumsum(np.insert(concat, 0, 0))
        elif tag == 'output':
            self._output_cum = np.cumsum(np.insert(concat, 0, 0))
        else:
            raise RuntimeError('unexpected value {} for tag.'.format(tag))
        return concat

    @property
    def concat_input(self):
        return self._concat_input

    @concat_input.setter
    def concat_input(self, concat):
        if isinstance(concat, bool):
            self._concat_input = concat
            self._input_cum = None
        else:
            self._concat_input = PropertyList(
                concat, lambda x: self._concat_check(x, 'input'))

    @property
    def concat_output(self):
        return self._concat_output

    @concat_output.setter
    def concat_output(self, concat):
        if isinstance(concat, bool):
            self._concat_output = concat
            self._output_cum = None
        else:
            self._concat_output = PropertyList(
                concat, lambda x: self._concat_check(x, 'output'))

    def _scale_check(self, scales):
        try:
            scales = np.ascontiguousarray(scales)
            if scales.ndim == 1:
                scales = np.array((np.zeros_like(scales), scales)).T.copy()
            if not (scales.ndim == 2 and scales.shape[-1] == 2):
                raise ValueError('I do not know how to interpret the shape '
                                 'of input_scales.')
        except Exception:
            raise ValueError('Invalid value for input_scales.')
        self._input_scales_diff = scales[:, 1] - scales[:, 0]
        return scales

    @property
    def input_scales(self):
        return self._input_scales

    @input_scales.setter
    def input_scales(self, scales):
        if scales is None:
            self._input_scales = None
            self._input_scales_diff = 1.
        else:
            self._input_scales = self._scale_check(scales)
            # we do not allow directly modify the elements of input_scales here
            # as it cannot trigger the update of input_scales_diff
            self._input_scales.flags.writeable = False # TODO: PropertyArray?

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

    @staticmethod
    def _args_setter(args, tag):
        if args is None:
            return ()
        else:
            try:
                return tuple(args)
            except Exception:
                raise ValueError('{}_args should be a tuple, instead of '
                                 '{}.'.format(tag, args))

    @staticmethod
    def _kwargs_setter(kwargs, tag):
        if kwargs is None:
            return {}
        else:
            try:
                return dict(kwargs)
            except Exception:
                raise ValueError('{}_kwargs should be a dict, instead of '
                                 '{}.'.format(tag, kwargs))

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


class Module(ModuleBase):
    """
    Basic wrapper for use-definied functions.
    
    Parameters
    ----------
    fun : callable or None, optional
        Callable returning the value of function, or `None` if undefined.
    jac : callable or None, optional
        Callable returning the value of Jacobian, or `None` if undefined.
    fun_and_jac : callable or None, optional
        Callable returning the function and Jacobian at the same time, or `None`
        if undefined.
    input_vars : str or 1-d array_like of str, optional
        Name(s) of input variable(s). Set to `('__var__',)` by default.
    output_vars : str or 1-d array_like of str, optional
        Name(s) of output variable(s). Set to `('__var__',)` by default.
    delete_vars : str or 1-d array_like of str, optional
        Name(s) of variable(s) to be deleted from the dict during runtime. Set
        to `()` by default.
    concat_input : bool or 1-d array_like of positive int, optional
        Controlling the recombination of input variables. Set to `False` by
        default.
    concat_output : bool or 1-d array_like of positive int, optional
        Controlling the recombination of output variables. Set to `False` by
        default.
    input_scales : None or array_like, optional
        Controlling the scaling of input variables. Set to `None` by default.
    label : str, optional
        Label of Module used in the `print_summary` method.
    fun_args, jac_args, fun_and_jac_args : array_like, optional
        Additional arguments to be passed to `fun`, `jac` and `fun_and_jac`.
        Will be stored as tuples.
    fun_kwargs, jac_kwargs, fun_and_jac_kwargs : dict, optional
        Additional keyword arguments to be passed to `fun`, `jac` and
        `fun_and_jac`.
    """
    def __init__(self, fun=None, jac=None, fun_and_jac=None,
                 input_vars=('__var__',), output_vars=('__var__',),
                 delete_vars=(), concat_input=False, concat_output=False,
                 input_scales=None, label=None, fun_args=(), fun_kwargs=None,
                 jac_args=(), jac_kwargs=None, fun_and_jac_args=(),
                 fun_and_jac_kwargs=None):
        self.fun = fun
        self.jac = jac
        self.fun_and_jac = fun_and_jac
        super().__init__(input_vars, output_vars, delete_vars, concat_input,
                         concat_output, input_scales, label, fun_args,
                         fun_kwargs, jac_args, jac_kwargs, fun_and_jac_args,
                         fun_and_jac_kwargs)


SurrogateScope = namedtuple('SurrogateScope', ['i_step', 'n_step'])


class Surrogate(ModuleBase):
    """
    Base class for surrogate modules.
    
    Parameters
    ----------
    input_size : int or None, optional
        The size of input variables. If None, will be inferred from
        `concat_input`.
    output_size : int or None, optional
        The size of output variables. If None, will be inferred from
        `concat_output`.
    scope : array_like of 2 ints, optional
        Will be unpacked as `(i_step, n_step)`, where `i_step` represents the
        index where the true `Module` should start to be replaced by the
        `Surrogate`, and `n_step` represents the number of `Module`s to be
        replaced.
    fit_options : dict, optional
        Additional keyword arguments for fitting the surrogate model.
    kwargs : dict, optional
        Additional keyword arguments to be passed to `Module.__init__`.
    
    Notes
    -----
    Unlike `Module`, the default value of `concat_input` will be `True`.
    """
    def __init__(self, input_size=None, output_size=None, scope=(0, 1),
                 fit_options=None, **kwargs):
        self._initialized = False
        if not 'concat_input' in kwargs:
            kwargs['concat_input'] = True
        super().__init__(**kwargs)
        if input_size is None:
            try:
                assert not isinstance(self.concat_input, bool)
                input_size = int(np.sum(self.concat_input))
                assert input_size > 0
            except Exception:
                raise ValueError(
                    'failed to infer input_size from concat_input.')
        if output_size is None:
            try:
                assert not isinstance(self.concat_output, bool)
                output_size = int(np.sum(self.concat_output))
                assert output_size > 0
            except Exception:
                raise ValueError(
                    'failed to infer output_size from concat_output.')
        self.input_size = input_size
        self.output_size = output_size
        self.scope = scope
        self.fit_options = fit_options
        if not hasattr(self, '_fun'):
            self._fun = None
        if not hasattr(self, '_jac'):
            self._jac = None
        if not hasattr(self, '_fun_and_jac'):
            self._fun_and_jac = None
        self._initialized = True

    @property
    def scope(self):
        return self._scope

    @scope.setter
    def scope(self, s):
        try:
            i_step, n_step = s
            assert n_step > 0
            self._scope = SurrogateScope(int(i_step), int(n_step))
        except Exception:
            raise ValueError('invalid value for scope.')

    @property
    def fit_options(self):
        return self._fit_options

    @fit_options.setter
    def fit_options(self, options):
        if options is None:
            self._fit_options = {}
        else:
            self._fit_options = dict(options)

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, size):
        if self._initialized:
            raise RuntimeError(
                'input_size cannot be modified after initialization.')
        else:
            try:
                size = int(size)
                assert size > 0
            except Exception:
                raise ValueError('input_size should be a positive int.')
            self._input_size = size

    @property
    def output_size(self):
        return self._output_size

    @output_size.setter
    def output_size(self, size):
        if self._initialized:
            raise RuntimeError(
                'output_size cannot be modified after initialization.')
        else:
            try:
                size = int(size)
                assert size > 0
            except Exception:
                raise ValueError('output_size should be a positive int.')
            self._output_size = size

    def fit(self, *args, **kwargs):
        raise NotImplementedError('Abstract Method.')

    @property
    def n_param(self):
        raise NotImplementedError('Abstract Property.')
