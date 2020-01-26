from collections import OrderedDict
import numpy as np
import copy

__all__ = ['VariableDict', 'PropertyList']


class VariableDict:
    
    def __init__(self):
        self._fun = OrderedDict()
        self._jac = OrderedDict()
    
    def __getitem__(self, key):
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
            new_dict = VariableDict()
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
            return new_dict
        else:
            raise ValueError('key should be a str, or a list/tuple/np.ndarray '
                             'of str.')
    
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


class PropertyList:
    # https://stackoverflow.com/a/39190103/12292488
    def __init__(self, iterable=(), check=None):
        if isinstance(iterable, PropertyList):
            self._list = iterable._list.copy()
        else:
            self._list = iterable
        self._check = check
        if callable(self._check):
            self.check()
        elif self._check is None:
            self._list = list(self._list)
        else:
            raise ValueError('check should be callable or None.')
        self.append = self._wrapper(self._list.append)
        self.extend = self._wrapper(self._list.extend)
        self.insert = self._wrapper(self._list.insert)
        self.remove = self._wrapper(self._list.remove)
        self.pop = self._wrapper(self._list.pop)
        self.clear = self._wrapper(self._list.clear)
        self.index = self._list.index
        self.count = self._list.count
        self.sort = self._wrapper(self._list.sort)
        self.reverse = self._wrapper(self._list.reverse)
        self.copy = lambda: copy.copy(self)
    
    def __getitem__(self, key):
        return self._list.__getitem__(key)

    def __setitem__(self, key, item):
        self._list.__setitem__(key, item)
        self.check()

    def __delitem__(self, key):
        self._list.__delitem__(key)
        self.check()

    def __len__(self):
        return self._list.__len__()
    
    def __iter__(self):
        return self._list.__iter__()
    
    def __next__(self):
        return self._list.__next__()
    
    def check(self):
        if self._check is not None:
            self._list = self._check(self._list)
    
    def _wrapper(self, f):
        def _wrapped(*args, **kwargs):
            res = f(*args, **kwargs)
            self.check()
            return res
        return _wrapped
