import numpy as np

__all__ = ['vectorize']


def vectorize(f, level=1):
    level = int(level)
    if level == 1:
        def fv(x):
            x = np.atleast_1d(x)
            if x.ndim == 1:
                return f(x)
            elif x.ndim == 2:
                return np.array([f(xi) for xi in x])
            elif x.ndim == 3:
                return np.array([[f(xj) for xj in xi] for xi in x])
            else:
                raise NotImplementedError(
                    'vectorizing for input > 3d is not implemented yet.')
    elif level == 2:
        def fv(x):
            x = np.atleast_1d(x)
            if x.ndim == 1 or x.ndim == 2:
                return f(x)
            elif x.ndim == 3:
                return f(x.reshape((-1, x.shape[-1]))).reshape(x.shape[:-1])
            else:
                raise NotImplementedError(
                    'vectorizing for input > 3d is not implemented yet.')
    else:
        raise ValueError('level should be 1 or 2.')
    return fv
