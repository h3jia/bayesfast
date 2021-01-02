try:
    from ray.util.multiprocessing import Pool as RayPool
    HAS_RAY = True
except Exception:
    HAS_RAY = False
try:
    from distributed import Client
    HAS_DASK = True
except Exception:
    HAS_DASK = False
try:
    from sharedmem import MapReduce
    HAS_SHAREDMEM = True
except Exception:
    HAS_SHAREDMEM = False
try:
    from loky import get_reusable_executor, reusable_executor
    HAS_LOKY = True
except Exception:
    HAS_LOKY = False
from multiprocess.pool import Pool
import warnings
# from copy import deepcopy
# we have to import Pool after Client to avoid some strange error

__all__ = ['ParallelBackend', 'get_backend', 'set_backend']

# TODO: maybe we should add a multiprocess-style chunksize for dask
#       since currently it's not good at handling a very large number of tasks
#       https://distributed.dask.org/en/latest/efficiency.html#use-larger-tasks
# TODO: should the default value be None or "multiprocess"?


class ParallelBackend:
    """
    The unified backend for parallelization.
    
    Currently, we support `multiprocess`, `dask`, `sharedmem` and `loky`.
    `multiprocess` usually has better performance on single-node machines, while
    `dask` can be used for multi-node parallelization. Note the following known
    issues: when used for sampling, (1) `dask` and `loky` do not respect the
    global bayesfast random seed; (2) `sharedmem` may not display the progress
    messages correctly (multiple messages in the same line); (3) `loky` does not
    print any messages at all in Jupyter. So we recommend using the default
    `multiprocess` backend when possible.
    
    Parameters
    ----------
    backend : None, int, Pool, Client or MapReduce, optional
        The backend for parallelization. If `None` or `int`, will be passed as
        the `processes` argument to initialize a Pool in a with context. Set to
        `None` by default.
    """
    def __new__(cls, backend=None):
        if isinstance(backend, ParallelBackend):
            return backend
        else:
            return super(ParallelBackend, cls).__new__(cls)

    def __init__(self, backend=None):
        if isinstance(backend, ParallelBackend):
            return
        self.backend = backend

    def __enter__(self):
        if self.backend is None or isinstance(self.backend, int):
            self._backend_activated = Pool(self.backend)
        elif HAS_SHAREDMEM and isinstance(self.backend, MapReduce):
            self.backend.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.backend is None or isinstance(self.backend, int):
            self._backend_activated.close()
            self._backend_activated.join()
            self._backend_activated = None
        elif HAS_SHAREDMEM and isinstance(self.backend, MapReduce):
            self.backend.__exit__(exc_type, exc_val, exc_tb)

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, be):
        if be is None or (isinstance(be, int) and be > 0):
            pass
        elif isinstance(be, Pool):
            pass
        elif HAS_RAY and isinstance(be, RayPool):
            pass
        elif HAS_DASK and isinstance(be, Client):
            pass
        elif HAS_SHAREDMEM and isinstance(be, MapReduce):
            pass
        elif HAS_LOKY and isinstance(be,
                                     reusable_executor._ReusablePoolExecutor):
            pass
        # elif be == 'serial':
        #     pass
        else:
            raise ValueError('invalid value for backend.')
        self._backend_activated = be
        self._backend = be

    @property
    def backend_activated(self):
        return self._backend_activated

    @property
    def kind(self):
        if self.backend is None or isinstance(self.backend, int):
            return 'multiprocess'
        elif isinstance(self.backend, Pool):
            return 'multiprocess'
        elif HAS_RAY and isinstance(self.backend, RayPool):
            return 'ray'
        elif HAS_DASK and isinstance(self.backend, Client):
            return 'dask'
        elif HAS_SHAREDMEM and isinstance(self.backend, MapReduce):
            return 'sharedmem'
        elif HAS_LOKY and isinstance(self.backend,
                                     reusable_executor._ReusablePoolExecutor):
            return 'loky'
        # elif self.backend == 'serial':
        #     return 'serial'
        else:
            raise RuntimeError('unexpected value for self.backend.')

    def map(self, fun, *iters):
        if self.backend_activated is None:
            raise RuntimeError('the backend is not activated. Please use it in '
                               'a with context.')
        elif isinstance(self.backend_activated, Pool):
            return self.backend_activated.starmap(fun, zip(*iters))
        elif HAS_RAY and isinstance(self.backend_activated, RayPool):
            return self.backend_activated.starmap(fun, list(zip(*iters)))
            # https://github.com/ray-project/ray/issues/11451
            # that's why I need to explicitly convert it to a list for now
        elif HAS_DASK and isinstance(self.backend_activated, Client):
            return self.gather(self.backend_activated.map(fun, *iters))
        elif HAS_SHAREDMEM and isinstance(self.backend_activated, MapReduce):
            return self.backend_activated.map(fun, list(zip(*iters)), star=True)
        elif HAS_LOKY and isinstance(self.backend_activated,
                                     reusable_executor._ReusablePoolExecutor):
            return self.gather(self.backend_activated.map(fun, *iters))
        # elif self.backend_activated == 'serial':
        #     return [deepcopy(fun)(*[i[j] for i in iters]) for j in range(l)]
        else:
            raise RuntimeError('unexpected value for self.backend_activated.')

    def map_async(self, fun, *iters):
        if self.backend_activated is None:
            raise RuntimeError('the backend is not activated. Please use it in '
                               'a with context.')
        elif isinstance(self.backend_activated, Pool):
            return self.backend_activated.starmap_async(fun, zip(*iters))
        elif HAS_RAY and isinstance(self.backend_activated, RayPool):
            return self.backend_activated.starmap_async(fun, list(zip(*iters)))
        elif HAS_DASK and isinstance(self.backend_activated, Client):
            return self.backend_activated.map(fun, *iters)
        elif HAS_SHAREDMEM and isinstance(self.backend_activated, MapReduce):
            warnings.warn('sharedmem does not support map_async. Using map '
                          'instead.', RuntimeWarning)
            return self.backend_activated.map(fun, list(zip(*iters)), star=True)
        elif HAS_LOKY and isinstance(self.backend_activated,
                                     reusable_executor._ReusablePoolExecutor):
            return self.backend_activated.map(fun, *iters)
        # elif self.backend_activated == 'serial':
        #     return self.map(fun, *iters)
        else:
            raise RuntimeError('unexpected value for self.backend_activated.')

    def gather(self, async_result):
        if self.backend_activated is None:
            raise RuntimeError('the backend is not activated. Please use it in '
                               'a with context.')
        elif isinstance(self.backend_activated, Pool):
            return async_result.get()
        elif isinstance(self.backend_activated, RayPool):
            return async_result.get()
        elif HAS_DASK and isinstance(self.backend_activated, Client):
            return self.backend_activated.gather(async_result)
        elif HAS_SHAREDMEM and isinstance(self.backend_activated, MapReduce):
            return async_result
        elif HAS_LOKY and isinstance(self.backend_activated,
                                     reusable_executor._ReusablePoolExecutor):
            return list(async_result)
        # elif self.backend_activated == 'serial':
        #     return async_result
        else:
            raise RuntimeError('unexpected value for self.backend_activated.')


_global_backend = ParallelBackend()


def get_backend():
    return _global_backend


def set_backend(backend):
    global _global_backend
    _global_backend = ParallelBackend(backend)
