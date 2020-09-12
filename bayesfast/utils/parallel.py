try:
    from distributed import Client
    HAS_DASK = True
except Exception:
    HAS_DASK = False
from multiprocess.pool import Pool
# we have to import Pool after Client to avoid some strange error

__all__ = ['ParallelBackend', 'get_backend', 'set_backend']

# TODO: maybe we should add a multiprocess-style chunksize for dask
#       since currently it's not good at handling a very large number of tasks
#       https://distributed.dask.org/en/latest/efficiency.html#use-larger-tasks
# TODO: should the default value be None or "multiprocess"?


class ParallelBackend:
    """
    The unified backend for parallelization.
    
    Currently, we support `multiprocess.Pool` and `distributed.Client`. The
    former has better performance on single-node machines, while the latter
    can be used for multi-node parallelization.
    
    Parameters
    ----------
    backend : None, int, Pool or Client, optional
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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.backend is None or isinstance(self.backend, int):
            self._backend_activated.close()
            self._backend_activated.join()
            self._backend_activated = None

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, be):
        if be is None or (isinstance(be, int) and be > 0):
            self._backend_activated = be
        elif isinstance(be, Pool):
            self._backend_activated = be
        elif HAS_DASK and isinstance(be, Client):
            self._backend_activated = be
        else:
            raise ValueError('invalid value for backend.')
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
        elif HAS_DASK and isinstance(self.backend, Client):
            return 'dask'
        else:
            raise RuntimeError('unexpected value for self.backend.')

    def map(self, fun, *iters):
        if self.backend_activated is None:
            raise RuntimeError('the backend is not activated. Please use it in '
                               'a with context.')
        elif isinstance(self.backend_activated, Pool):
            return self.backend_activated.starmap(fun, zip(*iters))
        elif HAS_DASK and isinstance(self.backend_activated, Client):
            return self.backend_activated.gather(
                self.backend_activated.map(fun, *iters))
        else:
            raise RuntimeError('unexpected value for self.backend_activated.')

    def map_async(self, fun, *iters):
        if self.backend_activated is None:
            raise RuntimeError('the backend is not activated. Please use it in '
                               'a with context.')
        elif isinstance(self.backend_activated, Pool):
            return self.backend_activated.starmap_async(fun, zip(*iters))
        elif HAS_DASK and isinstance(self.backend_activated, Client):
            return self.backend_activated.map(fun, *iters)
        else:
            raise RuntimeError('unexpected value for self.backend_activated.')

    def gather(self, async_result):
        if self.backend_activated is None:
            raise RuntimeError('the backend is not activated. Please use it in '
                               'a with context.')
        elif isinstance(self.backend_activated, Pool):
            return async_result.get()
        elif HAS_DASK and isinstance(self.backend_activated, Client):
            return self.backend_activated.gather(async_result)
        else:
            raise RuntimeError('unexpected value for self.backend_activated.')


_global_backend = ParallelBackend()


def get_backend():
    return _global_backend


def set_backend(backend):
    global _global_backend
    _global_backend = ParallelBackend(backend)
