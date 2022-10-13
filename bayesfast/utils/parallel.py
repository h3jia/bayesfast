from multiprocess.pool import Pool
from threadpoolctl import threadpool_limits
import warnings


__all__ = ['ParallelPool', 'get_pool', 'set_pool']


class ParallelPool:
    """
    The global pool for parallelization.

    This is used outside jax, e.g. to evaluate the true forward models. Note that jax evaluations
    won't observe the n_thread limit here, which is a
    `known issue <https://github.com/joblib/threadpoolctl/issues/127>`_.

    We use ``multiprocess`` for multi-processing, and ``threadpoolctl`` for multi-threading. Note
    that while the previous version of BayesFast also supports other backends like ``ray``,
    ``dask``, ``sharedmem`` and ``loky``, we now only keep ``multiprocess`` as it generally works
    better with single node setup. Please let us know if you would to have the other option(s) added
    back.

    Parameters
    ----------
    n_process : int, optional
        The number of processes to use. Set to ``4`` by default.
    n_thread : int, optional
        The number of threads per process. Set to ``1`` by default.
    """
    def __init__(self, n_process=4, n_thread=1):
        self.n_process = n_process
        self.n_thread = n_thread
        self._pool = None

    @property
    def n_process(self):
        """
        The number of processes to use.
        """
        return self._n_process

    @n_process.setter
    def n_process(self, n):
        try:
            n = int(n)
            assert n > 0
        except Exception:
            raise ValueError('n_process should be a positive int.')
        self._n_process = n

    @property
    def n_thread(self):
        """
        The number of threads per process.
        """
        return self._n_thread

    @n_thread.setter
    def n_thread(self, n):
        try:
            n = int(n)
            assert n > 0
        except Exception:
            raise ValueError('n_thread should be a positive int.')
        self._n_thread = n

    @property
    def pool(self):
        """
        The underlying multiprocess Pool object. Will be None unless activated in a with context.
        """
        return self._pool

    def __enter__(self):
        if self.n_process > 1:
            self._pool = Pool(self.n_process)
        # self._threadpool_limits = threadpool_limits(self.n_thread)
        # self._threadpool_limits._set_threadpool_limits()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.n_process > 1:
            self._pool.close()
            self._pool.join()
        self._pool = None
        # self._threadpool_limits.restore_original_limits()

    # @staticmethod
    # def _threadpool_wrapper(fun, n_thread):
    #     def f_wrapped(*args, **kwargs):
    #         with threadpool_limits(n_thread):
    #             return fun(*args, **kwargs)
    #     return f_wrapped

    @staticmethod
    def _set_thread(n_thread):
        from threadpoolctl import threadpool_limits
        threadpool_limits(n_thread)

    def _map(self, fun, *iters):
        if self.pool is None:
            if self.n_process == 1:
                return self.gather(self.map_async(fun, *iters))
            else:
                raise RuntimeError('the pool is not activated. Please use it in a with context.')
        elif isinstance(self.pool, Pool):
            # return self.pool.starmap(self._threadpool_wrapper(fun, self.n_thread), zip(*iters))
            return self.pool.starmap(fun, zip(*iters))
        else:
            raise RuntimeError('unexpected value for self.pool.')

    def map(self, fun, *iters):
        self._map(self._set_thread, [self.n_thread] * self.n_process)
        return self._map(fun, *iters)

    def _map_async(self, fun, *iters):
        if self.pool is None:
            if self.n_process == 1:
                return map(self._threadpool_wrapper(fun, self.n_thread), *iters)
            else:
                raise RuntimeError('the pool is not activated. Please use it in a with context.')
        elif isinstance(self.pool, Pool):
            return self.pool.starmap_async(fun, zip(*iters))
            # return self.pool.starmap_async(self._threadpool_wrapper(fun, self.n_thread),
            #                                zip(*iters))
        else:
            raise RuntimeError('unexpected value for self.pool.')

    def map_async(self, fun, *iters):
        self._map_async(self._set_thread, [self.n_thread] * self.n_process)
        return self._map_async(fun, *iters)

    def gather(self, async_result):
        if self.pool is None:
            if self.n_process == 1:
                return list(async_result)
            else:
                raise RuntimeError('the pool is not activated. Please use it in a with context.')
        elif isinstance(self.pool, Pool):
            return async_result.get()
        else:
            raise RuntimeError('unexpected value for self.pool.')


_global_pool = ParallelPool()


def get_pool():
    """
    Get the global parallel pool.
    """
    return _global_pool


def set_pool(n_process=4, n_thread=1):
    """
    Set the global parallel pool.

    Parameters
    ----------
    n_process : int, optional
        The number of processes to use. Set to ``4`` by default.
    n_thread : int, optional
        The number of threads per process. Set to ``1`` by default.
    """
    global _global_pool
    _global_pool = ParallelPool(n_process, n_thread)
