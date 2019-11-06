import warnings
USE_SHAREDMEM = False
USE_MULTIPROCESSING = False

try:
    from sharedmem import MapReduce
    class ParallelMap(MapReduce):
        """Using sharedmem.MapReduce as ParallelMap.
        
        Original docstring as below:
        """
        def __init__(self, n_jobs=None):
            super().__init__(np=n_jobs)
        __doc__ += MapReduce.__doc__
    USE_SHAREDMEM = True
except:
    USE_SHAREDMEM = False

if not USE_SHAREDMEM:
    try:
        from multiprocessing.pool import Pool
        class ParallelMap(Pool):
            """Use multiprocessing.Pool as ParallelMap.
            
            Original docstring as below:
            
            Class which supports an async version of applying functions to arguments.
            """
            def __init__(self, n_jobs=None):
                super().__init__(processes=n_jobs)
            #__doc__ += Pool.__doc__
        USE_MULTIPROCESSING = True
    except:
        USE_MULTIPROCESSING = False

if (not USE_SHAREDMEM) and (not USE_MULTIPROCESSING):
    warnings.warn('no available parallelization backend found.', RuntimeWarning)
