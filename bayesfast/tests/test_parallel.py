import bayesfast as bf
import numpy as np


def f(x):
    for _ in range(500):
        foo = np.einsum('ij,ij->', x, x)
    return foo


def test_parallel():
    # pytest will not automatically check if it's actually using the desired number of threads
    # for that, you may want to manually monitor the CPU usage
    m = 4
    n = 100
    x = np.ones((m, n, n))

    bf.parallel.set_pool(4, 1)
    with bf.parallel.get_pool() as pool:
        a = pool.map(f, x)
    assert np.array_equal(a, np.full((m,), n**2))

    bf.parallel.set_pool(1, 4)
    with bf.parallel.get_pool() as pool:
        b = pool.map(f, x)
    assert np.array_equal(b, np.full((m,), n**2))
