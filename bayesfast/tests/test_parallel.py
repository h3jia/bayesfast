import bayesfast as bf

bf.utils.parallel.set_backend(4)
be = bf.utils.parallel.get_backend()
fun_0 = lambda i: i
fun_1 = lambda a, b: a+ b


def test_parallel_m_0():
    with be as pool:
        res = pool.map(fun_0, range(4))
        assert res == list(range(4))


def test_parallel_m_1():
    with be as pool:
        res = pool.map(fun_1, range(4), range(4))
        assert res == list(range(0, 8, 2))


def test_parallel_gma_0():
    with be as pool:
        res = pool.gather(pool.map_async(fun_0, range(4)))
        assert res == list(range(4))


def test_parallel_gma_1():
    with be as pool:
        res = pool.gather(pool.map_async(fun_1, range(4), range(4)))
        assert res == list(range(0, 8, 2))
