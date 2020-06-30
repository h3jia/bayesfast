import bayesfast as bf

# TODO: finish this


be = ParallelBackend(4)

def fun_0(i):
    return i

def fun_1(a, b):
    return a + b

with be as pool:
    print(pool.map(fun_0, range(4)))

with be as pool:
    print(pool.map(fun_1, range(4), range(4)))

with be as pool:
    print(pool.gather(pool.map_async(fun_0, range(4))))

with be as pool:
    print(pool.gather(pool.map_async(fun_1, range(4), range(4))))
