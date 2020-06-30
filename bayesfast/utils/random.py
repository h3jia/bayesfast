import numpy as np

__all__ = ['get_generator', 'set_generator', 'spawn_generator']

# TODO: review the jump_current option of spawn_generator


_random_generator = np.random.default_rng()


def get_generator():
    return _random_generator


def set_generator(generator):
    global _random_generator
    _random_generator = np.random.default_rng(generator)


def spawn_generator(current_generator, n, jump_current=True):
    if not isinstance(current_generator, np.random.Generator):
        raise ValueError('current_generator should be a np.random.Generator.')
    try:
        n = int(n)
        assert n > 0
    except:
        raise ValueError('n should be a positive int.')
    spawned = [np.random.default_rng(current_generator.bit_generator.jumped(i))
               for i in range(1, n + 1)]
    if jump_current:
        current_generator.normal()
    return spawned
