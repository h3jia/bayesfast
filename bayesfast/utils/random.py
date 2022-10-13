import jax
import os

__all__ = ['get_key', 'set_key', 'spawn_key']


_random_key = jax.random.PRNGKey(int.from_bytes(os.urandom(7), byteorder='little'))


def get_key(update_current=True):
    """
    Get the global random key.
    """
    key = _random_key
    if update_current:
        spawn_key(0, True)
    return key


def set_key(key):
    """
    Set the global random key.

    Parameters
    ----------
    key : PRNG Key or int
        Will first check whether it is already a valid jax PRNG key. If yes, it will be set as the
        global random key. Otherwise, it will be passed to ``jax.random.PRNGKey`` to initialize a
        new PRNG key.
    """
    global _random_key
    try:
        foo = jax.random.normal(key, (1,)) # just to try if this works
        _random_key = key
    except TypeError:
        _random_key = jax.random.PRNGKey(key)


def spawn_key(n, update_current=True):
    """
    Spawn an array of new keys from the current global random key.

    Parameters
    ----------
    n : int
        The number of new keys to spawn. Will spawn `n+1` new keys and return the last `n` keys. The
        first key will be (optionally) used to update the global key (see the ``update_current``
        argument below).
    update_current : bool, optional
        Whether to also update the current global random key. If True, will use the first spawned
        key as the new global random key. Set to ``True`` by default.
    """
    global _random_key
    try:
        n = int(n)
        assert n >= 0
    except Exception:
        raise ValueError('n should be a positive int.')
    spawned = jax.random.split(_random_key, n + 1)
    if update_current:
        _random_key = spawned[0]
    return spawned[1:]
