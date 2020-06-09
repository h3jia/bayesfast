from distributed import Client, LocalCluster, get_client
import warnings

__all__ = ['check_client']

# TODO: add support for traditional Pool?


def check_client(client):
    if isinstance(client, Client):
        return client, False
    else:
        if client is None:
            # try to get the existing client; will create a new one if failed
            try:
                client = get_client()
                return client, False
            except:
                pass
        else:
            try:
                client = int(client)
                assert client > 0
            except:
                warnings.warn('I do not know how to get a client from what you '
                              'gave me. Falling back to client=None for now.',
                              RuntimeWarning)
                client = None
        cluster = LocalCluster(n_workers=client, threads_per_worker=1)
        client = Client(cluster)
        return client, True
