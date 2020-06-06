from distributed import Client, LocalCluster, get_client

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
                raise ValueError(
                    'I do not know how to get a client from what you gave me.')
        cluster = LocalCluster(n_workers=client, threads_per_worker=1)
        client = Client(cluster)
        return client, True
