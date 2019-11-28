from distributed import Client, LocalCluster, get_client

__all__ = ['check_client']


def check_client(client):
    if isinstance(client, Client):
        return client, False
    else:
        if client is None:
            try:
                client = get_client()
                return client, False
            except:
                pass
        if client is not None:
            try:
                client = int(client)
                client = client if client > 0 else None
            except:
                raise ValueError(
                    'I do not know how to get a client from what you gave me.')
        cluster = LocalCluster(n_workers=client, threads_per_worker=1)
        client = Client(cluster)
        return client, True
