from collections.abc import MutableSequence

__all__ = ['PropertyList']


class PropertyList(MutableSequence):

    def __init__(self, iterable=()):
        raise NotImplementedError
        self._list = list(iterable)

    def __getitem__(self, key):
        return self._list.__getitem__(key)

    def __setitem__(self, key, item):
        self._list.__setitem__(key, item)
        # trigger change handler

    def __delitem__(self, key):
        self._list.__delitem__(key)
        # trigger change handler

    def __len__(self):
        return self._list.__len__()

    def insert(self, index, item):
        self._list.insert(index, item)
        # trigger change handler
