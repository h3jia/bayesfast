__all__ = ['all_isinstance']

def all_isinstance(iterable, class_or_tuple):
    return all(isinstance(i, class_or_tuple) for i in iterable)
