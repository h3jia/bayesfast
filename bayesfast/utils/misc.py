__all__ = ['all_isinstance']

def all_isinstance(iterable, class_or_tuple):
    return (hasattr(iterable, '__iter__') and 
            all(isinstance(i, class_or_tuple) for i in iterable))
