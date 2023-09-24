import abc

import six


@six.add_metaclass(abc.ABCMeta)
class Dataset:  # (metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "__len__")
            and callable(subclass.__len__)
            and hasattr(subclass, "__getitem__")
            and callable(subclass.__getitem__)
        )
