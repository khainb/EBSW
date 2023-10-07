import abc


class NoiseAdd_er(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "add_noise") and callable(subclass.add_noise)
