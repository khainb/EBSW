import abc


class Saver(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "save_checkpoint")
            and callable(subclass.save_checkpoint)
            and hasattr(subclass, "save_best_weights")
            and callable(subclass.save_best_weights)
        )
