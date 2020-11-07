
__all__ = ['Scoring']


class Scoring:

    def __init__(self):
        pass

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def _repr_title(self):
        return f'{self.name} scoring'

    def __repr__(self):
        return f'{self._repr_title}'
