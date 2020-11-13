import pandas as pd

__all__ = ['read_csv']


def read_csv(path, sep, header=None):
    return list(
        pd.read_csv(path, sep=sep, header=None).drop_duplicates(keep='first').itertuples(
            index=False, name=None))
