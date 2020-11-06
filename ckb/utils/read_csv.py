import pandas as pd

__all__ = ['read_csv']

def read_csv(path, sep, header=None):
    return list(
        pd.read_csv(path, sep = sep, header = None).itertuples(
            index=False, name=None))