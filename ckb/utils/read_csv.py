import pandas as pd

__all__ = ["read_csv"]


def read_csv(path, sep, header=None):
    """Read a csv files of triplets and convert it to list of triplets.

    Parameters
    ----------
        sep (str): Separator used in the csv file.

    """
    return list(
        pd.read_csv(path, sep=sep, header=None)
        .drop_duplicates(keep="first")
        .itertuples(index=False, name=None)
    )
