import os
import pathlib

import pandas as pd

from mkb import datasets as mkb_datasets

from ..utils import read_csv


__all__ = ["Semanlink"]


class Semanlink(mkb_datasets.Dataset):
    """Semanlink dataset.

    Parameters
    ----------
        batch_size (int): Size of the batch.
        shuffle (bool): Whether to shuffle the dataset or not.
        pre_compute (bool): Pre-compute parameters such as weights when using translationnal model
            (TransE, DistMult, RotatE, pRotatE, ComplEx).
        num_workers (int): Number of workers dedicated to iterate on the dataset.
        seed (int): Random state.

    Attributes
    ----------
        train (list): Training set.
        valid (list): Validation set.
        test (list): Testing set.
        entities (dict): Index of entities.
        relations (dict): Index of relations.
        n_entity (int): Number of entities.
        n_relation (int): Number of relations.

    Examples
    --------

        >>> from ckb import datasets

        >>> dataset = datasets.Semanlink(batch_size=1, pre_compute=True, shuffle=True, seed=42)

        >>> dataset
        Semanlink dataset
            Batch size  1
            Entities  5454
            Relations  4
            Shuffle  True
            Train triples  6422
            Validation triples  803
            Test triples  803

    """

    def __init__(
        self, batch_size, shuffle=True, pre_compute=True, num_workers=1, seed=None
    ):

        self.filename = "semanlink"

        path = pathlib.Path(__file__).parent.joinpath(self.filename)

        super().__init__(
            train=read_csv(path=f"{path}/train.csv", sep="|"),
            valid=read_csv(path=f"{path}/valid.csv", sep="|"),
            test=read_csv(path=f"{path}/test.csv", sep="|"),
            classification=False,
            pre_compute=pre_compute,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            seed=seed,
        )
