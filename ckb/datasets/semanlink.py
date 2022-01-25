import json
import pathlib

from mkb import datasets as mkb_datasets

from ..utils import read_csv

__all__ = ["Semanlink"]


class Semanlink(mkb_datasets.Dataset):
    """Semanlink dataset.

    Train triplets gather entities created before 2019-06-01.
    Valid triplets gather entities created between 2019-06-01 and 2020-06-01.
    Test triplets gather entities created between 2020-06-01 and 2021-10-27.

    Parameters
    ----------
        batch_size (int): Size of the batch.
        use_labels (bool):
        shuffle (bool): Replaces the identifier of the entities with their textual label.
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

        >>> dataset = datasets.Semanlink(batch_size=1, pre_compute=False, shuffle=True, seed=42)

        >>> dataset
        Semanlink dataset
            Batch size  1
              Entities  32502
             Relations  40
               Shuffle  True
         Train triples  73828
         Validation triples  5035
         Test triples  6094

    """

    def __init__(
        self,
        batch_size,
        use_labels=True,
        shuffle=True,
        pre_compute=True,
        num_workers=1,
        seed=None,
    ):

        self.filename = "semanlink"

        path = pathlib.Path(__file__).parent.joinpath(self.filename)

        if use_labels:
            with open(f"{path}/labels.json", "r") as entities_labels:
                labels = json.load(entities_labels)

        train = read_csv(path=f"{path}/train.csv", sep="|")
        valid = read_csv(path=f"{path}/valid.csv", sep="|")
        test = read_csv(path=f"{path}/test.csv", sep="|")

        if use_labels:
            train = [(labels.get(h, h), r, labels.get(t, t)) for h, r, t in train]
            valid = [(labels.get(h, h), r, labels.get(t, t)) for h, r, t in valid]
            test = [(labels.get(h, h), r, labels.get(t, t)) for h, r, t in test]

        super().__init__(
            train=train,
            valid=valid,
            test=test,
            classification=False,
            pre_compute=pre_compute,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            seed=seed,
        )
