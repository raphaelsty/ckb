from mkb import datasets as mkb_datasets


class Dataset(mkb_datasets.Dataset):

    def __init__(self, train, batch_size, entities=None, relations=None, valid=None, test=None,
                 shuffle=True, pre_compute=True, num_workers=1, seed=None):

        super().__init__(train=train, batch_size=batch_size, entities=entities, relations=relations,
                         valid=valid, test=test, shuffle=shuffle, pre_compute=pre_compute,
                         num_workers=num_workers, seed=seed)
