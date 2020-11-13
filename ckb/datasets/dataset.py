from mkb import datasets as mkb_datasets

from torch.utils import data

import torch


class Dataset(mkb_datasets.Dataset):

    def __init__(self, train, batch_size, entities=None, relations=None, valid=None, test=None,
                 shuffle=True, pre_compute=True, num_workers=1, seed=None):

        super().__init__(train=train, batch_size=batch_size, entities=entities, relations=relations,
                         valid=valid, test=test, shuffle=shuffle, pre_compute=pre_compute,
                         num_workers=num_workers, seed=seed)

    def test_dataset(self, entities, relations, batch_size):
        return self.test_stream(triples=self.test, entities=entities, relations=relations,
                                batch_size=batch_size)

    def validation_dataset(self, entities, relations, batch_size):
        return self.test_stream(triples=self.valid, entities=entities, relations=relations,
                                batch_size=batch_size)

    def test_stream(self, triples, batch_size, entities, relations):
        head_loader = self._get_test_loader(
            triples=triples, entities=entities, relations=relations, batch_size=batch_size,
            mode='head-batch')

        tail_loader = self._get_test_loader(
            triples=triples, entities=entities, relations=relations, batch_size=batch_size,
            mode='tail-batch')

        return [head_loader, tail_loader]

    def _get_test_loader(self, triples, entities, relations, batch_size, mode):
        """Initialize test dataset loader."""
        test_dataset = TestDataset(
            triples=triples, true_triples=self.train + self.test + self.valid,
            entities=entities, relations=relations, mode=mode)

        return data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, num_workers=self.num_workers,
            collate_fn=TestDataset.collate_fn)


class TestDataset(mkb_datasets.base.TestDataset):
    """Test dataset loader dedicated to link prediction.
    Parameters:
        triples (list): Testing set.
        true_triples (list): Triples to filter when validating the model.
        entities (dict): Index of entities.
        relations (dict): Index of relations.
        mode (str): head-batch or tail-batch.
    Attributes:
        n_entity (int): Number of entities.
        n_relation (int): Number of relations.
        len (int): Number of training triplets.
    References:
        1. [Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space." arXiv preprint arXiv:1902.10197 (2019).](https://arxiv.org/pdf/1902.10197.pdf)
        2. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
    """

    def __init__(self, triples, true_triples, entities, relations, mode, entities_to_drop):
        super().__init__(triples=triples, true_triples=true_triples,
                         entities=entities, relations=relations, mode=mode)

        self.entities_to_drop = entities_to_drop

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        tmp = []

        if self.mode == 'head-batch':

            for rand_head in range(self.n_entity):

                # Exlude entities from candidates such as artificial entities created by the user.
                if rand_head in self.entities_to_drop:
                    tmp.append((-1e5, head))

                # Candidate answer:
                elif (rand_head, relation, tail) not in self.true_triples:
                    tmp.append((0, rand_head))

                # Actual target
                elif rand_head == head:
                    tmp.append((0, head))

                # Actual true triple that we filter out:
                else:
                    tmp.append((-1e5, head))

        elif self.mode == 'tail-batch':

            for rand_tail in range(self.n_entity):

                # Exlude entities from candidates such as artificial entities created by the user.
                if rand_tail in self.entities_to_drop:
                    tmp.append((-1e5, tail))

                # Candidate answer:
                elif (head, relation, rand_tail) not in self.true_triples:
                    tmp.append((0, rand_tail))

                # Actual target
                elif rand_tail == tail:
                    tmp.append((0, tail))

                # Actual true triple that we filter out:
                else:
                    tmp.append((-1e5, tail))

        tmp = torch.LongTensor(tmp)

        filter_bias = tmp[:, 0].float()

        negative_sample = tmp[:, 1]

        sample = torch.LongTensor((head, relation, tail))

        return sample, negative_sample, filter_bias, self.mode
