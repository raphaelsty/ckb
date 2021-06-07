import numpy as np
import torch

__all__ = ["NegativeSampling"]


class NegativeSampling:
    """Generate negative sample to train models.

    Parameters
    ----------
        size (int): Number of false triplets per sample.
        train_triples (list): Training triples allowing to generate only false triples.
        entities (dict): Entities of the dataset.
        relations (dict): Relations of the dataset.

    Examples
    --------
    >>> from ckb import datasets
    >>> from ckb import sampling

    >>> import torch
    >>> _ = torch.manual_seed(42)

    >>> train = [
    ...     ("Le stratege", "is_available", "Netflix"),
    ...     ("The Imitation Game", "is_available", "Netflix"),
    ...     ("Star Wars", "is_available", "Disney"),
    ...     ("James Bond", "is_available", "Amazon"),
    ... ]

    >>> dataset = datasets.Dataset(
    ...    train = train,
    ...    batch_size = 2,
    ...    seed = 42,
    ...    shuffle = False,
    ... )

    >>> negative_sampling = sampling.NegativeSampling(
    ...    size = 5,
    ...    train_triples = dataset.train,
    ...    entities = dataset.entities,
    ...    relations = dataset.relations,
    ...    seed = 42,
    ... )

    >>> sample = torch.tensor([[0, 0, 4], [1, 0, 4]])

    >>> negative_sample = negative_sampling.generate(sample, mode='tail-batch')

    >>> negative_sample
    tensor([[6, 3, 6, 2, 6],
            [6, 3, 6, 2, 6]])

    >>> negative_sample = negative_sampling.generate(sample, mode='head-batch')

    >>> train = [
    ...     ("Le stratege", "is_available", "Netflix"),
    ...     ("Le stratege", "is_available", "Le stratege"),
    ... ]

    >>> dataset = datasets.Dataset(
    ...    train = train,
    ...    batch_size = 2,
    ...    seed = 42,
    ...    shuffle = False,
    ... )

    >>> negative_sampling = sampling.NegativeSampling(
    ...    size = 5,
    ...    train_triples = dataset.train,
    ...    entities = dataset.entities,
    ...    relations = dataset.relations,
    ...    seed = 42,
    ... )

    >>> sample = torch.tensor([[0, 0, 1], [0, 0, 0]])

    >>> negative_sample = negative_sampling.generate(sample, mode='tail-batch')

    >>> negative_sample
    tensor([[0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0]])


    References
    ----------
    [^1]: [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(self, size, train_triples, entities, relations, seed=42):
        """Generate negative samples.
        size (int): Batch size of the negative samples generated.
        train_triples (list[(int, int, int)]): Set of positive triples.
        entities (dict | list): Set of entities.
        relations (dict | list): Set of relations.
        seed (int): Random state.
        """
        self.size = size

        self.n_entity = len(entities)

        self.n_relation = len(relations)

        self.true_head, self.true_tail = self.get_true_head_and_tail(train_triples)

        self._rng = np.random.RandomState(seed)  # pylint: disable=no-member

    @classmethod
    def _filter_negative_sample(cls, negative_sample, record):
        mask = np.in1d(negative_sample, record, assume_unique=True, invert=True)
        return negative_sample[mask]

    def generate(self, sample, mode):
        """Generate negative samples from a head, relation tail
        If the mode is set to head-batch, this method will generate a tensor of fake heads.
        If the mode is set to tail-batch, this method will generate a tensor of fake tails.
        """
        samples = []

        negative_entities = self._rng.randint(self.n_entity, size=self.size + 500)

        for head, relation, tail in sample:

            head, relation, tail = head.item(), relation.item(), tail.item()

            negative_entities_sample = []

            size = 0
            step = 0

            while size < self.size:

                if mode == "head-batch":

                    negative_entities_filtered = self._filter_negative_sample(
                        negative_sample=negative_entities,
                        record=self.true_head[(relation, tail)],
                    )

                elif mode == "tail-batch":

                    negative_entities_filtered = self._filter_negative_sample(
                        negative_sample=negative_entities,
                        record=self.true_tail[(head, relation)],
                    )

                size += negative_entities_filtered.size
                negative_entities_sample.append(negative_entities_filtered)

                step += 1
                if step > 100:
                    size += negative_entities.size
                    negative_entities_sample.append(negative_entities)

            negative_entities_sample = np.concatenate(negative_entities_sample)[: self.size]

            negative_entities_sample = torch.LongTensor(negative_entities_sample)

            samples.append(negative_entities_sample)

        return torch.stack(samples, dim=0).long()

    @staticmethod
    def get_true_head_and_tail(triples):
        """Build a dictionary to filter out existing triples from fakes ones."""
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:

            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)

            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:  # pylint: disable=E1141
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))

        for head, relation in true_tail:  # pylint: disable=E1141
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail
