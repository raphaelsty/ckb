from mkb import sampling as mkb_sampling

__all__ = ["NegativeSampling"]


class NegativeSampling(mkb_sampling.NegativeSampling):
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

    >>> negative_sample
    tensor([[6, 2, 2, 4, 3],
            [6, 2, 2, 4, 3]])

    References
    ----------
    [^1]: [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

    """

    def __init__(self, size, train_triples, entities, relations, seed=42):
        super().__init__(
            size=size,
            train_triples=train_triples,
            entities=entities,
            relations=relations,
            seed=42,
        )
