
__all__ = ['DistMult']


class DistMult:
    """DistMult scoring function.

    Example:

        >>> from ckb import models
        >>> from ckb import datasets
        >>> from ckb import scoring

        >>> import torch

        >>> _ = torch.manual_seed(42)

        >>> dataset = datasets.Semanlink(1)

        >>> model = models.DistillBert(
        ...    entities = dataset.entities,
        ...    relations = dataset.relations,
        ...    gamma = 9,
        ...    device = 'cpu',
        ...    scoring = scoring.DistMult(),
        ... )

        >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
        >>> model(sample)
        tensor([[-0.3350],
                [-0.8084]], grad_fn=<ViewBackward>)

        >>> sample = torch.tensor([[0, 0, 1], [2, 2, 1]])
        >>> model(sample)
        tensor([[-0.3135],
                [-0.5852]], grad_fn=<ViewBackward>)

        >>> sample = torch.tensor([[1, 0, 0], [1, 2, 2]])
        >>> model(sample)
        tensor([[-0.3135],
                [-0.5852]], grad_fn=<ViewBackward>)

        >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
        >>> negative_sample = torch.tensor([[1, 0], [1, 2]])

        >>> model(sample, negative_sample, mode='head-batch')
        tensor([[-0.3135, -0.3350],
                [-0.5852, -0.8084]], grad_fn=<ViewBackward>)

        >>> model(sample, negative_sample, mode='tail-batch')
        tensor([[-0.3135, -0.3350],
                [-0.5852, -0.8084]], grad_fn=<ViewBackward>)

    """

    def __init__(self):
        pass

    def __call__(self, head, relation, tail, mode, **kwargs):

        if mode == 'head-batch':

            score = head * (relation * tail)

        else:

            score = (head * relation) * tail

        return score.sum(dim=2)
