from math import pi

import torch

__all__ = ['ComplEx']


class ComplEx:
    """ComplEx scoring function.

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
        ...    scoring = scoring.ComplEx(),
        ... )

        >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
        >>> model(sample)
        tensor([[0.8402],
                [0.4317]], grad_fn=<ViewBackward>)

        >>> sample = torch.tensor([[0, 0, 1], [2, 2, 1]])
        >>> model(sample)
        tensor([[0.5372],
                [0.1728]], grad_fn=<ViewBackward>)

        >>> sample = torch.tensor([[1, 0, 0], [1, 2, 2]])
        >>> model(sample)
        tensor([[0.5762],
                [0.3085]], grad_fn=<ViewBackward>)

        >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
        >>> negative_sample = torch.tensor([[1, 0], [1, 2]])

        >>> model(sample, negative_sample, mode='head-batch')
        tensor([[0.5762, 0.8402],
                [0.3085, 0.4317]], grad_fn=<ViewBackward>)

        >>> model(sample, negative_sample, mode='tail-batch')
        tensor([[0.5372, 0.8402],
                [0.1728, 0.4317]], grad_fn=<ViewBackward>)

    """

    def __init__(self):
        pass

    def __call__(self, head, relation, tail, mode,  **kwargs):

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score

        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        return score.sum(dim=2)
