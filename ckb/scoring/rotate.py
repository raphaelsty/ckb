from .base import Scoring

from math import pi

import torch

__all__ = ["RotatE"]


class RotatE(Scoring):
    """RotatE scoring function.

    Examples
    --------

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
    ...    scoring = scoring.RotatE(),
    ... )

    >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
    >>> model(sample)
    tensor([[-186.5064],
            [-153.2208]], grad_fn=<ViewBackward>)

    >>> sample = torch.tensor([[0, 0, 1], [2, 2, 1]])
    >>> model(sample)
    tensor([[-203.6809],
            [-191.3758]], grad_fn=<ViewBackward>)

    >>> sample = torch.tensor([[1, 0, 0], [1, 2, 2]])
    >>> model(sample)
    tensor([[-204.0743],
            [-192.8306]], grad_fn=<ViewBackward>)

    >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
    >>> negative_sample = torch.tensor([[1, 0], [1, 2]])

    >>> model(sample, negative_sample, mode='head-batch')
    tensor([[-204.0743, -186.5064],
            [-192.8306, -153.2208]], grad_fn=<ViewBackward>)

    >>> model(sample, negative_sample, mode='tail-batch')
    tensor([[-203.6809, -186.5064],
            [-191.3758, -153.2208]], grad_fn=<ViewBackward>)

    """

    def __init__(self):
        super().__init__()
        self.pi = pi

    def __call__(self, head, relation, tail, gamma, embedding_range, mode, **kwargs):
        """Compute the score of given facts (heads, relations, tails).

        Parameters
        ----------
            head: Embeddings of heads.
            relation: Embeddings of relations.
            tail: Embeddings of tails.
            gamma: Constant integer to stretch the embeddings.
            embedding_range: Range of the embeddings.
            mode: head-batch or tail-batch.

        """
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = relation / (embedding_range.item() / self.pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == "head-batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head

        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        score = gamma.item() - score.sum(dim=2)

        return score
