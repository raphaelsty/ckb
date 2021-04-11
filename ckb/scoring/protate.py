from .base import Scoring

from math import pi

import torch

__all__ = ["pRotatE"]


class pRotatE(Scoring):
    """pRotatE scoring function.

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
    ...    scoring = scoring.pRotatE(),
    ... )

    >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
    >>> model(sample)
    tensor([[5.4199],
            [5.4798]], grad_fn=<ViewBackward>)

    >>> sample = torch.tensor([[0, 0, 1], [2, 2, 1]])
    >>> model(sample)
    tensor([[5.4521],
            [5.5248]], grad_fn=<ViewBackward>)

    >>> sample = torch.tensor([[1, 0, 0], [1, 2, 2]])
    >>> model(sample)
    tensor([[5.4962],
            [5.5059]], grad_fn=<ViewBackward>)

    >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
    >>> negative_sample = torch.tensor([[1, 0], [1, 2]])

    >>> model(sample, negative_sample, mode='head-batch')
    tensor([[5.4962, 5.4199],
            [5.5059, 5.4798]], grad_fn=<ViewBackward>)

    >>> model(sample, negative_sample, mode='tail-batch')
    tensor([[5.4521, 5.4199],
            [5.5248, 5.4798]], grad_fn=<ViewBackward>)

    """

    def __init__(self):
        super().__init__()
        self.pi = pi

    def __call__(
        self, head, relation, tail, gamma, embedding_range, modulus, mode, **kwargs
    ):
        """Compute the score of given facts (heads, relations, tails).

        Parameters
        ----------
            head: Embeddings of heads.
            relation: Embeddings of relations.
            tail: Embeddings of tails.
            gamma: Constant integer to stretch the embeddings.
            embedding_range: Range of the embeddings.
            modulus: Constant to multiply the score.
            mode: head-batch or tail-batch.

        """

        phase_head = head / (embedding_range.item() / self.pi)
        phase_relation = relation / (embedding_range.item() / self.pi)
        phase_tail = tail / (embedding_range.item() / self.pi)

        if mode == "head-batch":
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = gamma.item() - score.sum(dim=2) * modulus

        return score
