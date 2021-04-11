from .base import Scoring

import torch

__all__ = ["TransE"]


class TransE(Scoring):
    """TransE scoring function.

    Examples
    --------

    >>> from ckb import scoring

    >>> scoring.TransE()
    TransE scoring

    """

    def __init__(self):
        super().__init__()

    def __call__(self, head, relation, tail, gamma, mode, **kwargs):
        """Compute the score of given facts (heads, relations, tails).

        Parameters
        ----------
            head: Embeddings of heads.
            relation: Embeddings of relations.
            tail: Embeddings of tails.
            mode: head-batch or tail-batch.

        """
        if mode == "head-batch":

            score = head + (relation - tail)

        else:

            score = (head + relation) - tail

        return gamma.item() - torch.norm(score, p=1, dim=2)
