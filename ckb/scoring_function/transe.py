import torch

__all__ = ['TransE']


class TransE:

    def __init__(self):
        pass

    def __call__(self, head, relation, tail, gamma, mode):

        if mode == 'head-batch':

            score = head + (relation - tail)

        else:

            score = (head + relation) - tail

        return gamma.item() - torch.norm(score, p=1, dim=2)
