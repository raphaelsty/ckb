import torch

import torch.nn as nn

import mkb.models as mkb_models


__all__ = ['BaseModel']


class BaseModel(mkb_models.base.BaseModel):
    """Base model class.

    # TODO: ADD UNITS TESTS

    """

    def __init__(self, entities, relations, hidden_dim, entity_dim, relation_dim, gamma):

        super().__init__(
            entities=entities, relations=relations, hidden_dim=hidden_dim, entity_dim=entity_dim,
            relation_dim=relation_dim, gamma=gamma,
        )

        self.entities = {i: e for e, i in entities.items()}
        self.relations = {i: r for r, i in relations.items()}

        self.n_entity = len(entities)
        self.n_relation = len(relations)
        self.hidden_dim = hidden_dim
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.epsilon = 2

        self.embedding_range = nn.Parameter(
            torch.Tensor(
                [(self.gamma.item() + self.epsilon) / self.hidden_dim]),
            requires_grad=False
        )

        self.relation_embedding = nn.Parameter(
            torch.zeros(self.n_relation, self.relation_dim))

        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def encode(self, sample, negative_sample=None, mode=None):
        """Encode input sample, negative sample with respect to the mode."""

        head, relation, tail, shape = self.batch(
            sample=sample,
            negative_sample=negative_sample,
            mode=mode
        )

        if negative_sample is None:
            head = self.encoder(e=head).unsqueeze(1)
            tail = self.encoder(e=tail).unsqueeze(1)
        else:
            head, tail = self.negative_encoding(
                sample=sample, head=head, tail=tail, negative_sample=negative_sample, mode=mode)

        return head, relation, tail, shape

    def batch(self, sample, negative_sample=None, mode=None):

        sample, shape = self.format_sample(
            sample=sample,
            negative_sample=negative_sample
        )

        relation = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=sample[:, 1]
        ).unsqueeze(1)

        head = sample[:, 0]
        tail = sample[:, 2]

        head = [self.entities[h.item()] for h in head]
        tail = [self.entities[t.item()] for t in tail]

        return head, relation, tail, shape

    def negative_encoding(self, sample, head, tail, negative_sample, mode):

        negative_sample = [self.entities[e.item()] for e in negative_sample[0]]

        negative_sample = self.encoder(e=negative_sample)

        if mode == 'head-batch':

            head = torch.stack(
                [negative_sample for _ in range(sample.shape[0])])

            tail = self.encoder(e=tail).unsqueeze(1)

        elif mode == 'tail-batch':

            tail = torch.stack(
                [negative_sample for _ in range(sample.shape[0])])

            head = self.encoder(e=head).unsqueeze(1)

        return head, tail

    def encoder(self):
        """Encoder should be defined in the children class."""
        pass
