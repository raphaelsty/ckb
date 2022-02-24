import mkb.models as mkb_models
import torch
import torch.nn as nn

from ..scoring import RotatE

__all__ = ["BaseModel"]


class BaseModel(mkb_models.base.BaseModel):
    """Base model class.

    Examples
    --------

    >>> from ckb import models
    >>> from ckb import scoring
    >>> from mkb import datasets as mkb_datasets

    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> dataset = mkb_datasets.CountriesS1(1)

    >>> model = models.BaseModel(
    ...     entities = dataset.entities,
    ...     relations=dataset.relations,
    ...     hidden_dim=3,
    ...     gamma=3,
    ...     scoring=scoring.TransE(),
    ... )

    >>> sample = torch.tensor([[3, 0, 4], [5, 1, 6]])

    >>> head, relation, tail, shape = model.batch(sample)

    >>> head
    ['belize', 'falkland_islands']

    >>> tail
    ['morocco', 'saint_vincent_and_the_grenadines']

    """

    def __init__(self, entities, relations, hidden_dim, scoring, gamma):

        relation_dim = hidden_dim
        entity_dim = hidden_dim

        if isinstance(scoring, RotatE):
            relation_dim = relation_dim // 2

        super().__init__(
            entities=entities,
            relations=relations,
            hidden_dim=hidden_dim,
            entity_dim=entity_dim,
            relation_dim=relation_dim,
            gamma=gamma,
        )

        self.scoring = scoring

        self.entities = {i: e for e, i in entities.items()}
        self.relations = {i: r for r, i in relations.items()}

        self.n_entity = len(entities)
        self.n_relation = len(relations)
        self.hidden_dim = hidden_dim
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)

        self.epsilon = 2

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.hidden_dim]),
            requires_grad=False,
        )

        self.relation_embedding = nn.Parameter(torch.zeros(self.n_relation, self.relation_dim))

        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )

        self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

    @property
    def twin(self):
        return False

    def forward(self, sample, negative_sample=None, mode=None):
        """Compute scores of input sample, negative sample with respect to the mode."""

        head, relation, tail, shape = self.encode(
            sample=sample, negative_sample=negative_sample, mode=mode
        )

        score = self.scoring(
            **{
                "head": head,
                "relation": relation,
                "tail": tail,
                "gamma": self.gamma,
                "mode": mode,
                "embedding_range": self.embedding_range,
                "modulus": self.modulus,
            }
        )

        return score.view(shape)

    def encode(self, sample, negative_sample=None, mode=None):
        """Encode input sample, negative sample with respect to the mode."""

        head, relation, tail, shape = self.batch(
            sample=sample, negative_sample=negative_sample, mode=mode
        )

        if negative_sample is None:

            head = self.encoder(e=head, mode="head").unsqueeze(1)
            tail = self.encoder(e=tail, mode="tail").unsqueeze(1)

        else:

            head, tail = self.negative_encoding(
                sample=sample,
                head=head,
                tail=tail,
                negative_sample=negative_sample,
                mode=mode,
            )

        return head, relation, tail, shape

    def batch(self, sample, negative_sample=None, mode=None):
        """Process input sample."""
        sample, shape = self.format_sample(sample=sample, negative_sample=negative_sample)

        relation = torch.index_select(
            self.relation_embedding, dim=0, index=sample[:, 1]
        ).unsqueeze(1)

        head = sample[:, 0]
        tail = sample[:, 2]

        head = [self.entities[h.item()] for h in head]
        tail = [self.entities[t.item()] for t in tail]

        return head, relation, tail, shape

    def negative_encoding(self, sample, head, tail, negative_sample, mode):

        mode_encoder = "head" if mode == "head-batch" else "tail"

        negative_sample = torch.stack(
            [
                self.encoder([self.entities[e.item()] for e in ns], mode=mode_encoder)
                for ns in negative_sample
            ]
        )

        if mode == "head-batch":

            head = negative_sample

            tail = self.encoder(e=tail, mode="tail").unsqueeze(1)

        elif mode == "tail-batch":

            tail = negative_sample

            head = self.encoder(e=head, mode="head").unsqueeze(1)

        return head, tail

    def encoder(self, e):
        """Encoder should be defined in the children class."""
        return torch.zeros(len(e))
