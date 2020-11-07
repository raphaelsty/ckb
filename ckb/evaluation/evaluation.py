from mkb import evaluation as mkb_evaluation
from mkb import models as mkb_models
from creme import stats

import tqdm

import collections

import torch


__all__ = ['Evaluation']


class Evaluation(mkb_evaluation.Evaluation):
    """Wrapper for MKB evaluation module.

    Example:

        >>> from mkb import datasets
        >>> from ckb import evaluation
        >>> from ckb import models
        >>> from ckb import scoring

        >>> import torch

        >>> _ = torch.manual_seed(42)

        >>> train = [('mkb', 'is_a', 'library'), ('github', 'is_a', 'tool')]
        >>> valid = [('ckb', 'is_a', 'library'), ('github', 'is_a', 'tool')]
        >>> test = [('mkb', 'is_a', 'tool'), ('ckb', 'is_a', 'tool')]

        >>> dataset = datasets.Dataset(
        ...     batch_size = 1,
        ...     train = train,
        ...     valid = valid,
        ...     test = test,
        ...     seed = 42,
        ... )

        >>> dataset
        Dataset dataset
            Batch size         1
            Entities           5
            Relations          1
            Shuffle            True
            Train triples      2
            Validation triples 2
            Test triples       2

        >>> model = models.DistillBert(
        ...     entities = dataset.entities,
        ...     relations = dataset.relations,
        ...     gamma = 9,
        ...     scoring = scoring.TransE(),
        ...     device = 'cpu',
        ... )

        >>> model
        DistillBert model
            Entities embeddings dim  768
            Relations embeddings dim 768
            Gamma                    9.0
            Number of entities       5
            Number of relations      1

        >>> validation = evaluation.Evaluation(
        ...     entities = dataset.entities,
        ...     relations = dataset.relations,
        ...     true_triples = dataset.train + dataset.valid + dataset.test,
        ...     batch_size = 1,
        ...     device = 'cpu',
        ... )

        >>> validation.eval(model = model, dataset = dataset.valid)
        {'MRR': 0.3958, 'MR': 2.75, 'HITS@1': 0.0, 'HITS@3': 0.75, 'HITS@10': 1.0}

        >>> validation.eval_relations(model = model, dataset = dataset.valid)
        {'MRR_relations': 1.0, 'MR_relations': 1.0, 'HITS@1_relations': 1.0, 'HITS@3_relations': 1.0, 'HITS@10_relations': 1.0}

        >>> validation.detail_eval(model = model, dataset = dataset.valid, threshold = 1.5)
                  head                               tail                             metadata
                  MRR   MR HITS@1 HITS@3 HITS@10     MRR   MR HITS@1 HITS@3 HITS@10 frequency
        relation
        1_1       0.000  0.0    0.0    0.0     0.0  0.0000  0.0    0.0    0.0     0.0       0.0
        1_M       0.000  0.0    0.0    0.0     0.0  0.0000  0.0    0.0    0.0     0.0       0.0
        M_1       0.000  0.0    0.0    0.0     0.0  0.0000  0.0    0.0    0.0     0.0       0.0
        M_M       0.375  3.0    0.0    0.5     1.0  0.4167  2.5    0.0    1.0     1.0       1.0

    """

    def __init__(self, entities, relations, batch_size, true_triples=[], device='cuda', num_workers=1):

        super().__init__(
            entities=entities,
            relations=relations,
            batch_size=batch_size,
            true_triples=true_triples,
            device=device,
            num_workers=num_workers,
        )

        self.scoring = {
            'TransE': mkb_models.TransE,
            'DistMult': mkb_models.DistMult,
            'RotatE': mkb_models.RotatE,
            'pRotatE': mkb_models.pRotatE,
            'ComplEx': mkb_models.ComplEx,
        }

    def eval(self, model, dataset):
        """Evaluate selected model with the metrics: MRR, MR, HITS@1, HITS@3, HITS@10"""
        return super().eval(model=self.initialize(model=model), dataset=dataset)

    def eval_relations(self, model, dataset):
        """Evaluate selected model with the metrics: MRR, MR, HITS@1, HITS@3, HITS@10"""
        return super().eval_relations(model=self.initialize(model=model), dataset=dataset)

    def initialize(self, model):
        """Initialize model for evaluation"""
        embeddings = []
        with torch.no_grad():
            for _, e in tqdm.tqdm(model.entities.items(), position=0):
                embeddings.append(model.encoder([e]))
        embeddings = torch.cat(embeddings)

        mkb_model = self.scoring[model.scoring.name](
            entities={v: k for k, v in model.entities.items()},
            relations={v: k for k, v in model.relations.items()},
            gamma=model.gamma,
            hidden_dim=model.hidden_dim,
        )

        mkb_model.entity_embedding.data = embeddings.data
        mkb_model.relation_embedding.data = model.relation_embedding.data

        if model.scoring.name == 'pRotatE':
            mkb_model.modulus.data = model.modulus.data

        return mkb_model

    def detail_eval(self, model, dataset, threshold=1.5):
        """Divide input dataset relations into different categories (i.e. ONE-TO-ONE, ONE-TO-MANY,
        MANY-TO-ONE and MANY-TO-MANY) according to the mapping properties of relationships.

        Reference:
            1. [Bordes, Antoine, et al. "Translating embeddings for modeling multi-relational data." Advances in neural information processing systems. 2013.](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)
        """
        return super().detail_eval(
            model=self.initialize(model=model),
            dataset=dataset,
            threshold=threshold
        )
