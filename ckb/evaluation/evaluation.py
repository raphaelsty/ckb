import collections
import copy

import torch
import tqdm
from creme import stats
from mkb import evaluation as mkb_evaluation
from mkb import models as mkb_models
from mkb import utils as mkb_utils
from torch.utils import data

from ..datasets import TestDataset

__all__ = ["Evaluation"]


class Evaluation(mkb_evaluation.Evaluation):
    """Wrapper for MKB evaluation module.

    Examples
    --------

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

    >>> model.entities
    {0: 'mkb', 1: 'github', 2: 'ckb', 3: 'library', 4: 'tool'}

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

    """

    def __init__(
        self,
        entities,
        relations,
        batch_size,
        true_triples=[],
        device="cuda",
        num_workers=0,
    ):

        super().__init__(
            entities=entities,
            relations=relations,
            batch_size=batch_size,
            true_triples=true_triples,
            device=device,
            num_workers=num_workers,
        )

        self.scoring = {
            "TransE": mkb_models.TransE,
            "DistMult": mkb_models.DistMult,
            "RotatE": mkb_models.RotatE,
            "pRotatE": mkb_models.pRotatE,
            "ComplEx": mkb_models.ComplEx,
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

        if model.scoring.name == "pRotatE":
            mkb_model.modulus.data = model.modulus.data

        return mkb_model.to(self.device)

    def detail_eval(self, model, dataset, threshold=1.5):
        """Divide input dataset relations into different categories (i.e. ONE-TO-ONE, ONE-TO-MANY,
        MANY-TO-ONE and MANY-TO-MANY) according to the mapping properties of relationships.

        Reference:
            1. [Bordes, Antoine, et al. "Translating embeddings for modeling multi-relational data." Advances in neural information processing systems. 2013.](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)
        """
        return super().detail_eval(
            model=self.initialize(model=model), dataset=dataset, threshold=threshold
        )

    def _get_test_loader(self, triples, true_triples, entities, relations, mode):
        test_dataset = TestDataset(
            triples=triples,
            true_triples=true_triples,
            entities=entities,
            relations=relations,
            mode=mode,
        )

        return data.DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=TestDataset.collate_fn,
        )

    def get_entity_stream(self, dataset):
        """Get stream dedicated to link prediction."""

        head_loader = self._get_test_loader(
            triples=dataset,
            true_triples=self.true_triples,
            entities=self.entities,
            relations=self.relations,
            mode="head-batch",
        )

        tail_loader = self._get_test_loader(
            triples=dataset,
            true_triples=self.true_triples,
            entities=self.entities,
            relations=self.relations,
            mode="tail-batch",
        )

        return [head_loader, tail_loader]

    def compute_score(self, model, test_set, metrics, device):

        training = False
        if model.training:
            model = model.eval()
            training = True

        bar = mkb_utils.Bar(dataset=test_set, update_every=1)
        bar.set_description("Evaluation")

        for data in bar:

            sample = data["sample"].to(device)
            negative_sample = data["negative_sample"].to(device)
            filter_bias = data["filter_bias"].to(device)
            mode = data["mode"]

            if mode == "head-batch" or mode == "tail-batch":

                score = model(sample=sample, negative_sample=negative_sample, mode=mode)

            elif mode == "relation-batch":

                score = model(negative_sample)

            score += filter_bias

            argsort = torch.argsort(score, dim=1, descending=True)

            if mode == "head-batch":
                positive_arg = sample[:, 0]

            if mode == "relation-batch":
                positive_arg = sample[:, 1]

            elif mode == "tail-batch":
                positive_arg = sample[:, 2]

            batch_size = sample.size(0)

            for i in range(batch_size):
                # Notice that argsort is not ranking
                ranking = torch.nonzero(argsort[i, :] == positive_arg[i])[:1]
                assert ranking.size(0) == 1

                ranking = 1 + ranking.item()

                # ranking + 1 is the true ranking used in evaluation metrics
                metrics["MRR"].update(1.0 / ranking)

                metrics["MR"].update(ranking)

                metrics["HITS@1"].update(1.0 if ranking <= 1 else 0.0)

                metrics["HITS@3"].update(1.0 if ranking <= 3 else 0.0)

                metrics["HITS@10"].update(1.0 if ranking <= 10 else 0.0)

        if training:
            model = model.train()

        return metrics

    def compute_detailled_score(self, model, test_set, metrics, types_relations, device):

        training = False
        if model.training:
            model = model.eval()
            training = True

        bar = mkb_utils.Bar(dataset=test_set, update_every=1)
        bar.set_description("Evaluation")

        for data in bar:

            sample = data["sample"].to(device)
            negative_sample = data["negative_sample"].to(device)
            filter_bias = data["filter_bias"].to(device)
            mode = data["mode"]

            score = model(
                sample=sample,
                negative_sample=negative_sample,
                mode=mode,
            )

            score += filter_bias

            argsort = torch.argsort(score, dim=1, descending=True)

            if mode == "head-batch":
                positive_arg = sample[:, 0]

            elif mode == "tail-batch":
                positive_arg = sample[:, 2]

            batch_size = sample.size(0)

            for i in range(batch_size):
                # Notice that argsort is not ranking
                ranking = torch.nonzero(argsort[i, :] == positive_arg[i])[:1]
                assert ranking.size(0) == 1

                ranking = 1 + ranking.item()

                type_relation = types_relations[sample[:, 1][i].item()]

                # ranking + 1 is the true ranking used in evaluation metrics
                metrics[mode][type_relation]["MRR"].update(1.0 / ranking)

                metrics[mode][type_relation]["MR"].update(ranking)

                metrics[mode][type_relation]["HITS@1"].update(1.0 if ranking <= 1 else 0.0)

                metrics[mode][type_relation]["HITS@3"].update(1.0 if ranking <= 3 else 0.0)

                metrics[mode][type_relation]["HITS@10"].update(1.0 if ranking <= 10 else 0.0)

        if training:
            model = model.train()

        return metrics
