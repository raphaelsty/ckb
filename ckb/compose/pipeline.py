__all__ = ["Pipeline"]


import collections

import torch
from creme import stats
from mkb import utils


class Pipeline:
    """Pipeline dedicated to automate training model.

    Parameters
    ----------
        dataset (ckb.datasets): Dataset.
        model (ckb.models): Model.
        sampling (ckb.sampling): Negative sampling method.
        epochs (int): Number of epochs to train the model.
        validation (ckb.evaluation): Validation process.
        eval_every (int): When eval_every is set to 1, the model will be evaluated at every epochs.
        early_stopping_rounds (int): Stops training when model did not improve scores during
            `early_stopping_rounds` epochs.
        device (str): Device.

    Examples
    --------

    >>> from ckb import compose
    >>> from ckb import datasets
    >>> from ckb import evaluation
    >>> from ckb import losses
    >>> from ckb import models
    >>> from ckb import sampling
    >>> from ckb import scoring

    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> device = 'cpu'

    >>> train = [('mkb', 'is_a', 'library')]
    >>> valid = [('ckb', 'is_a', 'library'), ('github', 'is_a', 'tool')]
    >>> test = [('mkb', 'is_a', 'tool'), ('ckb', 'is_a', 'tool')]

    >>> dataset = datasets.Dataset(
    ...     batch_size = 1,
    ...     train = train,
    ...     valid = valid,
    ...     test = test,
    ...     seed = 42,
    ... )

    >>> model = models.DistillBert(
    ...     entities = dataset.entities,
    ...     relations = dataset.relations,
    ...     gamma = 9,
    ...     scoring = scoring.TransE(),
    ...     device = device,
    ... )

    >>> model = model.to(device)

    >>> optimizer = torch.optim.Adam(
    ...     filter(lambda p: p.requires_grad, model.parameters()),
    ...     lr = 0.00005,
    ... )

    >>> evaluation = evaluation.Evaluation(
    ...     entities = dataset.entities,
    ...     relations = dataset.relations,
    ...     true_triples = dataset.train + dataset.valid + dataset.test,
    ...     batch_size = 1,
    ...     device = device,
    ... )

    >>> sampling = sampling.NegativeSampling(
    ...     size = 1,
    ...     entities = dataset.entities,
    ...     relations = dataset.relations,
    ...     train_triples = dataset.train,
    ... )

    >>> pipeline = compose.Pipeline(
    ...     epochs = 1,
    ...     eval_every = 1,
    ...     early_stopping_rounds = 1,
    ...     device = device,
    ... )

    >>> pipeline = pipeline.learn(
    ...     model      = model,
    ...     dataset    = dataset,
    ...     evaluation = evaluation,
    ...     sampling   = sampling,
    ...     optimizer  = optimizer,
    ...     loss       = losses.Adversarial(alpha=0.5),
    ... )
    Validation:
            MRR: 0.3958
            MR: 2.75
            HITS@1: 0.0
            HITS@3: 0.75
            HITS@10: 1.0
            MRR_relations: 1.0
            MR_relations: 1.0
            HITS@1_relations: 1.0
            HITS@3_relations: 1.0
            HITS@10_relations: 1.0
    Test:
        MRR: 0.375
        MR: 2.75
        HITS@1: 0.0
        HITS@3: 1.0
        HITS@10: 1.0
        MRR_relations: 1.0
        MR_relations: 1.0
        HITS@1_relations: 1.0
        HITS@3_relations: 1.0
        HITS@10_relations: 1.0

    """

    def __init__(self, epochs, eval_every=2000, early_stopping_rounds=3, device="cpu"):
        self.epochs = epochs
        self.eval_every = eval_every
        self.early_stopping_rounds = early_stopping_rounds
        self.device = device

        self.metric_loss = stats.RollingMean(1000)

        self.round_without_improvement_valid = 0
        self.round_without_improvement_test = 0

        self.history_valid = collections.defaultdict(float)
        self.history_test = collections.defaultdict(float)

        self.valid_scores = {}
        self.test_scores = {}

        self.evaluation_done = False
        self.step = 0

    def learn(self, model, dataset, sampling, optimizer, loss, evaluation=None):

        for epoch in range(self.epochs):

            bar = utils.Bar(dataset=dataset, update_every=10)

            for data in bar:

                sample = data["sample"].to(self.device)
                mode = data["mode"]
                weight = data["weight"].to(self.device)

                if mode == "tail-batch":
                    continue

                triples = []
                for h, r, t in sample:
                    h, r, t = h.item(), r.item(), t.item()
                    triples.append((h, r, t))

                negative = self.in_batch_negative_triples(triples, sampling)

                if not negative[0]:
                    continue

                e_encode = []

                mapping_heads = {}
                mapping_tails = {}

                for index, (h, r, t) in enumerate(triples):
                    e_encode.append(self.entities[h])
                    e_encode.append(self.entities[t])
                    mapping_heads[h] = index
                    mapping_tails[t] = index

                embeddings = model.encoder(e_encode)

                heads = torch.stack(
                    [e for index, e in enumerate(embeddings) if index % 2 == 0], dim=0
                ).unsqueeze(1)
                tails = torch.stack(
                    [e for index, e in enumerate(embeddings) if index % 2 != 0], dim=0
                ).unsqueeze(1)

                relations = torch.index_select(
                    self.relation_embedding, dim=0, index=sample[:, 1]
                ).unsqueeze(1)

                score = model.scoring(
                    head=heads.to(self.device),
                    relation=relations.to(self.device),
                    tail=tails.to(self.device),
                    mode=mode,
                    gamma=self.gamma,
                )

                negative_scores = []
                for index, negative_sample in enumerate(negative):

                    tensor_h = []
                    tensor_r = []
                    tensor_t = []

                    for h, r, t in negative_sample:
                        tensor_h.append(heads[mapping_heads[h]])
                        tensor_r.append(relations[index])
                        tensor_t.append(tails[mapping_tails[t]])

                    tensor_h = torch.stack(tensor_h, dim=0)
                    tensor_r = torch.stack(tensor_r, dim=0)
                    tensor_t = torch.stack(tensor_t, dim=0)

                    negative_scores.append(
                        model.scoring(
                            head=tensor_h.to(self.device),
                            relation=tensor_r.to(self.device),
                            tail=tensor_t.to(self.device),
                            mode=mode,
                            gamma=self.gamma,
                        ).T
                    )

                negative_scores = torch.stack(negative_scores, dim=1).squeeze(0)

                error = loss(score, negative_scores, weight)

                error.backward()

                _ = optimizer.step()

                optimizer.zero_grad()

                self.metric_loss.update(error.item())

                bar.set_description(f"Epoch: {epoch}, loss: {self.metric_loss.get():4f}")

                # Avoid doing evaluation twice for the same parameters.
                self.evaluation_done = False
                self.step += 1

                if evaluation is not None and not self.evaluation_done:

                    if (self.step + 1) % self.eval_every == 0:

                        update_embeddings = True
                        self.evaluation_done = True

                        print(f"\n Epoch: {epoch}, step {self.step}.")

                        if dataset.valid:

                            self.valid_scores = evaluation.eval(
                                model=model,
                                dataset=dataset.valid,
                                update_embeddings=update_embeddings,
                            )

                            update_embeddings = False

                            self.valid_scores.update(
                                evaluation.eval_relations(
                                    model=model,
                                    dataset=dataset.valid,
                                    update_embeddings=update_embeddings,
                                )
                            )

                            self.print_metrics(
                                description="Validation:", metrics=self.valid_scores
                            )

                        if dataset.test:

                            self.test_scores = evaluation.eval(
                                model=model,
                                dataset=dataset.test,
                                update_embeddings=update_embeddings,
                            )

                            update_embeddings = False

                            self.test_scores.update(
                                evaluation.eval_relations(
                                    model=model,
                                    dataset=dataset.test,
                                    update_embeddings=update_embeddings,
                                )
                            )

                            self.print_metrics(description="Test:", metrics=self.test_scores)

                            if (
                                self.history_test["HITS@3"] > self.test_scores["HITS@3"]
                                and self.history_test["HITS@1"] > self.test_scores["HITS@1"]
                            ):
                                self.round_without_improvement_test += 1
                            else:
                                self.round_without_improvement_test = 0
                                self.history_test = self.test_scores
                        else:
                            if (
                                self.history_valid["HITS@3"] > self.valid_scores["HITS@3"]
                                and self.history_valid["HITS@1"] > self.valid_scores["HITS@1"]
                            ):
                                self.round_without_improvement_valid += 1
                            else:
                                self.round_without_improvement_valid = 0
                                self.history_valid = self.valid_scores

                        if (
                            self.round_without_improvement_valid == self.early_stopping_rounds
                            or self.round_without_improvement_test == self.early_stopping_rounds
                        ):

                            print(f"\n Early stopping at epoch {epoch}, step {self.step}.")

                            return self

        update_embeddings = True

        if dataset.valid and not self.evaluation_done:

            self.valid_scores = evaluation.eval(
                model=model, dataset=dataset.valid, update_embeddings=update_embeddings
            )

            update_embeddings = False

            self.valid_scores.update(evaluation.eval_relations(model=model, dataset=dataset.valid))

            self.print_metrics(description="Validation:", metrics=self.valid_scores)

        if dataset.test and not self.evaluation_done:

            self.test_scores = evaluation.eval(
                model=model, dataset=dataset.test, update_embeddings=update_embeddings
            )

            update_embeddings = False

            self.test_scores.update(evaluation.eval_relations(model=model, dataset=dataset.test))

            self.print_metrics(description="Test:", metrics=self.test_scores)

        return self

    @classmethod
    def print_metrics(cls, description, metrics):
        print(f"\t {description}")
        for metric, value in metrics.items():
            print(f"\t\t {metric}: {value}")

    @staticmethod
    def in_batch_negative_triples(triples, sampling):
        """Generate in batch negative triples. All input sample will have the same number of fake triples."""
        negative = []
        for index_head, (h, r, _) in enumerate(triples):
            fake = []
            for index_tail, (_, _, t) in enumerate(triples):

                if index_head == index_tail:
                    continue

                if t not in sampling.true_tail[(h, r)]:
                    fake.append((h, r, t))

            negative.append(fake)

        for index_tail, (_, r, t) in enumerate(triples):
            for index_head, (h, _, _) in enumerate(triples):

                if index_head == index_tail:
                    continue

                if h not in sampling.true_head[(r, t)]:
                    negative[index_tail].append((h, r, t))

        min_length = min(map(len, negative))
        return [x[: min(sampling.size, min_length)] for x in negative]
