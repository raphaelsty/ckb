__all__ = ["Pipeline"]


import collections

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
    <BLANKLINE>
     Epoch: 0, step 2.
         Validation:
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

                score = model(sample)

                if mode == "classification":

                    y = data["y"].to(self.device)

                    error = loss(score, y)

                else:

                    weight = data["weight"].to(self.device)

                    negative_sample = sampling.generate(
                        sample=sample,
                        mode=mode,
                    )

                    negative_sample = negative_sample.to(self.device)

                    negative_score = model(
                        sample=sample, negative_sample=negative_sample, mode=mode
                    )

                    error = loss(score, negative_score, weight)

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

                    self.evaluation_done = True

                    print(f"\n Epoch: {epoch}, step {self.step}.")

                    if dataset.valid:

                        self.valid_scores = evaluation.eval(model=model, dataset=dataset.valid)

                        self.valid_scores.update(
                            evaluation.eval_relations(model=model, dataset=dataset.valid)
                        )

                        self.print_metrics(description="Validation:", metrics=self.valid_scores)

                    if dataset.test:

                        self.test_scores = evaluation.eval(model=model, dataset=dataset.test)

                        self.test_scores.update(
                            evaluation.eval_relations(model=model, dataset=dataset.test)
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

                        self.print_metrics(description="Validation:", metrics=self.valid_scores)

                        self.print_metrics(description="Test:", metrics=self.test_scores)

                        return self

        if dataset.valid and not self.evaluation_done:

            self.valid_scores = evaluation.eval(model=model, dataset=dataset.valid)

            self.valid_scores.update(evaluation.eval_relations(model=model, dataset=dataset.valid))

            self.print_metrics(description="Validation:", metrics=self.valid_scores)

        if dataset.test and not self.evaluation_done:

            self.test_scores = evaluation.eval(model=model, dataset=dataset.test)

            self.test_scores.update(evaluation.eval_relations(model=model, dataset=dataset.test))

            self.print_metrics(description="Test:", metrics=self.test_scores)

        return self

    @classmethod
    def print_metrics(cls, description, metrics):
        print(f"\t {description}")
        for metric, value in metrics.items():
            print(f"\t\t {metric}: {value}")
