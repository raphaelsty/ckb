# Pipeline

Pipeline dedicated to automate training model.



## Parameters

- **epochs**

- **eval_every** – defaults to `1`

- **early_stopping_rounds** – defaults to `3`

- **device** – defaults to `cuda`



## Examples

```python
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
Epoch: 0.
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
<BLANKLINE>
Epoch: 0.
<BLANKLINE>
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
```

## Methods

???- note "learn"

???- note "print_metrics"

