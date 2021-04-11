# Evaluation

Wrapper for MKB evaluation module.



## Parameters

- **entities**

- **relations**

- **batch_size**

- **true_triples** – defaults to `[]`

- **device** – defaults to `cuda`

- **num_workers** – defaults to `1`

- **entities_to_drop** – defaults to `[]`

- **same_entities** – defaults to `{}`



## Examples

```python
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
```

## Methods

???- note "compute_detailled_score"

???- note "compute_score"

???- note "detail_eval"

    Divide input dataset relations into different categories (i.e. ONE-TO-ONE, ONE-TO-MANY, MANY-TO-ONE and MANY-TO-MANY) according to the mapping properties of relationships.

    Reference:     1. [Bordes, Antoine, et al. "Translating embeddings for modeling multi-relational data." Advances in neural information processing systems. 2013.](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)

    **Parameters**

    - **model**    
    - **dataset**    
    - **threshold**     – defaults to `1.5`    
    
???- note "eval"

    Evaluate selected model with the metrics: MRR, MR, HITS@1, HITS@3, HITS@10

    **Parameters**

    - **model**    
    - **dataset**    
    
???- note "eval_relations"

    Evaluate selected model with the metrics: MRR, MR, HITS@1, HITS@3, HITS@10

    **Parameters**

    - **model**    
    - **dataset**    
    
???- note "get_entity_stream"

    Get stream dedicated to link prediction.

    **Parameters**

    - **dataset**    
    
???- note "get_relation_stream"

    Get stream dedicated to relation prediction.

    **Parameters**

    - **dataset**    
    
???- note "initialize"

    Initialize model for evaluation

    **Parameters**

    - **model**    
    
???- note "solve_same_entities"

    Replace artificial entities by the target. Some description may be dedicated to the same entities.

    **Parameters**

    - **argsort**    
    
???- note "types_relations"

    Divide input dataset relations into different categories (i.e. ONE-TO-ONE, ONE-TO-MANY, MANY-TO-ONE and MANY-TO-MANY) according to the mapping properties of relationships.

    **Parameters**

    - **model**    
    - **dataset**    
    - **threshold**     – defaults to `1.5`    
    
