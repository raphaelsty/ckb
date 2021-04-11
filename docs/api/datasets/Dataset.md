# Dataset

Custom dataset creation

The Dataset class allows to iterate on the data of a dataset. Dataset takes entities as input, relations, training data and optional validation and test data. Training data, validation and testing must be organized in the form of a triplet list. Entities and relations must be in a dictionary where the key is the label of the entity or relationship and the value must be the index of the entity / relation.

## Parameters

- **train**

- **batch_size**

- **entities** – defaults to `None`

- **relations** – defaults to `None`

- **valid** – defaults to `None`

- **test** – defaults to `None`

- **shuffle** – defaults to `True`

- **pre_compute** – defaults to `True`

- **num_workers** – defaults to `1`

- **seed** – defaults to `None`


## Attributes

- **n_entity (int): Number of entities.**

    n_relation (int): Number of relations.


## Examples

```python
>>> from ckb import datasets

>>> train = [
...    ('🐝', 'is', 'animal'),
...    ('🐻', 'is', 'animal'),
...    ('🐍', 'is', 'animal'),
...    ('🦔', 'is', 'animal'),
...    ('🦓', 'is', 'animal'),
...    ('🦒', 'is', 'animal'),
...    ('🦘', 'is', 'animal'),
...    ('🦝', 'is', 'animal'),
...    ('🦞', 'is', 'animal'),
...    ('🦢', 'is', 'animal'),
... ]

>>> test = [
...    ('🐝', 'is', 'animal'),
...    ('🐻', 'is', 'animal'),
...    ('🐍', 'is', 'animal'),
...    ('🦔', 'is', 'animal'),
...    ('🦓', 'is', 'animal'),
...    ('🦒', 'is', 'animal'),
...    ('🦘', 'is', 'animal'),
...    ('🦝', 'is', 'animal'),
...    ('🦞', 'is', 'animal'),
...    ('🦢', 'is', 'animal'),
... ]

>>> dataset = datasets.Dataset(train=train, test=test, batch_size=2, seed=42, shuffle=False)

>>> dataset
Dataset dataset
    Batch size  2
    Entities  11
    Relations  1
    Shuffle  False
    Train triples  10
    Validation triples  0
    Test triples  10

>>> dataset.entities
{'🐝': 0, '🐻': 1, '🐍': 2, '🦔': 3, '🦓': 4, '🦒': 5, '🦘': 6, '🦝': 7, '🦞': 8, '🦢': 9, 'animal': 10}
```

## Methods

???- note "fetch"

???- note "get_train_loader"

    Initialize train dataset loader.

    **Parameters**

    - **mode**    
    
???- note "mapping_entities"

    Construct mapping entities.

    
???- note "mapping_relations"

    Construct mapping relations.

    
???- note "test_dataset"

???- note "test_stream"

???- note "validation_dataset"

## References

[^1]: [Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space." arXiv preprint arXiv:1902.10197 (2019).](https://arxiv.org/pdf/1902.10197.pdf)
[^2]: [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

