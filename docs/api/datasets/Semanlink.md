# Semanlink

Semanlink dataset.



## Parameters

- **batch_size**

- **shuffle** – defaults to `True`

- **pre_compute** – defaults to `True`

- **num_workers** – defaults to `1`

- **seed** – defaults to `None`


## Attributes

- **train (list): Training set.**

    valid (list): Validation set. test (list): Testing set. entities (dict): Index of entities. relations (dict): Index of relations. n_entity (int): Number of entities. n_relation (int): Number of relations.


## Examples

```python
>>> from ckb import datasets

>>> dataset = datasets.Semanlink(batch_size=1, pre_compute=True, shuffle=True, seed=42)

>>> dataset
Semanlink dataset
    Batch size  1
    Entities  5454
    Relations  4
    Shuffle  True
    Train triples  6422
    Validation triples  803
    Test triples  803
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

