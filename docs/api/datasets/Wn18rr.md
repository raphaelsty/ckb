# Wn18rr

Wn18rr dataset.



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

>>> dataset = datasets.Wn18rr(batch_size=1, pre_compute=True, shuffle=True, seed=42)

>>> dataset
Wn18rr dataset
    Batch size  1
    Entities  40943
    Relations  11
    Shuffle  True
    Train triples  86835
    Validation triples  3034
    Test triples  3134
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

[^1]: [Liang Yao, Chengsheng Mao, and Yuan Luo. 2019. Kg- bert: Bert for knowledge graph completion. arXiv preprint arXiv:1909.03193.](https://arxiv.org/abs/1909.03193)

