# NegativeSampling

Generate negative sample to train models.



## Parameters

- **size**

- **train_triples**

- **entities**

- **relations**

- **seed** – defaults to `42`



## Examples

```python
>>> from ckb import datasets
>>> from ckb import sampling

>>> import torch
>>> _ = torch.manual_seed(42)

>>> train = [
...     ("Le stratege", "is_available", "Netflix"),
...     ("The Imitation Game", "is_available", "Netflix"),
...     ("Star Wars", "is_available", "Disney"),
...     ("James Bond", "is_available", "Amazon"),
... ]

>>> dataset = datasets.Dataset(
...    train = train,
...    batch_size = 2,
...    seed = 42,
...    shuffle = False,
... )

>>> negative_sampling = sampling.NegativeSampling(
...    size = 5,
...    train_triples = dataset.train,
...    entities = dataset.entities,
...    relations = dataset.relations,
...    seed = 42,
... )

>>> sample = torch.tensor([[0, 0, 4], [1, 0, 4]])

>>> negative_sample = negative_sampling.generate(sample, mode='tail-batch')

>>> negative_sample
tensor([[6, 3, 6, 2, 6],
        [6, 3, 6, 2, 6]])

>>> negative_sample = negative_sampling.generate(sample, mode='head-batch')

>>> negative_sample
tensor([[6, 2, 2, 4, 3],
        [6, 2, 2, 4, 3]])
```

## Methods

???- note "generate"

    Generate negative samples from a head, relation tail

    If the mode is set to head-batch, this method will generate a tensor of fake heads. If the mode is set to tail-batch, this method will generate a tensor of fake tails.

    **Parameters**

    - **sample**    
    - **mode**    
    
???- note "get_true_head_and_tail"

    Build a dictionary to filter out existing triples from fakes ones.

    - **triples**    
    
## References

[^1]: [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

