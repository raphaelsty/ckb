## CKB
Contextual knowledge bases.


This is an informal implementation of the model focusing on the link prediction task [Inductive Entity Representations from Text via Link Prediction](https://arxiv.org/abs/2010.03496) which is mainly an ablation study of [KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation](https://arxiv.org/abs/1911.06136). This tool is based on the library [MKB](https://github.com/raphaelsty/mkb). CKB is designed to be compatible with HuggingFace's models.


The CKB library is dedicated to knowledge bases and allows to fine-tune HuggingFace models using the link prediction task. The objective of this fine-tuning task is to make accurate embeddings of knowledge graph entities. The link prediction task aims to train a model to find the missing element of an RDF triplet. For the triplet `(France, is_part_of, ?)`, the model may find the entity `Europe`. This library replaces the embeddings traditionally used with models dedicated to knowledge graphs by an encoder (TransE vs BERT). Here the encoder is a pre-trained transformer. Using a transformer has many advantages such as building contextualized latent representations of entities. Moreover this model can encode entities it has never seen with the textual description of the entity. The training time is much longer than a classic TransE model, however the model converges with fewer epochs.

There is only one model available for the moment "Distillbert". It is relatively easy to integrate a new model from HuggingFace. Do not hesitate to open an issue if you need to.


## Installation

```
pip install git+https://github.com/raphaelsty/ckb
```

## Train your own model:

```python
from ckb import compose
from ckb import datasets
from ckb import evaluation
from ckb import losses
from ckb import models
from ckb import sampling
from ckb import scoring

import torch

_ = torch.manual_seed(42)

device = 'cuda' #  You should own a GPU, it is very slow with cpu.

# Train, valid and test sets are a list of triples.
train = [
    ('My Favorite Carrot Cake Recipe', 'made_with', 'Brown Sugar'),
    ('My Favorite Carrot Cake Recipe', 'made_with', 'Oil'),
    ('My Favorite Carrot Cake Recipe', 'made_with', 'Applesauce'),
    
    ('Classic Cheesecake Recipe', 'made_with', 'Block cream cheese'),
    ('Classic Cheesecake Recipe', 'made_with', 'Sugar'),
    ('Classic Cheesecake Recipe', 'made_with', 'Sour cream'),
]

valid = [
    ('My Favorite Carrot Cake Recipe', 'made_with', 'A bit of sugar'), 
    ('Classic Cheesecake Recipe', 'made_with', 'Eggs')
]

test = [
    ('My Favorite Strawberry Cake Recipe', 'made_with', 'Fresh Strawberry')
]

# Initialize the dataset, batch size should be small to avoid RAM exceed. 
dataset = datasets.Dataset(
    hidden_dim = 500,
    batch_size = 1,
    train = train,
    valid = valid,
    test = test,
    seed = 42,
)

model = models.DistillBert(
    entities = dataset.entities,
    relations = dataset.relations,
    gamma = 9,
    scoring = scoring.TransE(), # Scoring function. TransE is suitable.
    device = device,
)

model = model.to(device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr = 0.00005,
)
    
evaluation = evaluation.Evaluation(
    entities = dataset.entities,
    relations = dataset.relations,
    true_triples = dataset.train + dataset.valid + dataset.test,
    batch_size = 1,
    device = device,
)

# Number of negative samples to show to the model for each batch.
# Should be small to avoid memory error.
sampling = sampling.NegativeSampling(
    size = 1,
    entities = dataset.entities,
    relations = dataset.relations,
    train_triples = dataset.train,
)

pipeline = compose.Pipeline(
    epochs = 20,
    eval_every = 3, # Eval the model every {eval_every} epochs.
    early_stopping_rounds = 1, 
    device = device,
)

pipeline = pipeline.learn(
    model = model,
    dataset = dataset,
    evaluation = evaluation,
    sampling = sampling,
    optimizer = optimizer,
    loss = losses.Adversarial(alpha=0.5),
)
```

## Encode entities:

```python
embeddings = {}

for _, e in model.entities.items():
    with torch.no_grad():
        embeddings[e] = model.encoder([e]).cpu()
```


## Encode new entities:

```python
new_entities = [
    'My favourite apple pie',
    'How to make croissant',
    'Pain au chocolat with coffee',
]

embeddings = {}

for e in new_entities:
    with torch.no_grad():
        embeddings[e] = model.encoder([e]).cpu()
```

## Save trained model:

```python
torch.save(model, 'model_ckb.pth')
```

## Load saved model:

```python
model = torch.load(f'model_ckb.pth')

device = 'cuda'
model.device = device # Ugly but necessary for now.
model.to(device)
```
