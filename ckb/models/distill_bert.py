import transformers

import importlib

import torch

from .base import BaseModel

from ..scoring import TransE
from ..scoring import RotatE


__all__ = ['DistillBert']


class DistillBert(BaseModel):
    """DistillBert for contextual representation of entities.

    Parameters:
        gamma (int): A higher gamma parameter increases the upper and lower bounds of the latent
            space and vice-versa.
        entities (dict): Mapping between entities id and entities label.
        relations (dict): Mapping between relations id and entities label.

    Example:

        >>> from ckb import models
        >>> from ckb import datasets

        >>> import torch

        >>> _ = torch.manual_seed(42)

        >>> dataset = datasets.Semanlink(1)

        >>> model = models.DistillBert(
        ...    hidden_dim = 50,
        ...    entities = dataset.entities,
        ...    relations = dataset.relations,
        ...    gamma = 9,
        ...    device = 'cpu',
        ... )

        >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
        >>> model(sample)
        tensor([[3.1645],
                [3.2653]], grad_fn=<ViewBackward>)

        >>> sample = torch.tensor([[0, 0, 1], [2, 2, 1]])
        >>> model(sample)
        tensor([[2.5616],
                [0.8435]], grad_fn=<ViewBackward>)

        >>> sample = torch.tensor([[1, 0, 0], [1, 2, 2]])
        >>> model(sample)
        tensor([[1.1692],
                [1.1021]], grad_fn=<ViewBackward>)

        >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
        >>> negative_sample = torch.tensor([[1], [1]])

        >>> model(sample, negative_sample, mode='head-batch')
        tensor([[1.1692],
                [1.1021]], grad_fn=<ViewBackward>)

        >>> model(sample, negative_sample, mode='tail-batch')
        tensor([[2.5616],
                [0.8435]], grad_fn=<ViewBackward>)

    """

    def __init__(self,  hidden_dim, entities, relations, scoring=TransE(), gamma=9, device='cuda'):
        
        super(DistillBert, self).__init__(
            hidden_dim=hidden_dim,
            entities=entities,
            relations=relations,
            scoring=scoring,
            gamma=gamma
        )

        self.model_name = 'distilbert-base-uncased'

        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(
            self.model_name)

        self.max_length = self.tokenizer.max_model_input_sizes[self.model_name]

        self.device = device

        self.l1 = transformers.DistilBertModel.from_pretrained(self.model_name)
        
        self.l2 = torch.nn.Linear(768, hidden_dim)

    def encoder(self, e):
        """Encode input entities descriptions.

        Parameters:
            e (list): List of description of entities.

        Returns:
            Torch tensor of encoded entities.
        """
        inputs = self.tokenizer.batch_encode_plus(
            e,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
        )

        output = self.l1(
            input_ids=torch.tensor(inputs['input_ids']).to(self.device),
            attention_mask=torch.tensor(
                inputs['attention_mask']).to(self.device)
        )

        hidden_state = output[0]

        pooler = hidden_state[:, 0]

        return self.l2(pooler)
