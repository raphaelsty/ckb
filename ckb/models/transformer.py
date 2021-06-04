import importlib

import torch
import transformers

from ..scoring import RotatE, TransE
from .base import BaseModel

__all__ = ["Transformer"]


class Transformer(BaseModel):
    """Transformer for contextual representation of entities.

    Parameters
    ----------
        gamma (int): A higher gamma parameter increases the upper and lower bounds of the latent
            space and vice-versa.
        entities (dict): Mapping between entities id and entities label.
        relations (dict): Mapping between relations id and entities label.

    Examples
    --------

    >>> import torch
    >>> _ = torch.manual_seed(42)

    >>> from ckb import models
    >>> from ckb import datasets

    >>> from transformers import BertTokenizer
    >>> from transformers import BertModel

    >>> dataset = datasets.Semanlink(1)

    >>> model = models.Transformer(
    ...    model = BertModel.from_pretrained('bert-base-uncased'),
    ...    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased'),
    ...    entities = dataset.entities,
    ...    relations = dataset.relations,
    ...    gamma = 9,
    ...    device = 'cpu',
    ... )

    >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
    >>> model(sample)
    tensor([[3.5500],
            [3.2861]], grad_fn=<ViewBackward>)

    >>> sample = torch.tensor([[0, 0, 1], [2, 2, 1]])
    >>> model(sample)
    tensor([[-227.8486],
            [-197.0484]], grad_fn=<ViewBackward>)

    >>> sample = torch.tensor([[1, 0, 0], [1, 2, 2]])
    >>> model(sample)
    tensor([[-227.8378],
            [-196.5193]], grad_fn=<ViewBackward>)

    >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
    >>> negative_sample = torch.tensor([[1], [1]])

    >>> model(sample, negative_sample, mode='head-batch')
    tensor([[-227.8378],
            [-196.5193]], grad_fn=<ViewBackward>)

    >>> model(sample, negative_sample, mode='tail-batch')
    tensor([[-227.8486],
            [-197.0484]], grad_fn=<ViewBackward>)

    """

    def __init__(
        self,
        model,
        tokenizer,
        entities,
        relations,
        scoring=TransE(),
        hidden_dim=None,
        gamma=9,
        device="cuda",
    ):
        if hidden_dim is None:
            hidden_dim = 768

        super(Transformer, self).__init__(
            hidden_dim=hidden_dim,
            entities=entities,
            relations=relations,
            scoring=scoring,
            gamma=gamma,
        )

        self.tokenizer = tokenizer

        self.max_length = self.tokenizer.model_max_length

        self.device = device

        self.l1 = model

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
            padding="max_length",
            return_token_type_ids=True,
        )

        output = self.l1(
            input_ids=torch.tensor(inputs["input_ids"]).to(self.device),
            attention_mask=torch.tensor(inputs["attention_mask"]).to(self.device),
        )

        hidden_state = output[0]

        return hidden_state[:, 0]
