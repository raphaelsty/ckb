import importlib

import torch
import transformers

from ..scoring import RotatE, TransE
from .base import BaseModel

__all__ = ["FlauBERT"]


class FlauBERT(BaseModel):
    """FlauBERT for contextual representation of entities.

    Parameters
    ----------
        gamma (int): A higher gamma parameter increases the upper and lower bounds of the latent
            space and vice-versa.
        entities (dict): Mapping between entities id and entities label.
        relations (dict): Mapping between relations id and entities label.

    Examples
    --------

    >>> from ckb import models
    >>> from ckb import datasets

    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> dataset = datasets.Semanlink(1)

    >>> model = models.FlauBERT(
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
    tensor([[-25.7671],
            [-15.1038]], grad_fn=<ViewBackward>)

    >>> sample = torch.tensor([[1, 0, 0], [1, 2, 2]])
    >>> model(sample)
    tensor([[-24.8325],
            [-14.1621]], grad_fn=<ViewBackward>)

    >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
    >>> negative_sample = torch.tensor([[1], [1]])

    >>> model(sample, negative_sample, mode='head-batch')
    tensor([[-24.8325],
            [-14.1621]], grad_fn=<ViewBackward>)

    >>> model(sample, negative_sample, mode='tail-batch')
    tensor([[-25.7670],
            [-15.1038]], grad_fn=<ViewBackward>)

    >>> model = models.FlauBERT(
    ...    entities = dataset.entities,
    ...    relations = dataset.relations,
    ...    gamma = 9,
    ...    device = 'cpu',
    ... )

    >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
    >>> model(sample)
    tensor([[3.6564],
            [3.5718]], grad_fn=<ViewBackward>)

    """

    def __init__(
        self,
        entities,
        relations,
        scoring=TransE(),
        hidden_dim=None,
        gamma=9,
        device="cuda",
    ):
        if hidden_dim is None:
            hidden_dim = 768
            init_l2 = False
        else:
            init_l2 = True

        super(FlauBERT, self).__init__(
            hidden_dim=hidden_dim,
            entities=entities,
            relations=relations,
            scoring=scoring,
            gamma=gamma,
        )

        self.model_name = "flaubert/flaubert_base_uncased"

        self.tokenizer = transformers.FlaubertTokenizer.from_pretrained(self.model_name)

        self.max_length = self.tokenizer.max_model_input_sizes[self.model_name]

        self.device = device

        self.l1 = transformers.FlaubertModel.from_pretrained(self.model_name)

        if init_l2:
            self.l2 = torch.nn.Linear(768, hidden_dim)
        else:
            self.l2 = None

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

        pooler = hidden_state[:, 0]

        if self.l2 is not None:
            pooler = self.l2(pooler)

        return pooler
