import transformers

import importlib

import torch

from .base import BaseModel
from ..scoring_function import TransE


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
        ...    entities = dataset.entities,
        ...    relations = dataset.relations,
        ...    gamma = 9,
        ...    device = 'cpu',
        ... )

        >>> sample = torch.tensor([[0, 0, 0]])

        >>> model(sample)
        tensor([[3.6489]], grad_fn=<ViewBackward>)

    """

    def __init__(self, entities, relations, scoring_function=TransE(), gamma=9, device='cuda'):

        super(DistillBert, self).__init__(
            hidden_dim=768,
            entity_dim=768,
            relation_dim=768,
            entities=entities,
            relations=relations,
            gamma=gamma
        )

        self.scoring_function = scoring_function

        self.model_name = 'distilbert-base-uncased'

        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(
            self.model_name)

        self.max_length = self.tokenizer.max_model_input_sizes[self.model_name]

        self.device = device

        self.l1 = transformers.DistilBertModel.from_pretrained(self.model_name)

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

        return pooler

    def forward(self, sample, negative_sample=None, mode=None):
        """Compute scores of input sample, negative sample with respect to the mode."""

        head, relation, tail, shape = self.encode(
            sample=sample,
            negative_sample=negative_sample,
            mode=mode
        )

        score = self.scoring_function(
            head=head,
            relation=relation,
            tail=tail,
            gamma=self.gamma,
            mode=mode,
        )

        return score.view(shape)
