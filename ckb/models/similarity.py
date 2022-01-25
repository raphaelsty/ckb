__all__ = ["Similarity"]

import torch

from ..scoring import TransE
from .base import BaseModel


class Similarity(BaseModel):
    """Sentence Similarity models wrapper.

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

    >>> from transformers import AutoTokenizer, AutoModel

    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    >>> model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    >>> _ = torch.manual_seed(42)

    >>> dataset = datasets.Semanlink(1, pre_compute=False)

    >>> model = models.Similarity(
    ...    model = model,
    ...    tokenizer = tokenizer,
    ...    entities = dataset.entities,
    ...    relations = dataset.relations,
    ...    gamma = 9,
    ...    device = 'cpu',
    ... )

    >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
    >>> model(sample)
    tensor([[3.5273],
            [3.6367]], grad_fn=<ViewBackward>)

    >>> sample = torch.tensor([[0, 0, 1], [2, 2, 1]])
    >>> model(sample)
    tensor([[-78.3936],
            [-79.7217]], grad_fn=<ViewBackward>)


    >>> sample = torch.tensor([[1, 0, 0], [1, 2, 2]])
    >>> model(sample)
    tensor([[-78.1690],
            [-80.2369]], grad_fn=<ViewBackward>)

    >>> sample = torch.tensor([[0, 0, 0], [2, 2, 2]])
    >>> negative_sample = torch.tensor([[0], [2]])

    >>> model(sample, negative_sample, mode='head-batch')
    tensor([[3.5273],
            [3.6367]], grad_fn=<ViewBackward>)

    >>> model(sample, negative_sample, mode='tail-batch')
    tensor([[3.5273],
            [3.6367]], grad_fn=<ViewBackward>)

    References
    ----------
    1. [Sentence Similarity models](https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=downloads)

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
            init_l2 = False
        else:
            init_l2 = True

        super(Similarity, self).__init__(
            hidden_dim=hidden_dim,
            entities=entities,
            relations=relations,
            scoring=scoring,
            gamma=gamma,
        )

        self.tokenizer = tokenizer
        self.model = model
        self.max_length = list(self.tokenizer.max_model_input_sizes.values())[0]
        self.device = device

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
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        sentence_embeddings = self.mean_pooling(output=output, attention_mask=attention_mask)

        if self.l2 is not None:
            sentence_embeddings = self.l2(sentence_embeddings)

        return sentence_embeddings

    @staticmethod
    def mean_pooling(output, attention_mask):
        """Mean pooling.

        References
        ----------
        1. [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
        """
        token_embeddings = (
            output.last_hidden_state
        )  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
