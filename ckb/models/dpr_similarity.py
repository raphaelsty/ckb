__all__ = ["DPRSimilarity"]

import torch

from ..scoring import TransE
from .base import BaseModel


class DPRSimilarity(BaseModel):
    """Two tower Sentence Similarity models wrapper.

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

    >>> head_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    >>> tail_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    >>> head_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    >>> tail_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    >>> _ = torch.manual_seed(42)

    >>> dataset = datasets.Semanlink(1, pre_compute=False)

    >>> model = models.DPRSimilarity(
    ...    head_model = head_model,
    ...    tail_model = tail_model,
    ...    head_tokenizer = head_tokenizer,
    ...    tail_tokenizer = tail_tokenizer,
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
        head_model,
        tail_model,
        head_tokenizer,
        tail_tokenizer,
        entities,
        relations,
        scoring=TransE(),
        hidden_dim=None,
        gamma=9,
        device="cuda",
    ):

        if hidden_dim is None:
            hidden_dim = 768

        super(DPRSimilarity, self).__init__(
            hidden_dim=hidden_dim,
            entities=entities,
            relations=relations,
            scoring=scoring,
            gamma=gamma,
        )

        self.head_tokenizer = head_tokenizer
        self.tail_tokenizer = tail_tokenizer

        self.head_model = head_model
        self.tail_model = tail_model

        self.head_max_length = list(self.head_tokenizer.max_model_input_sizes.values())[0]
        self.tail_max_length = list(self.tail_tokenizer.max_model_input_sizes.values())[0]

        self.device = device

    def encoder(self, e, mode=None):
        """Encode input entities descriptions.

        Parameters:
            e (list): List of description of entities.

        Returns:
            Torch tensor of encoded entities.
        """

        if mode is None:
            mode = "head"

        tokenizer = self.head_tokenizer if mode == "head" else self.tail_tokenizer
        max_length = self.head_max_length if mode == "head" else self.tail_max_length

        inputs = tokenizer.batch_encode_plus(
            e,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_token_type_ids=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        if mode == "head":

            output = self.head_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        else:

            output = self.tail_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        sentence_embeddings = self.mean_pooling(output=output, attention_mask=attention_mask)

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
