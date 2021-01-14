from .dataset import TestDataset
from .dataset import Dataset
from .fb15k237 import Fb15k237
from .qsemanlink import QSemanlink
from .semanlink import Semanlink
from .wn18rr import Wn18rr

__all__ = [
    'Dataset',
    'QSemanlink',
    'Semanlink',
    'Wn18rr',
    'Fb15k237',
]
