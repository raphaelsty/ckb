
__all__ = ['filter_entities']


def filter_entities(dataset):
    """Exclude entities of the input dataset"""
    return list(set([h for h, _, _ in dataset]))
