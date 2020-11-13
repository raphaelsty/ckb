
__all__ = ['filter_entities', 'get_same_entities']


def filter_entities(dataset):
    """Exclude entities of the input dataset"""
    return list(set([t for _, _, t in dataset]))


def get_same_entities(dataset, entities):
    same_entities = {}
    # (The question is a portal to the original entity)
    for h, r, t in dataset:
        if r == 'has_question':
            same_entities[entities[t]] = entities[h]

    # (The answer is a portal to the original entity and not the question)
    for h, r, t in dataset:
        if r == 'has_answer':
            same_entities[entities[t]] = same_entities[entities[h]]

    return same_entities
