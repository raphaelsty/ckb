from mkb import sampling as mkb_sampling

__all__ = ['NegativeSampling']


class NegativeSampling(mkb_sampling.NegativeSampling):

    def __init__(self, size, train_triples, entities, relations, seed=42):
        super().__init__(size=size, train_triples=train_triples,
                         entities=entities, relations=relations, seed=42)
