from mkb import losses as mkb_losses


class Adversarial(mkb_losses.Adversarial):
    """Self-adversarial negative sampling loss function.
    References:
        1. [Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space." arXiv preprint arXiv:1902.10197 (2019).](https://arxiv.org/pdf/1902.10197.pdf)
        2. [Knowledge Graph Embedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
    """

    def __init__(self, alpha=0.5):
        super().__init__(alpha=alpha)
