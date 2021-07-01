import torch.distributed as dist


class BaseSearcher:
    def __init__(self):
        self.rank = dist.get_rank()
        self.searched = False

    def search_step(self, x, model):
        raise NotImplemented('search_step must be implemented in sub-classes')

    def get_topk_arch(self, k):
        raise NotImplemented('get_topk_arch must be implemented in sub-classes')


