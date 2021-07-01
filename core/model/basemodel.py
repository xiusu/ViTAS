import torch.nn as nn
import math
from network.trunc_norm import trunc_normal_

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.param_setted = False
        self.searcher = None



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                trunc_normal_(m.bias, std=1e-6)

    def _init_weights_trunc(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                trunc_normal_(m.bias, std=1e-6)


    def _init_params(self, init = 'trunc'):
        if init == 'ortho':
            self.apply(self._init_weights)
        elif init == 'trunc':
            self.apply(self._init_weights_trunc)
        else:
            raise RuntimeError("not support init type: {}".format(init))

    def reset_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or issubclass(m.__class__, nn.BatchNorm2d):
                m.reset_running_stats()

    def add_searcher(self, searcher, start_iter=0):
        if self.searcher is None:
            self.searcher = {}
        self.searcher[start_iter] = searcher

    def remove_searcher(self):
        self.searcher = None

    def get_loss(self, logits, label, **kwargs):
        raise NotImplementedError()

    def set_subnet(self, idx_list):
        for cb, idx in zip(self.net, idx_list):
            if self.heads_share:
                if idx == 4:
                    temp_idx = 1
                else:
                    temp_idx = 0
            else:
                temp_idx = idx

            for b_idx, block in enumerate(cb):
                if b_idx != temp_idx:
                    for param in block.parameters():
                        param.requires_grad = False
                else:
                    for param in block.parameters():
                        param.requires_grad = True
