import torch
import torch.nn as nn
import torch.nn.functional as F
from core.search_space import init_model
import copy
import torch.distributed as dist
import time
if __name__ == '__main__':
    from basemodel import BaseModel
else:
    from .basemodel import BaseModel


class Net(BaseModel):
    def __init__(self, cfg_net):
        super(Net, self).__init__()

        self.heads_share = cfg_net.get('heads_share', False)
        self.net = init_model(cfg_net)

        self._init_params()
        self.rank = dist.get_rank()



    def forward(self, input, subnet=None, **kwargs):
        if isinstance(input, dict):
            x = input['images']
        else:
            x = input

        # train supernet
        if subnet is not None:
            self.name_list = []
            self.block_list = []
            for name, block in self.net.named_children():
                self.name_list.append(name)
                self.block_list.append(block)
        
            self.len_block = len(self.block_list)
            subnet_m = subnet[:self.len_block]
            subnet_c = subnet[self.len_block:]
            subnet_c_temp = copy.deepcopy(subnet_c)
            side = kwargs['side']

        # train supernet
        if subnet is not None:
            self.set_subnet(subnet_m)
            for idx, name, block in zip(subnet_m, self.name_list, self.block_list):
                if 'Patch_init' in name:
                    patch_embed = subnet_c_temp.pop(0)
                    block[idx].c_mult_idx = patch_embed
                    block[idx].side = side
                    x = block[idx](x)

                elif 'Block' in name or 'id' in name:
                    if self.heads_share:
                        assert idx <= 4, f'idx should leq than 4, it is {idx}'
                        if idx == 4:
                            x = block[1](x) # id op
                        else:
                            block[0].c_mult_idx1 = subnet_c_temp.pop(0)     # First FC in Attention block
                            block[0].c_mult_idx2 = patch_embed              # skip line
                            block[0].c_mult_idx3 = subnet_c_temp.pop(0)     # First FC in MLP block
                            block[0].c_mult_idx4 = patch_embed              # skip line
                            block[0].heads = idx
                            block[0].side = side
                            x = block[0](x)                                 # normal block
                    else:
                        raise RuntimeError('not share heads')

                # other normal blocks
                else:
                    block[idx].side = side
                    x = block[idx](x)
            logits = x
            
        # retrain searched architecture
        else:
            logits = self.normal_step(x)

        return logits


    def normal_step(self, x):
        for block in self.net:
            total_op = len(block)
            assert total_op == 1
            x = block[0](x)
        return x
    
