if __name__ == '__main__':
    from  base_searcher import BaseSearcher
else:
    from .base_searcher import BaseSearcher


import torch
import torch.distributed as dist
#from core.search_space.ops import channel_mults
import random
import time



class UniformSearcher(BaseSearcher):
    def __init__(self, **kwargs):
        super(UniformSearcher, self).__init__()
        self.rank = dist.get_rank()
        for k in kwargs:
            setattr(self, k, kwargs[k])
        self.flops_constrant = kwargs.pop('flops_constrant', 400e6)
        assert hasattr(self, 'channel_percent'), "channel groups for nas"
        #assert self.side_scheme in ['AutoSlim', 'BCNet'], 'should have side_scheme'
        #self.side = 0 #left

    def generate_subnet(self, model, id_prop = None):
        # We do not adpot this file
        raise RuntimeError('Stop')
        assert id_prop is not None, "id_prop should be a small number"

        subnet_m = []
        subnet_c = []

        for name, block in model.net.named_children():
            assert len(block) in [1,2], "block len less than 2, depends id op"
            subnet_m.append(random.randint(0, len(block) - 1))
            if 'Patch_init' in name:
                subnet_c.append(random.choice(self.channel_percent))
            elif 'Block' in name or 'id' in name:
                subnet_c.append(random.choice(self.channel_percent))
                subnet_c.append(random.choice(self.channel_percent))

        subnet_m = torch.IntTensor(subnet_m)
        subnet_c = torch.FloatTensor(subnet_c)
        dist.broadcast(subnet_m,0)   
        dist.broadcast(subnet_c,0)             
        subnet_m = subnet_m.tolist()
        subnet_c = subnet_c.tolist()
        if self.side_scheme == 'BCNet':
            if self.side == 0:          ##left
                self.side = 1           ##right
            elif self.side == 1:        ##right
                self.side = 0           ##left
        return subnet_m, subnet_c, self.side