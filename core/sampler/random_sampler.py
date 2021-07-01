"""
choose a subnet by random search
"""

from core.sampler.base_sampler import BaseSampler
#from core.search_space.ops import channel_mults
import random


class RandomSampler(BaseSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise RuntimeError("stop")

    def generate_subnet(self):
        subnet = []
        for block in self.model.net:
            subnet.append(random.randint(0, len(block) - 1))
        for idx, block in enumerate(self.model.net):
            if getattr(block[0], 'channel_search', False):
                subnet.append(random.randint(0, len(channel_mults) - 1))
            else:
                subnet.append(channel_mults.index(1.0))
        return subnet
