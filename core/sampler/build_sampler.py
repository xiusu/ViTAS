from .random_sampler import RandomSampler
from .evolution.evolution_sampler import EvolutionSampler


def build_sampler(cfg, model, tester, net_cfg, **kwargs):
    sampler_type = cfg.get('type', 'evolution')
    kwargs = cfg.get('kwargs', {})
    if sampler_type == 'evolution':
        return EvolutionSampler(model=model, tester=tester, net_cfg=net_cfg, **kwargs)
    elif sampler_type == 'random':
        return RandomSampler(model=model, tester=tester, net_cfg=net_cfg, **kwargs)
    else:
        raise NotImplementedError(f'Sampler type {sampler_type} not implemented.')
