from .uniform_searcher import UniformSearcher

def build_searcher(searcher_type, **kwargs):
    if searcher_type == 'uniform':
        return UniformSearcher(**kwargs)
    else:
        raise RuntimeError('not support search_type {}'.format(searcher_type))