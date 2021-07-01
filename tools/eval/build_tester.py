from core.dataset.build_dataloader import build_dataloader
import torch.distributed as dist
from tools.eval.imagenet.tester import ImagenetTester
import os


def build_tester(cfg_stg, test_loader, model, **kwargs):
    ''' Build tester and return '''

    #kwargs = {}
    kwargs['rank'] = dist.get_rank()
    kwargs['world_size'] = dist.get_world_size()
    task_type = cfg_stg.get('task_type', '')
    dataloader_func = build_dataloader

    model_folder = os.path.join(cfg_stg['save_path'], 'checkpoint')
    model_name = cfg_stg.get('load_name', None)

    if task_type in ['imagenet-test']:
        tester = ImagenetTester
    else:
        raise RuntimeError('Wrong task_type of {}, task_type musk be in verify-test/attribute-test/gaze-test'
                           '/imagenet-test/tracking-test/feature-out/smoking-test'.format(task_type))

    # build tester
    final_tester = tester(test_loader, model, model_folder, model_name,  **kwargs)
    return final_tester
