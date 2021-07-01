import torch.distributed as dist
from tensorboardX import SummaryWriter
from core.utils.misc import AverageMeter
from core.utils.logger import create_logger
from tools.trainer.imagenet.trainer import ImagenetTrainer


def build_trainer(cfg_stg, dataloader, model, optimizer, now, **kwargs):
    ''' Build trainer and return '''
    # choose trainer function
    #kwargs = {}
    kwargs['rank'] = dist.get_rank()
    kwargs['world_size'] = dist.get_world_size()
    kwargs['max_epochs'] = cfg_stg['max_epochs']
    kwargs['quantization'] = cfg_stg.get('quantization', None)
    kwargs['data_time'] = AverageMeter(20)
    kwargs['forw_time'] = AverageMeter(20)
    kwargs['batch_time'] = AverageMeter(20)
    kwargs['mixed_training'] = cfg_stg.get('mixed_training', False)
    #kwargs['task_keys'] = search_space.search_space.task_names
    if cfg_stg['task_type'] in ['imagenet']:
        trainer = ImagenetTrainer
        kwargs['disp_loss'] = AverageMeter()
        kwargs['disp_acc_top1'] = AverageMeter()
        kwargs['disp_acc_top5'] = AverageMeter()
        kwargs['task_has_accuracy'] = True #search_space.head.task_has_accuracy
    else:
        raise RuntimeError('task_type {} invalid, must be in imagenet'.format(cfg_stg['task_type']))

    if now != '':
        now = '_' + now

    for k, v in cfg_stg['kwargs'].items():
        kwargs[k] = v
    
    # build logger
    if cfg_stg['task_type'] in ['verify']:
        logger = create_logger('global_logger',
                               '{}/log/log_task{}_train{}.txt'.format(cfg_stg['save_path'], now, model.task_id))
    # TRACKING_TIP
    elif cfg_stg['task_type'] in ['attribute', 'gaze', 'imagenet', 'tracking', 'smoking']:
        logger = create_logger('global_logger',
                               '{}/log/'.format(cfg_stg['save_path']) + '/log_train{}.txt'.format(now))
    else:
        raise RuntimeError('task_type musk be in verify/attribute/gaze/imagenet/tracking')
    tb_logger = SummaryWriter('{}/events'.format(cfg_stg['save_path']))

    # build trainer
    final_trainer = trainer(dataloader, model, optimizer,
                            cfg_stg.get('print_freq', 20), cfg_stg['save_path'] + '/checkpoint',
                            cfg_stg.get('snapshot_freq', 5000), logger, tb_logger,
                            **kwargs)
    return final_trainer


if __name__ == '__main__':
    pass
