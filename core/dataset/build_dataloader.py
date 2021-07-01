import torch
from data.imagenet import *
from data.augmentation import get_deit_aug, get_deit_xaa_xcj_aug, get_deit_test_aug
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def build_dataloader(cfg_data, is_test = False):
    if 'batch_size' not in cfg_data:
        batch_size = cfg_data['imagenet']['batch_size']
    else:
        batch_size = cfg_data['batch_size']
    num_workers = cfg_data['workers']
    data_dir = cfg_data['data_dir']

    if is_test:
        test_aug = get_deit_test_aug
        dataset = datasets.ImageFolder(data_dir,test_aug(res=224))
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=sampler)
    else:
        # for retraining searched architectures
        if cfg_data.get('augmentation') == 'deit':
            train_aug = get_deit_aug
        # for training supernet
        elif cfg_data.get('augmentation') == 'deit_xaa_xcj':
            train_aug = get_deit_xaa_xcj_aug
        else:
            raise RuntimeError("not support augmentation: {}".format(cfg_data.get('augmentation')))
        dataset = datasets.ImageFolder(data_dir,train_aug(res=224))
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=sampler, drop_last=True)
    
    return loader, sampler