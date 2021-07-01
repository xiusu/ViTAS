import os
import time
from os.path import join, exists

import torch
import torch.distributed as dist
from util.torch_dist_sum import *

from util.meter import AverageMeter
from util.accuracy import accuracy
from tools.eval.base_tester import BaseTester

class ImagenetTester(BaseTester):

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        for k, v in kwargs.items():
            setattr(self, k, v)

        #check customized trainer has all required attrs
        self.required_atts = ('rank', 'local_rank', 'world_size')
        for att in self.required_atts:
            if not hasattr(self, att):
                raise RuntimeError(f'ImagenetTester must has attr: {att}')
        self.dataloader = None

    @torch.no_grad()
    def test(self, subnet=None, side=None, eval_mode = True):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
    
        ''' test setted model in self '''
        if not self.model_loaded:
            self.load()
            self.model_loaded = True

        # switch to train mode
        if eval_mode:
            self.model.eval()

        end = time.time()
        for i, (img, target) in enumerate(self.test_loader):
            # measure data loading time
            
            data_time.update(time.time() - end)

            img = img.cuda(self.local_rank, non_blocking=True)
            target = target.cuda(self.local_rank, non_blocking=True)

            output = self.model(img, subnet = subnet, side = side)

            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], img.size(0))
            top5.update(acc5[0], img.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        sum1, cnt1, sum5, cnt5 = torch_dist_sum(self.local_rank, top1.sum, top1.count, top5.sum, top5.count)
        top1_acc = sum(sum1.float()) / sum(cnt1.float())
        top5_acc = sum(sum5.float()) / sum(cnt5.float())

        return top1_acc, top5_acc




    def load(self):
        if self.model_loaded or self.model_name is None:
            return
        # load state_dict
        ckpt_path = join(self.model_folder, self.model_name)
        assert exists(ckpt_path), f'{ckpt_path} not exist.'
        if self.rank == 0:
            print(f'==[rank{self.rank}]==loading checkpoint from {ckpt_path}')

        checkpoint =  torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])

        ckpt_keys = set(checkpoint['model'].keys())
        own_keys = set(self.model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
           print(f'==[rank{self.rank}]==**missing key**:{k}')
        print(f'==[rank{self.rank}]==load model done.')