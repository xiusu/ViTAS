import time
from os.path import join, exists

import torch
import torch.distributed as dist
from tools.trainer.base_trainer import BaseTrainer
from util.meter import AverageMeter, ProgressMeter
import math
from util.mixup import Mixup
import torch.nn.functional as F
from core.utils.flops import count_flops
from core.utils.measure import measure_model
import random
import torch.nn as nn
import time
import glob
import os


class ImagenetTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        for k,v in kwargs.items():
            setattr(self, k, v)
        
        # check customized trainer has all required attrs
        self.required_atts = ('rank', 'local_rank', 'world_size')
        for att in self.required_atts:
            if not hasattr(self, att):
                raise RuntimeError(f'ImagenetTester must has attr: {att}')
        if self.rank == 0:
            print(f'[rank{self.rank}]ImagenetTrainer build done.')
            # print(self.model)
        if not hasattr(self, 'GPU_search'):
            self.GPU_search = False

        if not hasattr(self, 'heads_share'):
            self.heads_share = False
            
        if not hasattr(self, 'search_id'):
            self.search_id = True

        if not hasattr(self, 'mixup_stg'):
            self.mixup_stg = False
        
        if self.mixup_stg == 'mixup':
            print("only mixup")
            self.mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=0, prob=1.0, switch_prob=0, mode='batch', label_smoothing=0.1, num_classes=1000)
        elif self.mixup_stg == 'cutmix':
            print("only cutmix")
            self.mixup_fn = Mixup(mixup_alpha=0, cutmix_alpha=1.0, prob=1.0, switch_prob=0, mode='batch', label_smoothing=0.1, num_classes=1000)
        else:
            print("all mixup_fn")
            self.mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=1000)


    def train(self, **kwargs):
        if self.search == True:
            self.init_search()


        for epoch in range(self.start_epoch, self.max_epochs):
            self.train_sampler.set_epoch(epoch)
            self.train_one_ep(epoch = epoch)

            if self.rank == 0:
                self.save(epoch)

    def train_one_ep(self, **kwargs):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            len(self.dataloader),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(kwargs['epoch']))
            # switch to train mode
        self.model.train()
        end = time.time()
        for i, (samples, targets) in enumerate(self.dataloader):
            self.adjust_learning_rate(i, epoch = kwargs['epoch'], iteration_per_epoch=len(self.dataloader))
            data_time.update(time.time() - end)

            samples = samples.cuda(self.local_rank, non_blocking=True)
            targets = targets.cuda(self.local_rank, non_blocking=True)

            if self.mix_up:
                samples, targets = self.mixup_fn(samples, targets)
            
            # compute output
            if self.search == True:
                subnet = self.generate_subnet()
                output = self.model(samples, subnet = subnet, side = 'left')
                if self.side_scheme == 'BCNet':
                   output_r = self.model(samples, subnet = subnet, side = 'right') 
                   output = output + output_r
            else:
                output = self.model(samples)

            if not self.mix_up:
                targets = self.label_smoothing(targets, batch_size = output.size(0), num_class = output.size(1))
            loss = torch.sum(-targets * F.log_softmax(output, dim=-1), dim=-1).mean()

            losses.update(loss.item(), samples.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0 and self.rank == 0:
                progress.display(i)
    
    def adjust_learning_rate(self, i, epoch, iteration_per_epoch):
        T = epoch * iteration_per_epoch + i
        warmup_iters = self.warm_up * iteration_per_epoch
        total_iters = (self.max_epochs - self.warm_up) * iteration_per_epoch

        if epoch < self.warm_up:
            lr = self.base_lr * 1.0 * T / warmup_iters
        else:
            T = T - warmup_iters
            lr = 0.5 * self.base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save(self, epoch):
        ''' save search_space '''
        self.delete_checkpoints(self.save_path+'/*')
        path = join(self.save_path, 'epoch_{}_ckpt.pth.tar'.format(epoch))
        torch.save({'epoch': epoch + 1, 'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)
        path_recent = join(self.save_path, 'recent_ckpt.pth.tar')
        torch.save({'epoch': epoch + 1, 'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}, path_recent)
        
        if self.rank == 0:
            print('[rank{}]Saved search_space to {}.'.format(self.rank, path))
    

    def load(self, ckpt_path):
        assert exists(ckpt_path), f'{ckpt_path} not exist.'
        checkpoint =  torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        if self.rank == 0:
            print('load [resume] search_space done, '
                             f'current iter is {self.cur_iter}')

    def generate_subnet(self):
        assert self.id_prop is not None, "id_prop should be a small number"

        subnet_m = []
        subnet_c = []

        
        for name, block in zip(self.name_list, self.block_list):
            if 'Block' in name or 'id' in name:
                # id op
                if random.random() < self.id_prop:
                    if self.heads_share:
                        subnet_m.append(len(block) - 1 + 3) #3 heads
                    else:
                        subnet_m.append(len(block) - 1)
                # norm op
                else:
                    if self.heads_share:
                        subnet_m.append(random.randint(0,len(block) - 2 + 3))
                    else:
                        subnet_m.append(random.randint(0, max(len(block) - 2,0)))
            elif 'Patch_init' in name:
                subnet_m.append(random.randint(0, len(block)-1))
            else:
                subnet_m.append(0)

            if 'Patch_init' in name:
                subnet_c.append(random.randint(0, len(self.channel_percent)-1))
            elif 'Block' in name or 'id' in name:
                subnet_c.append(random.randint(0, len(self.channel_percent)-1))
                subnet_c.append(random.randint(0, len(self.channel_percent)-1))

        subnet_m = torch.IntTensor(subnet_m).cuda()
        subnet_c = torch.IntTensor(subnet_c).cuda()
        dist.broadcast(subnet_m,0)   
        dist.broadcast(subnet_c,0)             
        subnet_m = subnet_m.tolist()
        subnet_c = subnet_c.tolist()

        subnet_c = self.covert_channels(subnet_c)
        subnet_m.extend(subnet_c)
        return subnet_m
    
    def covert_channels(self, subnet_c):
        subnet_percent = []
        for i in subnet_c:
            subnet_percent.append(self.channel_percent[i])
        return subnet_percent

    def init_search(self):
        assert self.side_scheme in ['AutoSlim', 'BCNet'], 'side scheme is {}'.format(self.side_scheme)
        self.name_list = []
        self.block_list = []
        for name, block in self.model.module.net.named_children():
            self.name_list.append(name)
            self.block_list.append(block)
        self.len_block = len(self.block_list)

        if not hasattr(self, 'search_id'):
            self.search_id = True

        if self.search_id:
            if self.GPU_search:
                self.prop_search_GPU()
            else:
                self.prop_search()
        else:
            self.id_prop = 0.5


    def prop_search(self):
        self.id_prop = 0.5
        if self.rank == 0:
            print(f'start prop_search: {self.id_prop}')
        
        if not self.search_id:
            print('Not search ID')
            return
        while(True):
            FLOPs_count = 0
            for i in range(self.ps_num):
                subnet = self.generate_subnet()
                FLOPs_count += count_flops(self.model, subnet_m = subnet[:self.len_block], subnet_c = subnet[self.len_block:], heads_share = self.heads_share)
    
            FLOPs_count = FLOPs_count / self.ps_num
            if FLOPs_count > self.flops_constraint:
                self.id_prop = self.id_prop + 0.01
                if self.rank == 0:
                    print(f'Too large, id_prop: {self.id_prop}, FLOPs_count: {FLOPs_count}, flops_constraint: {self.flops_constraint}')
            elif FLOPs_count < self.flops_min:
                self.id_prop = self.id_prop - 0.01
                if self.rank == 0:
                    print(f'Too small, id_prop: {self.id_prop}, FLOPs_count: {FLOPs_count}, flops_constraint: {self.flops_min}')
            else:
                if self.rank == 0:
                    print("all_right, id_prop is {}, FLOPs_count: {}".format(self.id_prop, FLOPs_count))
                break
                
            if self.id_prop >= 1 or self.id_prop <= 0:
                raise RuntimeError(f'stop, id_prop: {self.id_prop}')

    def prop_search_GPU(self):
        self.id_prop = 0.5
        if self.rank == 0:
            print(f'start prop_search_GPU: {self.id_prop}')
        
        if not self.search_id:
            print('Not search ID')
            return

        finished = torch.Tensor([0]).cuda()
        if self.rank == 0:
            while(True):
                Pics_count = 0
                time_cost = []
                for i in range(self.ps_num):
                    subnet = self.generate_subnet()
                    time_cost.append(measure_model(self.model, subnet = subnet, eval_times = 10))

                time_cost.sort()
                Pics_count = sum(time_cost[(self.ps_num//2-2): (self.ps_num//2+2)])
                
                Pics_count = Pics_count / 4
                if Pics_count > self.GPUs_constraint:
                    self.id_prop = self.id_prop - 0.01
                    if self.rank == 0:
                        print(f'Too large, id_prop: {self.id_prop}, Pics_count: {Pics_count}, GPUs_constraint: {self.GPUs_constraint}')
                    finished = torch.Tensor([0]).cuda()
                    dist.broadcast(finished,0)
                elif Pics_count < self.GPUs_min:
                    self.id_prop = self.id_prop + 0.01
                    if self.rank == 0:
                        print(f'Too small, id_prop: {self.id_prop}, Pics_count: {Pics_count}, GPUs_min: {self.GPUs_min}')
                    finished = torch.Tensor([0]).cuda()
                    dist.broadcast(finished,0)
                else:
                    print("all_right, id_prop is {}, Pics_count: {}".format(self.id_prop, Pics_count))
                    finished = torch.Tensor([1]).cuda()
                    dist.broadcast(finished,0)
                    break
                if self.id_prop >= 1 or self.id_prop <= 0:
                    raise RuntimeError(f'stop, id_prop: {self.id_prop}')
        else:
            while(True):
                finished = torch.Tensor([0]).cuda()
                for i in range(self.ps_num):
                    subnet = self.generate_subnet()
                    measure_model(self.model, subnet = subnet, eval_times = 10)
                dist.broadcast(finished,0)
                if finished[0] == 1:
                    break




        
    def label_smoothing(self, label, batch_size, num_class):
        label = label.long()
        label_smooth = torch.zeros((batch_size, num_class)).cuda()
        label_smooth.scatter_(1, label.unsqueeze(1), 1)
        ones_idx = label_smooth == 1
        zeros_idx = label_smooth == 0
        label_smooth[ones_idx] = 0.9
        label_smooth[zeros_idx] = 0.1 / (num_class - 1)
        return label_smooth




    def delete_checkpoints(self, path):
        
        checkpoints = glob.glob(path)
        checkpoints.sort(key=os.path.getmtime)
        total = len(checkpoints)
        if total > 21:
            os.remove(checkpoints[0])
