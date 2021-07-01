from abc import abstractmethod
from core.model.net import Net
from tools.eval.base_tester import BaseTester
import torch.distributed as dist
import torch
import time
from core.utils.flops import count_flops
from core.utils.measure import measure_model
import os
import copy


class BaseSampler:
    def __init__(self, model: Net, tester: BaseTester, **kwargs):
        self.model = model
        self.tester = tester
        self.rank = dist.get_rank()
        log_path = kwargs.pop('log_path', None)
        self.flops_min = kwargs.pop('flops_min', 0)
        if log_path is not None and self.rank == 0:
            self.logger = open(os.path.join(log_path, 'subnet.log'), 'a')
        else:
            self.logger = None
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def forward_subnet(self, input):
        """
        run one step
        :return:
        """
        pass

    def eval_subnet(self, subnet_list):
        if self.rank == 0:
            print(f'num: {self.num}')
        self.num = self.num + 1
        if not isinstance(subnet_list, list):
            subnet_list = subnet_list.tolist()
        
        subnet_temp = copy.deepcopy(subnet_list[:self.len_block])
        subnet_temp.extend(self.covert_channels(subnet_list[self.len_block:]))

        if self.GPU_search:
            Pics_count = measure_model(self.model, subnet = subnet_temp)
            Pics_count = int(Pics_count)
            Pics_count = torch.IntTensor([Pics_count]).cuda()
            dist.broadcast(Pics_count,0)   
            Pics_count = Pics_count[0].item()
            if self.rank == 0:
                print('==subnet: {}, Pics_count: {}'.format(str(subnet_list), Pics_count))
            if Pics_count > self.GPUs_constraint or Pics_count < self.GPUs_min:
                return 0 - (Pics_count - self.GPUs_constraint) / self.GPUs_min
        else:
            flops = count_flops(self.model, subnet_m = subnet_list[:self.len_block], subnet_c = self.covert_channels(subnet_list[self.len_block:]), heads_share = self.heads_share)
            if self.rank == 0:
                print('==subnet: {}, FLOPs: {}'.format(str(subnet_list), flops))
            if flops > self.flops_constraint or flops < self.flops_min:     # drop this subnet
                return 5 - (flops - self.flops_constraint) / self.flops_constraint
                
        torch.distributed.barrier()
        if getattr(self, 'cal_train_mode', True):  #no cal
            # recal bn
            self.model.module.reset_bn()
            self.model.train()
            # need to forward twice to recal bn - (2 x 50k / gpu_num) imgs
            #self.tester.dataloader = None
            time.sleep(2)  # process may be stuck in dataloader
            score, _ = self.tester.test(copy.deepcopy(subnet_temp), side = 'left', eval_mode = False)


            if self.side_scheme == 'BCNet':
                time.sleep(2)
                score_r, _ = self.tester.test(copy.deepcopy(subnet_temp), side = 'right', eval_mode = False)
                score = (score + score_r)/2


            elif self.side_scheme == 'AutoSlim':
                pass

            if self.logger is not None:
                if self.GPU_search:
                    self.logger.write('{}-{}-{}-{}\n'.format(str(subnet_list), score, Pics_count, self.num))
                    self.logger.flush()
                else:
                    self.logger.write('{}-{}-{}-{}\n'.format(str(subnet_list), score, flops, self.num))
                    self.logger.flush()
            return score
        else:      #cal twice and test once
            # recal bn
            self.model.module.reset_bn()
            self.model.train()
            # need to forward twice to recal bn - (2 x 50k / gpu_num) imgs
            # self.tester.dataloader = None
            time.sleep(2)  # process may be stuck in dataloader
            score = self.tester.test(subnet_list)
            time.sleep(2)  # process may be stuck in dataloader
            score = self.tester.test(subnet_list)
            time.sleep(2)  # process may be stuck in dataloader
            if self.rank == 0:
                print('==training subnet: {}, score: {}'.format(str(subnet_list), score))
            self.model.eval()
            score = self.tester.test(subnet_list, eval_mode = False)
            if self.logger is not None:
                self.logger.write('{}-{}-{}\n'.format(str(subnet_list), score, flops))
                self.logger.flush()
            return score

    @abstractmethod
    def generate_subnet(self):
        """
        generate one subnet
        :return: block indexes for each choice block
        """

    def sample(self):
        subnet_eval_dict = {}
        subnet_eval_list = []
        while len(subnet_eval_dict) < self.search_num:
            # first, generate a subnet
            if self.rank == 0:
                subnet = self.generate_subnet()
                subnet = torch.FloatTensor(subnet)
            else:
                subnet = torch.zeros([len(self.generate_subnet())], dtype=torch.FloatTensor)
            dist.broadcast(subnet, 0)
            subnet = subnet.tolist()
            subnet_t = tuple(subnet)
            if subnet_eval_dict.get(subnet_t) is not None:
                # already searched
                continue
            # set subnet
            score = self.eval_subnet(subnet)
            if score == 0:  # flops not suitable, continue to next subnet
                continue
            if self.rank == 0:
                print('==testing subnet: {}, score: {}'.format(str(subnet), score))
            subnet_eval_dict[subnet_t] = score
            subnet_eval_list.append((subnet_t, score))
        sorted_subnet = sorted(subnet_eval_dict.items(), key=lambda i:i[1], reverse=True)
        sorted_subnet_key = [x[0] for x in sorted_subnet]
        subnet_topk = sorted_subnet_key[:self.sample_num]
        if self.rank == 0:
            print('== search result ==')
            print(sorted_subnet)
            print('== best subnet ==')
            print(subnet_topk)
            print('== subnet eval list ==')
            print([x[1] for x in subnet_eval_list])
        return None

