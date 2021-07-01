import importlib
from core.model.net import Net
from core.searcher import build_searcher
from core.sampler import build_sampler
from core.dataset.build_dataloader import build_dataloader
from tools.eval.build_tester import build_tester
from tools.trainer import build_trainer
# from cellular.parallel import DistributedModel as DM
# import cellular.pape.distributed as dist
import time
from os.path import join, exists
import os
from core.utils.measure import measure_model
import torch
from core.utils.flops import count_flops

from torch.nn.parallel import DistributedDataParallel
from util.dist_init import dist_init
from data.augmentation import get_deit_aug
from util.weight_decay import create_params

#VIT_tiny: 1284003840
#VIT_small: 4648906752
            
class NAS_Vit:
    def __init__(self, config):
        self.cfg_net = config.pop('model')
        self.cfg_search = config.pop('search')
        self.cfg_sample = config.pop('sample')
        self.cfg_retrain = config.pop('retrain')
        self.cfg_test = config.pop('test')
        self.port = config.pop('port')
    
    def run(self):
        #search config
        cfg_search_stg = self.cfg_search.pop('strategy')
        cfg_search_data = self.cfg_search.pop('data')
        
        #sample config
        cfg_sample_sampler = self.cfg_sample.pop('sampler')
        cfg_sample_stg = self.cfg_sample.pop('strategy')
        cfg_sample_data = self.cfg_sample.pop('data')

        cfg_search_stg['kwargs']['flops_constraint'] = cfg_sample_sampler['kwargs']['flops_constraint']
        cfg_search_stg['kwargs']['flops_min'] = cfg_sample_sampler['kwargs']['flops_min']

        if cfg_sample_sampler['kwargs'].get('GPU_search', False):
            cfg_search_stg['kwargs']['GPUs_constraint'] = cfg_sample_sampler['kwargs']['GPUs_constraint']
            cfg_search_stg['kwargs']['GPUs_min'] = cfg_sample_sampler['kwargs']['GPUs_min']

        cfg_search_stg['kwargs']['base_lr'] = cfg_search_stg['optimizer']['lr']

        #retrain config
        cfg_retrain_stg = self.cfg_retrain.pop('strategy')
        cfg_retrain_data = self.cfg_retrain.pop('data')
        cfg_retrain_stg['kwargs']['base_lr'] = cfg_search_stg['optimizer']['lr']

        #test config
        cfg_test_stg = self.cfg_test.pop('strategy')
        cfg_test_data = self.cfg_test.pop('data')

        self.rank, self.local_rank, self.world_size = dist_init(self.port)

        cfg_search_data['batch_size'] = cfg_search_data['batch_size'] // self.world_size
        cfg_sample_data['batch_size'] = cfg_sample_data['batch_size'] // self.world_size
        cfg_retrain_data['batch_size'] = cfg_retrain_data['batch_size'] // self.world_size
        cfg_test_data['batch_size'] = cfg_test_data['batch_size'] // self.world_size

        # retrain
        if self.cfg_retrain['flag'] or self.cfg_sample['flag']:
            self._build_model(self.cfg_net)
        
        # search
        # build path
        cfg_sample_stg['save_path'] = join(cfg_search_stg['save_path'], 'search')
        cfg_retrain_stg['save_path'] = join(cfg_search_stg['save_path'], 'retrain')
        cfg_test_stg['save_path'] = join(cfg_search_stg['save_path'], 'retrain')
        cfg_search_stg['save_path'] = join(cfg_search_stg['save_path'], 'search')
        if self.cfg_search['flag']:
            # join path for search and sample

            if self.rank == 0:
                if not exists(join(cfg_search_stg['save_path'], 'checkpoint')):
                    os.makedirs(join(cfg_search_stg['save_path'], 'checkpoint'))
                if not exists(join(cfg_search_stg['save_path'], 'events')):
                    os.makedirs(join(cfg_search_stg['save_path'], 'events'))
                if not exists(join(cfg_search_stg['save_path'], 'log')):
                    os.makedirs(join(cfg_search_stg['save_path'], 'log'))

            self._build_searcher(cfg_search_data, cfg_search_stg)
            self.search()
        
        # sample
        if self.cfg_sample['flag']:
            if not exists(cfg_sample_stg['save_path']):
                os.makedirs(cfg_sample_stg['save_path'])
            self._build_sampler(cfg_sample_sampler, cfg_sample_data, cfg_sample_stg, self.cfg_net)
            self.sample()
            self.subnet_candidates = self.sampler.subnet_top1

        # retrain
        if self.cfg_retrain['flag']:
            if self.rank == 0:
                if not exists(join(cfg_retrain_stg['save_path'], 'checkpoint')):
                    os.makedirs(join(cfg_retrain_stg['save_path'],'checkpoint'))
                if not exists(join(cfg_retrain_stg['save_path'], 'events')):
                    os.makedirs(join(cfg_retrain_stg['save_path'], 'events'))
                if not exists(join(cfg_retrain_stg['save_path'], 'log')):
                    os.makedirs(join(cfg_retrain_stg['save_path'], 'log'))
            if hasattr(self, 'subnet_candidates') and self.subnet_candidates is not None:
                cfg_subnet = self.subnet_candidates
            elif 'model' in self.cfg_retrain:
                cfg_subnet = self.cfg_retrain['model']
            else:
                raise ValueError('no model config specified!')
            self._build_model(cfg_subnet)
            self._build_retrainer(cfg_retrain_data, cfg_retrain_stg)
            self.retrain()

        # test
        if self.cfg_test['flag']:
            if 'model' in self.cfg_test:
                cfg_subnet = self.cfg_test['model']
            elif 'model' in self.cfg_retrain:
                cfg_subnet = self.cfg_retrain['model']
            else:
                raise ValueError('no model config specified!')

            self._build_model(cfg_subnet)
            avg_t  = measure_model(self.model)

            if self.rank == 0:
                print('Retraining subnet FLOPs:')
                print(count_flops(self.model))
                print('average gpu pics:')
                print(avg_t)
            torch.distributed.barrier()
            self._build_tester(cfg_test_data, cfg_test_stg)
            self.test(cfg_test_stg)


    def _build_model(self, cfg_net):
        model = Net(cfg_net)
        model = model.cuda()
        model = DistributedDataParallel(model, device_ids=[self.local_rank], find_unused_parameters=True)
        self.model = model

    def _build_searcher(self, cfg_data_search, cfg_stg_search):
        ##!!!! a little different need change according to vit
        self.search_dataloader, self.search_sampler = build_dataloader(cfg_data_search, is_test = False)
        params = create_params(self.model, weight_decay = cfg_stg_search['optimizer']['weight_decay'])
        opt = torch.optim.AdamW(params, lr=cfg_stg_search['optimizer']['lr'], weight_decay=cfg_stg_search['optimizer']['weight_decay'])
        self.search_trainer = build_trainer(cfg_stg_search, self.search_dataloader, self.model, opt
                                        , time.strftime("%Y%m%d_%H%M%S", time.localtime()), local_rank = self.local_rank, train_sampler = self.search_sampler)
        if cfg_stg_search.get('resume', False):
            if cfg_stg_search['load_name'] == 'recent_ckpt.pth.tar':
                if os.path.exists(join(cfg_stg_search['save_path'], 'checkpoint', cfg_stg_search['load_name'])):
                    self.search_trainer.load(join(cfg_stg_search['save_path'], 'checkpoint', cfg_stg_search['load_name']))
            else:
                self.search_trainer.load(join(cfg_stg_search['save_path'], 'checkpoint', cfg_stg_search['load_name']))

    def _build_retrainer(self, cfg_data_retrain, cfg_stg_retrain):
        self.retrain_dataloader, self.retrain_sampler = build_dataloader(cfg_data_retrain, is_test = False)
        params = create_params(self.model, weight_decay = cfg_stg_retrain['optimizer']['weight_decay'])
        opt = torch.optim.AdamW(params, lr=cfg_stg_retrain['optimizer']['lr'], weight_decay=cfg_stg_retrain['optimizer']['weight_decay'])
        self.trainer = build_trainer(cfg_stg_retrain, self.retrain_dataloader, self.model, opt
                                        , '', local_rank = self.local_rank, train_sampler = self.retrain_sampler)
        if cfg_stg_retrain.get('resume', False):
            if cfg_stg_retrain['load_name'] == 'recent_ckpt.pth.tar':
                if os.path.exists(join(cfg_stg_retrain['save_path'], 'checkpoint', cfg_stg_retrain['load_name'])):
                    self.trainer.load(join(cfg_stg_retrain['save_path'], 'checkpoint', cfg_stg_retrain['load_name']))
            else:
                self.trainer.load(join(cfg_stg_retrain['save_path'], 'checkpoint', cfg_stg_retrain['load_name']))

    def search(self):
        self.search_trainer.train()
        #self.model.remove_searcher()
    

    def retrain(self):
        # count ops
        avg_t = measure_model(self.model)
        if self.rank == 0:
            print('Retraining subnet FLOPs:')
            print(count_flops(self.model))
            print('Average running time:')
            print(f'pics_1s: {avg_t}')
            print("Retraining Params")
            print(sum([m.numel() for m in self.model.module.net.parameters()]))
        torch.distributed.barrier()
        self.trainer.train()
        
        

    def _build_sampler(self, cfg_sample_sampler, cfg_data_sample, cfg_stg_sample, cfg_net):
        self.sample_dataloader, self.sample_sampler = build_dataloader(cfg_data_sample, is_test = True)
        self.tester = build_tester(cfg_stg_sample, self.sample_dataloader, self.model, local_rank = self.local_rank)
        self.sampler = build_sampler(cfg_sample_sampler, self.model, self.tester, cfg_net)
    
    def sample(self):
        self.sampler.sample()

    def _build_tester(self, cfg_data_test, cfg_stg_test):
        self.test_dataloader, self.test_sampler = build_dataloader(cfg_data_test, is_test = True)
        self.tester = build_tester(cfg_stg_test, self.test_dataloader, self.model, local_rank = self.local_rank)
    

    def test(self, cfg_test_stg):
        for ckpt_iter in range(cfg_test_stg['start'], cfg_test_stg['end'], cfg_test_stg['strip']):
            model_name = f'epoch_{ckpt_iter}_ckpt.pth.tar'
            model_folder = cfg_test_stg.get('load_path', cfg_test_stg['save_path'])
            model_folder = join(model_folder, 'checkpoint')
            while not exists(join(model_folder, model_name)):
                if self.rank == 0:
                    print('{} not exists, waiting for training,'
                          ' current time: {}'.format(join(model_folder, model_name),
                                                     time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                time.sleep(300)
            self.model.eval()
            self.tester.set_model_path(model_name=model_name, model_folder=model_folder)
            top1, top5 = self.tester.test()
            if self.rank == 0:
                print(f'model_name: {model_name}, top1: {top1}, top5: {top5}')
            time.sleep(1)





        


