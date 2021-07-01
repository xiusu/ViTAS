
from collections import defaultdict
import os
import torch

class Info():
    def __init__(self):
        super().__init__()
        self.best_avg = -1
        self._info = defaultdict(list)

    def add(self, task_names, value):
        self._info[task_names].append(value)

    def avg_(self, prefix, ckpt):
        total_task = 0
        total_avg = 0
        for k in self._info:
            avg = sum(self._info[k]) / len(self._info[k])
            total_task += 1
            total_avg += avg
        if total_task != 0:
            if total_avg / total_task > self.best_avg:
                self.best_avg = total_avg / total_task
                ckpt['best_avg'] = self.best_avg
                torch.save(ckpt, os.path.join(prefix, 'best_ckpt.pth.tar'))
        self._info = defaultdict(list)

class BaseTester(object):
    ''' Base Trainer class '''

    def __init__(self, test_loader, model, model_folder, model_name):
        self.test_loader = test_loader
        self.model = model
        self.set_model_path(model_name, model_folder)
        # self.dataloader_fun = dataloader_fun
        self.best_loaded = False
        self.info = Info()

    def test(self, subnet):
        raise RuntimeError('BaseTester cannot test')

    def load(self):
        raise RuntimeError('BaseTester cannot load search_space')

    def gen_dataloader(self):
        raise RuntimeError('BaseTester cannot generate dataloader')

    def set_model_path(self, model_name, model_folder=None):
        self.model_name = model_name
        if model_folder is not None:
            self.model_folder = model_folder
        self.model_loaded = False

    def eval_init(self):
        raise RuntimeError('BaseTester cannot init evaluation')

    def save_eval_result(self):
        raise RuntimeError('BaseTester cannot save evaluation result')

    def predict_single_img(self):
        raise RuntimeError('BaseTester cannot predict single image')
