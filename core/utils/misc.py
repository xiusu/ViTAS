import numpy as np
from functools import reduce

import torch

class AverageMeter(object):
    ''' Computes and stores the average and current value '''
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        self.history = []
        self.count = 0
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.all = 0.0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.all = self.sum / 3600
        if self.length > 0:
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]
            self.avg = np.mean(self.history)
        else:
            self.avg = self.sum / self.count

def get_cls_accuracy(output, target, topk=(1, ), ignore_indices=[-1]):
    """Computes the precision@k for the specified values of k"""
    target = target.long()
    masks = [target != idx for idx in ignore_indices]
    mask = reduce(lambda x, y : x&y, masks)
    keep = torch.nonzero(mask).squeeze()
    if keep.numel() <= 0:
        return [torch.cuda.FloatTensor([1]).zero_()]
    if keep.dim() == 0:
        keep = keep.view(-1)
    assert keep.dim() == 1, keep.dim()
    target = target[keep]
    output = output[keep]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
