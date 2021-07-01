from torch.autograd import Variable
import torch.nn as nn
import torch
import time
import sys
import yaml
from core.model.net import Net
def count_fc_flops_params(m, x, y):
    ret = 2 * m.weight.numel()
    n_ps = m.weight.numel()
    if m.bias is None:
        ret -= m.bias.size(0)
    else:
        n_ps += m.bias.size(0)
    m.flops = torch.Tensor([ret])
    m.n_params = torch.Tensor([n_ps])

def count_conv_flops_params(m, x, y):
    c_out, c_in, ks_h, ks_w = m.weight.size()
    out_h, out_w = y.size()[-2:]
    n_ps = m.weight.numel()
    if m.bias is None:
        ret = (2 * c_in * ks_h * ks_w - 1) * out_h * out_w * c_out / m.groups
    else:
        ret = 2 * c_in * ks_h * ks_w * out_h * out_w * c_out / m.groups
        n_ps += m.bias.size(0)
    m.flops = torch.Tensor([ret])
    m.n_params = torch.Tensor([n_ps])

def count_bn_params(m, x, y):
    n_ps = 0
    if m.weight is not None:
        n_ps += m.weight.numel()
    if m.bias is not None:
        n_ps += m.bias.numel()
    m.n_params = torch.Tensor([n_ps])

def flops_str(FLOPs):
    preset = [(1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'K')]

    for p in preset:
        if FLOPs // p[0] > 0:
            N = FLOPs / p[0]
            ret = "%.3f%s" % (N, p[1])
            return ret
    ret = "%.1f" % (FLOPs)
    return ret

@torch.no_grad()
def measure_model(model, subnet = None , input_shape=[3, 224, 224], eval_times = 10):
    model.eval()
    fake_data = {'images': torch.randn([256] + input_shape)}
    if torch.cuda.is_available():
        fake_data['images'] = fake_data['images'].cuda()
    model(fake_data, subnet = subnet, side = 'left')
    s_t = time.time()
    for i in range(eval_times):
        model(fake_data, subnet = subnet, side = 'left')
    avg_time = (time.time() - s_t) /eval_times
    pics = 1 / avg_time * 256
    #print(f'ave_time: {avg_time}, pics: {pics}')
    return pics

if __name__ == '__main__':
    cfg = sys.argv[1]
    config = yaml.load(open(sys.argv[1], 'r')).pop('test')
    model_cfg = config['model']
    model = Net(model_cfg)
    data_cfg = config.pop('data')
    input_shape = [data_cfg['final_channel'], data_cfg['final_height'], data_cfg['final_width']]
    print(measure_model(model, input_shape=input_shape))
