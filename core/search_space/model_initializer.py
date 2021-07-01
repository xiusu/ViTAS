import torch.nn as nn
from core.search_space.ops import OPS
from core.search_space.ops import FC
import copy
import torch


def re_map_model(FLAGs, model_id, heads, ops):
    if FLAGs:
        return heads, ops
    else:
        temp_id = model_id.pop(0)
        
        if temp_id == 0:
            ops = [ops[0]]
            if len(heads)>0:
                heads = [heads[0]]
        elif temp_id >= len(heads):
            assert ops[-1] == 'id', f'ops: {ops}'
            ops = [ops[-1]]
            
        else:
            heads = [heads[temp_id]]
            ops = [ops[0]]

        return heads, ops

def re_map_channel(FLAGs, FLAGs_channel_id = None, channel_id = None, at_temp = None, at_out = None, MLP_temp = None, MLP_out =None, type = None):
    if FLAGs:
        if type == 'Patch_init':
            return at_out, None
        elif type == 'Block':
            return at_temp, at_out, MLP_temp, MLP_out
    else:
        if type == 'Patch_init':
            FLAGs_channel_id = channel_id.pop(0)
            at_out = round(at_out * FLAGs_channel_id)
            return at_out, FLAGs_channel_id
        elif type == 'Block':
            at_temp = round(at_temp * channel_id.pop(0))
            at_out = round(at_out * FLAGs_channel_id)
            MLP_temp = round(MLP_temp * channel_id.pop(0))
            MLP_out = round(MLP_out * FLAGs_channel_id)
            return at_temp, at_out, MLP_temp, MLP_out

        elif type == 'id':
            channel_id.pop(0)
            channel_id.pop(0)
        else:
            raise RuntimeError(f'not support type {type}')

def covert_channels(subnet_c, channel_percent):
    subnet_percent = []
    for i in subnet_c:
        subnet_percent.append(channel_percent[i])
    return subnet_percent




def init_model(cfg_net):
    model = nn.ModuleList()
    net_id = cfg_net.pop('net_id', None)
    channel_percent = cfg_net.pop('channel_percent', False)
    model_len = cfg_net.pop('model_len', False)
    drop_path_rate = cfg_net.pop('drop_path_rate', 0)
    heads_share = cfg_net.pop('heads_share', False)
    qkv = cfg_net.pop('qkv', 0)
    depth = cfg_net.pop('depth')
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]



    if net_id is not None:
        FLAGs = False
        net_id = [int(i) for i in net_id.replace('  ', ' ').split(' ')]
        model_id = net_id[:model_len]
        channel_id = net_id[model_len:]
        channel_id = covert_channels(channel_id, channel_percent)
    else:
        FLAGs = True
        model_id = False
        channel_id = False


    inp_temp = 3 # hard code
    for _type in cfg_net:
        if _type == 'backbone':
            for stage in cfg_net[_type]:
                #[n, input, heads(int or list)/patchs, Attention_temp, out, MLP_temp, MLP_out, channel_search, op]
                n, inp, heads, at_temp, at_out, MLP_temp, MLP_out, c_search, ops  = cfg_net[_type][stage]
                for i in range(n):
                    temp_heads, temp_ops = re_map_model(FLAGs, model_id, heads, ops)
                    module_ops = nn.ModuleList()
                    for op in temp_ops:
                        if op == 'Patch_init':
                            for p in temp_heads:
                                at_out_t, FLAGs_channel_id = re_map_channel(FLAGs, channel_id = channel_id, at_out = at_out, type = 'Patch_init')
                                module_ops = module_ops.append(OPS[op](inp_temp, p, at_out_t, c_search))
                                inp_temp = at_out_t
                        elif op == 'Block':
                            if heads_share:
                                at_temp_t, at_out_t, MLP_temp_t, MLP_out_t = re_map_channel(FLAGs, FLAGs_channel_id = FLAGs_channel_id, channel_id = channel_id, at_temp = at_temp, at_out = at_out, MLP_temp = MLP_temp, MLP_out = MLP_out, type = 'Block')
                                module_ops = module_ops.append(OPS[op](inp_temp, temp_heads, at_temp_t, at_out_t, MLP_temp_t, MLP_out_t, dpr[i], c_search, qkv))
                                inp_temp = MLP_out_t
                            else:
                                for h in temp_heads:
                                    at_temp_t, at_out_t, MLP_temp_t, MLP_out_t = re_map_channel(FLAGs, FLAGs_channel_id = FLAGs_channel_id, channel_id = channel_id, at_temp = at_temp, at_out = at_out, MLP_temp = MLP_temp, MLP_out = MLP_out, type = 'Block')
                                    module_ops = module_ops.append(OPS[op](inp_temp, h, at_temp_t, at_out_t, MLP_temp_t, MLP_out_t, dpr[i], c_search, qkv))
                                    inp_temp = MLP_out_t
                        elif op == 'id':
                            re_map_channel(FLAGs, FLAGs_channel_id = FLAGs_channel_id, channel_id = channel_id, type = 'id')
                            module_ops.append(OPS[op]())
                        elif op == 'Norm':
                            module_ops.append(OPS[op](inp_temp))
                        elif op == 'FC':
                            module_ops.append(OPS[op](inp_temp, at_out))
                            inp_temp = at_out
                        else:
                            raise RuntimeError("not support op: {}".format(op))
                    model.add_module(f'{_type}_{stage}_{op}_{i}', module_ops)

    if not FLAGs:
        assert len(model_id) == 0, f'model_id: {model_id}'
        assert len(channel_id) == 0, f'channel_id: {channel_id}'

    if model_id is not False:
        assert len(model_id) == 0, f'model_id: {model_id}'
        assert len(channel_id) == 0, f'channel_id: {channel_id}'

    return model

