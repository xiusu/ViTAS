from core.search_space.ops import Attention, Mlp, FC, Identity
import torch.nn as nn
import copy


def count_flops(model, subnet_m = None, subnet_c = None, input_shape = [3, 224, 224], heads_share = False):

    flops = []
    c, w, h = input_shape
    if subnet_m is None and subnet_c is None:
        # for searched architectures
        for m in model.module.modules():
            # Embedding layer
            if isinstance(m, nn.Conv2d):
                w = (w + m.padding[0] * 2 - m.kernel_size[0]) // m.stride[0] + 1
                h = (h + m.padding[1] * 2 - m.kernel_size[1]) // m.stride[1] + 1
                flops.append(m.in_channels * m.out_channels * w * h // m.groups * m.kernel_size[0] * m.kernel_size[1])
                c_in = m.out_channels

            elif isinstance(m, Attention):
                # first fc
                temp_m = m.qkv
                c_out = temp_m.out_features
                flops.append( h * w * c_in * c_out)

                c_in = c_out // 3
                # MSA
                flops.append(2 * (h * w)**2 * c_in)   

                # second fc
                temp_m = m.proj
                c_out = temp_m.out_features
                flops.append( h * w * c_in * c_out)
                c_in = c_out
                

            elif isinstance(m, Mlp):
                # first fc
                temp_m = m.fc1
                c_out = temp_m.out_features
                flops.append( h * w * c_in * c_out)
                c_in = c_out


                # second fc
                temp_m = m.fc2
                c_out = temp_m.out_features
                flops.append( h * w * c_in * c_out)
                c_in = c_out

            
            elif isinstance(m, FC):
                c_out = m.oup
                flops.append( h * w * c_in * c_out)

    else: 
        # for supernet and sampler
        assert len(subnet_m) == len(model.module.net), f'supernet: {len(model.module.net)} should have same len with subnet_m: {len(subnet_m)}'
        subnet_c_copy = copy.deepcopy(subnet_c)
        c_in = c

        FLAGs = False
        for ms, idx in zip(model.module.net, subnet_m):
            if heads_share and FLAGs:
                if idx == 4:
                    op_idx = 1
                else:
                    op_idx = 0
            else:
                # FLAGs for the first Embedding layer
                FLAGs = True
                op_idx = idx

            for m in ms[op_idx].modules():
                if isinstance(m, nn.Conv2d):
                    FLAGs_channel_id = subnet_c_copy.pop(0)
                    c_out = round(m.out_channels * FLAGs_channel_id)
                    w = (w + m.padding[0] * 2 - m.kernel_size[0]) // m.stride[0] + 1
                    h = (h + m.padding[1] * 2 - m.kernel_size[1]) // m.stride[1] + 1
                    flops.append(c_in * c_out * w * h // m.groups * m.kernel_size[0] * m.kernel_size[1])
                    c_in = c_out

                elif isinstance(m, Identity):
                    subnet_c_copy.pop(0)
                    subnet_c_copy.pop(0)
                    flops.append(0)
                    flops.append(0)

                elif isinstance(m, Attention):
                    # first fc
                    temp_m = m.out_dim1
                    c_out = round(temp_m * subnet_c_copy.pop(0))
                    flops.append( h * w * c_in * c_out)

                    c_in = c_out // 3
                    # MSA
                    flops.append(2 * (h * w)**2 * c_in)   

                    # second fc
                    temp_m = m.out_dim2
                    c_out = round(temp_m * FLAGs_channel_id)
                    flops.append( h * w * c_in * c_out)
                    c_in = c_out
                
                elif isinstance(m, Mlp):
                    # first fc
                    temp_m = m.fc1
                    c_out = round(temp_m.out_features * subnet_c_copy.pop(0))
                    flops.append( h * w * c_in * c_out)
                    c_in = c_out

                    # second fc
                    temp_m = m.fc2
                    c_out = round(temp_m.out_features * FLAGs_channel_id)
                    flops.append( h * w * c_in * c_out)
                    c_in = c_out

                elif isinstance(m, FC):
                    c_out = m.oup
                    flops.append( h * w * c_in * c_out)
        assert len(subnet_c_copy) == 0, f'subnet_c should be 0, subnet_c_copy: {subnet_c_copy}, subnet_c: {subnet_c}'
    return sum(flops)


