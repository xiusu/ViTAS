import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import random
import torch.distributed as dist
from functools import partial
import copy
import math


OPS = OrderedDict()
# CAUTION: The assign order is Strict

'''
ops for single path one shot net, based on shufflenetv2
'''
OPS['Choice_3'] = lambda inp, oup, t, stride, c_search: \
    SNetInvertedResidual(inp, oup, stride, conv_list=[1, 3, 1], max_ks=3, channel_search=c_search)
OPS['Choice_5'] = lambda inp, oup, t, stride, c_search: \
    SNetInvertedResidual(inp, oup, stride, conv_list=[1, 5, 1], max_ks=5, channel_search=c_search)
OPS['Choice_7'] = lambda inp, oup, t, stride, c_search: \
    SNetInvertedResidual(inp, oup, stride, conv_list=[1, 7, 1], max_ks=7, channel_search=c_search)
OPS['Choice_x'] = lambda inp, oup, t, stride, c_search: \
    SNetInvertedResidual(inp, oup, stride, conv_list=[3, 1, 3, 1, 3, 1], max_ks=3, channel_search=c_search)
'''
end ops for single path one shot net, based on shufflenetv2
'''


OPS['ir_3x3_nse'] = lambda inp, oup, t, stride, weight_num, bn_weight_num, c_search: InvertedResidual(inp=inp, oup=oup, t=t, stride=stride, k=3, channel_search=c_search,
                                                                           activation=HSwish, use_se=False, weight_num=weight_num, bn_weight_num=bn_weight_num)
OPS['ir_5x5_nse'] = lambda inp, oup, t, stride, weight_num, bn_weight_num, c_search: InvertedResidual(inp=inp, oup=oup, t=t, stride=stride, k=5, channel_search=c_search,
                                                                           activation=HSwish, use_se=False, weight_num=weight_num, bn_weight_num=bn_weight_num)
OPS['ir_7x7_nse'] = lambda inp, oup, t, stride, weight_num, bn_weight_num, c_search: InvertedResidual(inp=inp, oup=oup, t=t, stride=stride, k=7, channel_search=c_search,
                                                                           activation=HSwish, use_se=False, weight_num=weight_num, bn_weight_num=bn_weight_num)
OPS['ir_3x3_se'] = lambda inp, oup, t, stride, weight_num, bn_weight_num, c_search: InvertedResidual(inp=inp, oup=oup, t=t, stride=stride, k=3, channel_search=c_search,
                                                                           activation=HSwish, use_se=True, weight_num=weight_num, bn_weight_num=bn_weight_num)
OPS['ir_5x5_se'] = lambda inp, oup, t, stride, weight_num, bn_weight_num, c_search: InvertedResidual(inp=inp, oup=oup, t=t, stride=stride, k=5, channel_search=c_search,
                                                                           activation=HSwish, use_se=True, weight_num=weight_num, bn_weight_num=bn_weight_num)
OPS['ir_7x7_se'] = lambda inp, oup, t, stride, weight_num, bn_weight_num, c_search: InvertedResidual(inp=inp, oup=oup, t=t, stride=stride, k=7, channel_search=c_search,
                                                                           activation=HSwish, use_se=True, weight_num=weight_num, bn_weight_num=bn_weight_num)


OPS['ir_3x3'] = lambda inp, oup, t, stride, weight_num, bn_weight_num, c_search: InvertedResidual(inp=inp, oup=oup, t=t, stride=stride, k=3, channel_search=c_search, weight_num=weight_num, bn_weight_num=bn_weight_num)
OPS['ir_5x5'] = lambda inp, oup, t, stride, weight_num, bn_weight_num, c_search: InvertedResidual(inp=inp, oup=oup, t=t, stride=stride, k=5, channel_search=c_search, weight_num=weight_num, bn_weight_num=bn_weight_num)
OPS['ir_7x7'] = lambda inp, oup, t, stride, weight_num, bn_weight_num, c_search: InvertedResidual(inp=inp, oup=oup, t=t, stride=stride, k=7, channel_search=c_search, weight_num=weight_num, bn_weight_num=bn_weight_num)

OPS['nr_3x3'] = lambda inp, oup, t, stride, c_search: NormalResidual(inp=inp, oup=oup, t=t, stride=stride, k=3, channel_search=c_search)
OPS['nr_5x5'] = lambda inp, oup, t, stride, c_search: NormalResidual(inp=inp, oup=oup, t=t, stride=stride, k=5, channel_search=c_search)
OPS['nr_7x7'] = lambda inp, oup, t, stride, c_search: NormalResidual(inp=inp, oup=oup, t=t, stride=stride, k=7, channel_search=c_search)


OPS['nb_3x3'] = lambda inp, oup, t, stride, c_search: DualBlock(inp=inp, oup=oup, t=t, stride=stride, k=3, channel_search=c_search)
OPS['nb_5x5'] = lambda inp, oup, t, stride, c_search: DualBlock(inp=inp, oup=oup, t=t, stride=stride, k=5, channel_search=c_search)
OPS['nb_7x7'] = lambda inp, oup, t, stride, c_search: DualBlock(inp=inp, oup=oup, t=t, stride=stride, k=7, channel_search=c_search)

OPS['rec_3x3'] = lambda inp, oup, t, stride, c_search: RecBlock(inp=inp, oup=oup, t=t, stride=stride, k=3, channel_search=c_search)
OPS['rec_5x5'] = lambda inp, oup, t, stride, c_search: RecBlock(inp=inp, oup=oup, t=t, stride=stride, k=5, channel_search=c_search)
OPS['rec_7x7'] = lambda inp, oup, t, stride, c_search: RecBlock(inp=inp, oup=oup, t=t, stride=stride, k=7, channel_search=c_search)

OPS['ds_3x3'] = lambda inp, oup, t, stride, c_search: DepthwiseSeparableConv(inp=inp, oup=oup, t=t, stride=stride, k=3, channel_search=c_search)
OPS['ds_5x5'] = lambda inp, oup, t, stride, c_search: DepthwiseSeparableConv(inp=inp, oup=oup, t=t, stride=stride, k=5, channel_search=c_search)
OPS['ds_7x7'] = lambda inp, oup, t, stride, c_search: DepthwiseSeparableConv(inp=inp, oup=oup, t=t, stride=stride, k=7, channel_search=c_search)

OPS['lb_3x3'] = lambda inp, oup, t, stride, c_search: LinearBottleneck(inp=inp, oup=oup, stride=stride, k=3, channel_search=c_search)
OPS['lb_5x5'] = lambda inp, oup, t, stride, c_search: LinearBottleneck(inp=inp, oup=oup, stride=stride, k=5, channel_search=c_search)
OPS['lb_7x7'] = lambda inp, oup, t, stride, c_search: LinearBottleneck(inp=inp, oup=oup, stride=stride, k=7, channel_search=c_search)



OPS['conv2d'] = lambda inp, oup, t, stride, weight_num, bn_weight_num, c_search: Conv2d(inp=inp, oup=oup, stride=stride, k=1, channel_search=c_search, weight_num=weight_num, bn_weight_num=bn_weight_num)
OPS['conv3x3'] = lambda inp, oup, t, stride, weight_num, bn_weight_num, c_search: Conv2d(inp=inp, oup=oup, stride=stride, k=3, channel_search=c_search, weight_num=weight_num, bn_weight_num=bn_weight_num)
OPS['Structure_blocks'] = lambda inp, oup, c_search: Structure_blocks(in_channel=inp, out_channel=oup, reduction = c_search)


OPS['Patch_init'] = lambda inp, p, at_out, c_search: Patch_init(in_chans=3, patch_size =p, embed_dim=at_out, c_search = c_search)
OPS['Block'] = lambda inp, h, at_temp, at_out, MLP_temp, MLP_out, dpr, c_search, qkv: Block(dim=inp, num_heads=h, at_temp=at_temp, at_out=at_out, MLP_temp=MLP_temp, MLP_out=MLP_out, drop_path = dpr, c_search = c_search, qkv = qkv)
OPS['id'] = lambda: Identity()
OPS['Norm'] = lambda inp: Norm(normalized_shape=inp)
OPS['FC'] = lambda inp, oup: FC(dim_in=inp, dim_out=oup, use_bn = False, dp = 0, act = None)
'''
ops for single path one shot net, based on shufflenetv2
'''


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        #out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = DynamicLinear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = DynamicLinear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.c_mult_idx1 = None
        self.c_mult_idx2 = None

    def forward(self, x):
        self.fc1.c_mult_idx = self.c_mult_idx1
        self.fc2.c_mult_idx = self.c_mult_idx2
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, at_temp = None, at_out = None, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., qkv = 0):
        super().__init__()
        assert qkv in [0, 1, 2], f'qkv should in pre-set modes, qkv: {qkv}'
        self.num_heads = num_heads
        if isinstance(self.num_heads, list):
            head_dim = dim // 3
        else:
            head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv_value = qkv
        self.out_dim1 = at_temp
        self.out_dim2 = at_out
        if self.qkv_value == 0:
            self.qkv = DynamicLinear(dim, at_temp, bias = qkv_bias)
        elif self.qkv_value == 1:
            self.qk = DynamicLinear(dim, at_temp//3 * 2, bias=qkv_bias)
            self.v0 = DynamicLinear(dim, at_temp//3, bias=qkv_bias)
            self.v1 = DynamicLinear(dim, at_temp//3, bias=qkv_bias)
            self.v2 = DynamicLinear(dim, at_temp//3, bias=qkv_bias)
            self.v3 = DynamicLinear(dim, at_temp//3, bias=qkv_bias)
        elif self.qkv_value == 2:
            self.qk0 = DynamicLinear(dim, at_temp//3 * 2, bias=qkv_bias)
            self.qk1 = DynamicLinear(dim, at_temp//3 * 2, bias=qkv_bias)
            self.qk2 = DynamicLinear(dim, at_temp//3 * 2, bias=qkv_bias)
            self.qk3 = DynamicLinear(dim, at_temp//3 * 2, bias=qkv_bias)
            self.v = DynamicLinear(dim, at_temp//3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = DynamicLinear(at_temp // 3, at_out)
        self.proj_drop = nn.Dropout(proj_drop)
        self.c_mult_idx1 = None
        self.c_mult_idx2 = None
        self.side = None
        self.heads = None


    def forward(self, x):
        self.proj.c_mult_idx = self.c_mult_idx2
        if self.heads is not None:
            num_heads = self.num_heads[self.heads]
        else:
            num_heads = self.num_heads
        
        if self.qkv_value == 0:
            self.qkv.c_mult_idx = self.c_mult_idx1
            qkv = self.qkv(x)
        elif self.qkv_value == 1:
            self.qk.c_mult_idx = self.c_mult_idx1
            self.v0.c_mult_idx = self.c_mult_idx1
            self.v1.c_mult_idx = self.c_mult_idx1
            self.v2.c_mult_idx = self.c_mult_idx1
            self.v3.c_mult_idx = self.c_mult_idx1
            qk = self.qk(x)
            if self.heads == 0:
                v = self.v0(x)
                self.set_grad_v(v0 = False)
            elif self.heads == 1:
                v = self.v1(x)
                self.set_grad_v(v1 = False)
            elif self.heads == 2:
                v = self.v2(x)
                self.set_grad_v(v2 = False)
            elif self.heads == 3:
                v = self.v3(x)
                self.set_grad_v(v3 = False)
            else:
                raise RuntimeError(f'heads:{self.heads}')
            qkv = torch.cat((qk,v), dim = 2)

        elif self.qkv_value == 2:
            self.qk0.c_mult_idx = self.c_mult_idx1
            self.qk1.c_mult_idx = self.c_mult_idx1
            self.qk2.c_mult_idx = self.c_mult_idx1
            self.qk3.c_mult_idx = self.c_mult_idx1
            self.v.c_mult_idx = self.c_mult_idx1
            v = self.v(x)
            if self.heads == 0:
                qk = self.qk0(x)
                self.set_grad_qk(qk0 = False)
            elif self.heads == 1:
                qk = self.qk1(x)
                self.set_grad_qk(qk1 = False)
            elif self.heads == 2:
                qk = self.qk2(x)
                self.set_grad_qk(qk2 = False)
            elif self.heads == 3:
                qk = self.qk3(x)
                self.set_grad_qk(qk3 = False)

            else:
                raise RuntimeError(f'heads:{self.heads}')
            qkv = torch.cat((qk,v), dim = 2)

        B, N, C = qkv.shape
        qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads // 3).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C // 3)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def set_grad_v(self, v0 = True, v1 = True, v2 = True, v3 = True):
        if v0:
            for param in self.v0.parameters():
                param.requires_grad = False
        if v1:
            for param in self.v1.parameters():
                param.requires_grad = False
        if v2:
            for param in self.v2.parameters():
                param.requires_grad = False
        if v3:
            for param in self.v3.parameters():
                param.requires_grad = False

    def set_grad_qk(self, qk0 = True, qk1 = True, qk2 = True, qk3 = True):
        if qk0:
            for param in self.qk0.parameters():
                param.requires_grad = False
        if qk1:
            for param in self.qk1.parameters():
                param.requires_grad = False
        if qk2:
            for param in self.qk2.parameters():
                param.requires_grad = False
        if qk3:
            for param in self.qk3.parameters():
                param.requires_grad = False




class DynamicLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.side = None

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        if C != self.normalized_shape[0]:
            assert self.side is not None, "need to choose which side to eval"
            if self.side == 'left':
                w = self.weight[:C].contiguous()
                b = self.bias[:C].contiguous() if self.bias is not None else None
            elif self.side == 'right':
                w = self.weight[(self.normalized_shape[0] - C) : ].contiguous()
                b = self.bias[(self.normalized_shape[0] - C) : ].contiguous()
            shape = torch.Size((C,))
            return F.layer_norm(x, shape, w, b, self.eps)
        else:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        



class Norm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        self.classifier = kwargs.pop('classifier','token')
        super().__init__(*args, **kwargs)
        self.side = None

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        if C != self.normalized_shape[0]:
            assert self.side is not None, "need to choose which side to eval"
            if self.side == 'left':
                w = self.weight[:C].contiguous()
                b = self.bias[:C].contiguous() if self.bias is not None else None
            elif self.side == 'right':
                w = self.weight[(self.normalized_shape[0]-C):].contiguous()
                b = self.bias[(self.normalized_shape[0]-C):].contiguous() if self.bias is not None else None
            else:
                raise RuntimeError("stop and check")
            shape =  torch.Size((C,))
            x = F.layer_norm(x, shape, w, b, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        if self.classifier == 'token':
            return x[:, 0]
        elif self.classifier == 'mean':
            return x[:, 1:].mean(dim=1)
        else: 
            raise RuntimeError("not support classifier: {}".format(self.classifier))



class Block(nn.Module):
    def __init__(self, dim, num_heads, at_temp=None, at_out= None, MLP_temp= None, MLP_out=None,  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=partial(DynamicLayerNorm, eps=1e-6), qkv = 0, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, at_temp, at_out, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, qkv = qkv)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        #mlp_hidden_dim = int(dim * mlp_ratio)
        self.qkv = qkv
        self.mlp = Mlp(in_features=dim, hidden_features=MLP_temp, out_features=MLP_out, act_layer=act_layer, drop=drop)

        self.c_mult_idx1 = None
        self.c_mult_idx2 = None
        self.c_mult_idx3 = None
        self.c_mult_idx4 = None
        self.side = None
        self.heads = None

    def forward(self, x):
        for i in self.modules():
            i.side = self.side
        self.attn.c_mult_idx1 = self.c_mult_idx1
        self.attn.c_mult_idx2 = self.c_mult_idx2
        self.mlp.c_mult_idx1 = self.c_mult_idx3
        self.mlp.c_mult_idx2 = self.c_mult_idx4
        self.attn.heads = self.heads

        B, N, C = x.shape
        out1 = self.drop_path(self.attn(self.norm1(x)))
        B1, N1, C1 = out1.shape
        out1 = out1 + x
        
        out2 = self.drop_path(self.mlp(self.norm2(out1)))
        B2, N2, C2 = out2.shape
        out2 = out2 + out1

        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return out2





class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if isinstance(embed_dim, list):
            embed_temp = embed_dim[-1]
            assert embed_temp == max(embed_dim), f'embed_temp: {embed_temp} should eq to max of embed: {embed_dim}'
        elif isinstance(embed_dim, int):
            embed_temp = embed_dim
        else:
            raise RuntimeError("not support embed: {}".format(embed_dim))

        self.proj = DynamicConv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.c_mult_idx = None
        self.side = None

    def forward(self, x):
        self.proj.c_mult_idx = self.c_mult_idx
        self.proj.side = self.side

        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# custom conv2d for channel search
class DynamicConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.channel_search = kwargs.pop('channel_search', False)
        super().__init__(*args, **kwargs)
        self.out_c = self.out_channels
        self.c_mult_idx = None
        self.side = None

    def forward(self, input, **kwargs):
        B, C, H, W = input.shape
        if self.c_mult_idx is not None:
            assert self.c_mult_idx>=0 and self.c_mult_idx<=1, "c_mult_idx should in range [0, 1], it is {}".format(self.c_mult_idx)
            out_c = round(self.c_mult_idx * self.out_channels)

        else:
            out_c = self.out_channels
        
        # train supernet
        if out_c != self.out_channels or C != self.in_channels:
            in_c = C
            #out_c = int(self.out_channels / max(channel_mults) * random.choice(channel_mults))
            assert self.side is not None, "need to choose which side to eval"
            if self.side == 'left':
                w = self.weight[:out_c, :in_c].contiguous()
                b = self.bias[:out_c].contiguous() if self.bias is not None else None
            elif self.side == 'right':
                w = self.weight[(self.out_channels - out_c):, (self.in_channels-in_c):].contiguous()
                b = self.bias[(self.out_channels - out_c):].contiguous() if self.bias is not None else None
            return F.conv2d(input, w, b, self.stride,
                            self.padding, self.dilation, self.groups)

        # train searched architecture
        else:
            return super().forward(input)

class DynamicLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.channel_search = kwargs.pop('channel_search', False)
        super().__init__(*args, **kwargs)
        self.c_mult_idx = None
        self.side = None


    def forward(self, input):
        if self.c_mult_idx is not None:
            assert self.c_mult_idx>=0 and self.c_mult_idx<=1, "c_mult_idx should in range [0,1], it is {}".format(self.c_mult_idx)
            out = round(self.c_mult_idx * self.out_features)
            self.c_mult_idx = None
        else:
            out = self.out_features
        in_dim = input.shape[-1]
        if out != self.out_features or in_dim != self.in_features:
            if self.side == 'left':
                w = self.weight[:out, :in_dim].contiguous()
                b = self.bias[:out].contiguous() if self.bias is not None else None
            elif self.side == 'right':
                w = self.weight[(self.out_features-out):, (self.in_features-in_dim):].contiguous()
                b = self.bias[(self.out_features-out):].contiguous() if self.bias is not None else None
            return F.linear(input, w, b)
        else:
            return super().forward(input)




class Patch_init(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, drop_rate = 0, **kwargs):
        super().__init__()

        self.patch_embed = PatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.c_mult_idx = None
        self.side = None

    def forward(self, x):
        self.patch_embed.c_mult_idx = self.c_mult_idx
        self.patch_embed.side = self.side


        x = self.patch_embed(x)
        B = x.shape[0]
        D = x.shape[2]

        if self.side == 'left':
            cls_tokens = self.cls_token[0,0,:D].reshape(1,1,D).expand(B, -1,-1)
            pos_embed = self.pos_embed[0, -1, :D].reshape(1, -1, D)
        elif self.side == 'right':
            cls_tokens = self.cls_token[0,0,self.embed_dim-D : ].reshape(1,1,D).expand(B, -1, -1)
            pos_embed = self.pos_embed[0, -1, self.embed_dim-D :].reshape(1, -1, D)
        else:
            cls_tokens = self.cls_token.expand(B, -1, -1)   
            pos_embed = self.pos_embed

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)
        return x


class Structure_blocks(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=1, act='nn.ReLU'):
        super(Structure_blocks, self).__init__()

        self.fc1 = FC(in_channel, int(in_channel//reduction), False, 0, act='nn.ReLU')
        self.fc2 = FC(int(in_channel//reduction), out_channel, False, 0, None)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x


class Parameter_module(nn.Module):
    def __init__(self, size, n=False):
        super(Parameter_module, self).__init__()
        self.size = size
        self.n = n
        self.params = nn.parameter.Parameter(torch.rand(*size))
        self.init()

    def init(self):
        # weights
        if len(self.size) == 4:
            self.params.data.normal_(0, math.sqrt(2. / self.n))
        elif len(self.size) == 1:
            if self.n == 1:
                self.params.data.fill_(1)
            elif self.n == 0:
                self.params.data.zero_()
            else:
                raise RuntimeError("not supp n——{}".format(self.n))
        else:
            raise RuntimeError("not supp size——{}".format(self.size))






class DynamicBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        self.channel_search = kwargs.pop('channel_search', True)
        super().__init__(*args, **kwargs)

    def forward(self, input):
        if self.channel_search:
            in_c = input.shape[1]

            self._check_input_dim(input)
            # exponential_average_factor is self.momentum set to
            # (when it is available) only so that if gets updated
            # in ONNX graph when this node is exported to ONNX.
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum
            w = self.weight[:in_c].contiguous()
            b = self.bias[:in_c].contiguous() if self.bias is not None else None
            r_mean = self.running_mean[:in_c].contiguous()
            r_var = self.running_var[:in_c].contiguous()
            return F.batch_norm(
                input, r_mean, r_var, w, b,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        else:
            return super().forward(input)




def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class SNetInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, conv_list, max_ks, channel_search):
        super(SNetInvertedResidual, self).__init__()
        self.stride = stride
        self.channel_search = channel_search
        self.c_mult_idx = None
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.stride == 1:
            branch2 = []
            for idx, conv_ks in enumerate(conv_list):  # last pw conv can not use channel search
                branch2.append(DynamicConv2d(oup_inc, oup_inc, conv_ks, padding=(conv_ks - 1) // 2,
                                         groups=oup_inc if conv_ks != 1 else 1, bias=False, channel_search=channel_search if idx != len(conv_list) - 1 else False))
                if idx == len(conv_list) - 1:
                    setattr(branch2[-1], 'last_pw', True)
                branch2.append(DynamicBatchNorm2d(oup_inc, channel_search=channel_search if idx != len(conv_list) - 1 else False))
                if conv_ks == 1:
                    branch2.append(nn.ReLU(inplace=True))
            self.branch2 = nn.Sequential(*branch2)
        else:
            self.branch1 = nn.Sequential(
                # dw
                DynamicConv2d(inp, inp, max_ks, stride=2, padding=(max_ks - 1) // 2,
                              groups=inp, bias=False, channel_search=channel_search),
                DynamicBatchNorm2d(inp, channel_search=channel_search),
                # pw-linear
                DynamicConv2d(inp, oup_inc, 1, 1, 0, bias=False, channel_search=channel_search),
                DynamicBatchNorm2d(oup_inc, channel_search=channel_search),
                nn.ReLU(inplace=True),
            )

            branch2 = []
            first_pw = True
            first_down = True
            channel_num = inp
            for conv_ks in conv_list:
                if first_pw and conv_ks == 1:
                    branch2.append(DynamicConv2d(channel_num, oup_inc, conv_ks, stride=2 if conv_ks != 1 else 1,
                                             padding=(conv_ks - 1) // 2,
                                             groups=inp if conv_ks != 1 else 1, bias=False))
                    channel_num = oup_inc
                    first_pw = False
                elif first_down and conv_ks != 1:
                    branch2.append(DynamicConv2d(channel_num, channel_num, conv_ks,
                                             stride=2,
                                             padding=(conv_ks - 1) // 2,
                                             groups=channel_num if conv_ks != 1 else 1, bias=False))
                    first_down = False
                else:
                    branch2.append(DynamicConv2d(channel_num, channel_num, conv_ks,
                                             stride=1,
                                             padding=(conv_ks - 1) // 2,
                                             groups=channel_num if conv_ks != 1 else 1, bias=False))
                branch2.append(DynamicBatchNorm2d(channel_num))
                if conv_ks == 1:
                    branch2.append(nn.ReLU(inplace=True))
            self.branch2 = nn.Sequential(*branch2)

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if self.channel_search:
            assert self.c_mult_idx is not None
            for module in self.modules():
                if isinstance(module, DynamicConv2d):
                    module.out_c = int(module.out_channels * channel_mults[self.c_mult_idx])
                    if module.out_c % 2 == 1:
                        module.out_c += 1
                if getattr(module, 'last_pw', False):
                    module.out_c = x.shape[1] // 2
        if 1 == self.stride:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            b2 = self.branch2(x2)
            out = self._concat(x1, b2)
        elif 2 == self.stride:
            b1, b2 = self.branch1(x), self.branch2(x)
            out = self._concat(b1, b2)

        return channel_shuffle(out, 2)


'''
end ops for single path one shot net, based on shufflenetv2
'''

class FC(nn.Module):
    def __init__(self, dim_in, dim_out, use_bn, dp=0., act='nn.ReLU'):
        super(FC, self).__init__()
        self.oup = dim_out
        self.module = []
        #print(f'dim_in: {dim_in}, dim_out: {dim_out}')
        self.module.append(DynamicLinear(dim_in, dim_out))
        if use_bn:
            self.module.append(nn.BatchNorm1d(dim_out))
        if act is not None:
            self.module.append(eval(act)(inplace=True))
        if dp != 0:
            self.module.append(nn.Dropout(dp))
        self.module = nn.Sequential(*self.module)
        self.side = None

    def forward(self, x):
        self.module[0].side = self.side
        if x.dim() != 2:
            x = x.flatten(1)
        return self.module(x)


class BasicOp(nn.Module):

    def __init__(self, oup, **kwargs):
        super(BasicOp, self).__init__()
        self.oup = oup
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def get_output_channles(self):
        return self.oup



class Conv2d(BasicOp):
    def __init__(self, inp, oup, stride, k, activation=nn.ReLU, **kwargs):
        self.bn_weight_num = kwargs.pop('bn_weight_num', False)
        self.channel_search = kwargs.pop('channel_search', False)
        self.weight_num = kwargs.pop('weight_num', False)
        #self.structure_len = kwargs.pop('structure_len', False)
        super(Conv2d, self).__init__(oup, **kwargs)
        self.stride = stride
        self.k = k
        kwargs_weight = {'weight_num': self.weight_num, 'channel_search': self.channel_search}
        kwargs_bn = {'weight_num': self.bn_weight_num, 'channel_search': self.channel_search}
        self.conv = nn.Sequential(
            DynamicConv2d_multiconv2d(inp, oup, kernel_size=k, stride=stride, padding=k//2, bias=False, **kwargs_weight),
            DynamicBatchNorm2d_multiBN(oup, **kwargs_bn),
            activation()
        )
        self.structure_tensor = None

    def forward(self, x):
        if self.structure_tensor is not None:
            self.conv[0].structure_tensor = self.structure_tensor
            self.conv[1].structure_tensor = self.structure_tensor
        return self.conv(x)

'''

class Conv2d(BasicOp):
    def __init__(self, inp, oup, stride, k, activation=nn.ReLU, **kwargs):
        self.weight_num = kwargs.pop('weight_num', False)
        super(Conv2d, self).__init__(oup, **kwargs)
        self.stride = stride
        self.k = k
        if self.weight_num:
            oup_temp = oup * self.weight_num
        else:
            oup_temp = oup
        channel_search = kwargs.pop('channel_search', False)
        self.conv = nn.Sequential(
            DynamicConv2d(inp, oup_temp, kernel_size=k, stride=stride, padding=k//2, bias=False, channel_search=channel_search, weight_num = self.weight_num),
            DynamicBatchNorm2d(oup_temp, channel_search=channel_search, weight_num = self.weight_num),
            activation()
        )

    def forward(self, x):
        if self.weight_num:
            self.conv[0].free_index = torch.FloatTensor([0.33, 0.33, 0.33]).cuda()
            self.conv[1].free_index = torch.FloatTensor([0.33, 0.33, 0.33]).cuda()
        return self.conv(x)
'''

class LinearBottleneck(BasicOp):
    def __init__(self, inp, oup, stride, k, activation=nn.ReLU, **kwargs):
        super(LinearBottleneck, self).__init__(oup, **kwargs)
        channel_search = kwargs.pop('channel_search', False)

        neck_dim = oup // 4
        self.conv1 = DynamicConv2d(inp, neck_dim, kernel_size=1, stride=1, bias=False, channel_search=channel_search)
        self.bn1 = DynamicBatchNorm2d(neck_dim, channel_search=channel_search)
        self.act1 = activation()
        self.conv2 = DynamicConv2d(neck_dim, neck_dim, kernel_size=k, stride=stride, padding=k//2, bias=False, channel_search=channel_search)
        self.bn2 = DynamicBatchNorm2d(neck_dim, channel_search=channel_search)
        self.act2 = activation()
        self.conv3 = DynamicConv2d(neck_dim, oup, kernel_size=1, bias=False, channel_search=channel_search)
        self.bn3 = DynamicBatchNorm2d(oup, channel_search=channel_search)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)

        return out


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, inplace=self.inplace) / 6
        return out




class SqueezeExcite_multiSE(nn.Module):
    def __init__(self, in_channel,
                 reduction=4,
                 squeeze_act=nn.ReLU(inplace=True),
                 excite_act=HSigmoid(inplace=True), **kwargs):
        self.channel_search = kwargs.pop('channel_search', False)
        self.weight_num = kwargs.pop('weight_num', False)
        #self.structure_len = kwargs.pop('structure_len', False)
        super(SqueezeExcite_multiSE, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        kwargs_weight = {'weight_num': self.weight_num, 'channel_search': self.channel_search}
        self.squeeze_conv = DynamicConv2d_multiconv2d(in_channel, in_channel // reduction, 1, 1, padding=0, bias=True, **copy.deepcopy(kwargs_weight))
        '''
        self.squeeze_conv = nn.Conv2d(in_channels=in_channel,
                                      out_channels=in_channel // reduction,
                                      kernel_size=1,
                                      bias=True)
        '''
        self.squeeze_act = squeeze_act

        self.excite_conv = DynamicConv2d_multiconv2d(in_channel // reduction, in_channel, 1, 1, padding=0, bias=True, **copy.deepcopy(kwargs_weight))
        '''
        self.excite_conv = nn.Conv2d(in_channels=in_channel // reduction,
                                     out_channels=in_channel,
                                     kernel_size=1,
                                     bias=True)
        '''
        self.excite_act = excite_act
        self.structure_tensor = None

    def forward(self, inputs):
        if self.structure_tensor is not None:
            self.squeeze_conv.structure_tensor = self.structure_tensor
            self.excite_conv.structure_tensor = self.structure_tensor

        feature_pooling = self.global_pooling(inputs)
        feature_squeeze_conv = self.squeeze_conv(feature_pooling)
        feature_squeeze_act = self.squeeze_act(feature_squeeze_conv)
        feature_excite_conv = self.excite_conv(feature_squeeze_act)
        feature_excite_act = self.excite_act(feature_excite_conv)
        return inputs * feature_excite_act



'''
class SqueezeExcite(nn.Module):
    def __init__(self, in_channel,
                 reduction=4,
                 squeeze_act=nn.ReLU(inplace=True),
                 excite_act=HSigmoid(inplace=True)):
        super(SqueezeExcite, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.squeeze_conv = nn.Conv2d(in_channels=in_channel,
                                      out_channels=in_channel // reduction,
                                      kernel_size=1,
                                      bias=True)
        self.squeeze_act = squeeze_act
        self.excite_conv = nn.Conv2d(in_channels=in_channel // reduction,
                                     out_channels=in_channel,
                                     kernel_size=1,
                                     bias=True)
        self.excite_act = excite_act

    def forward(self, inputs):
        feature_pooling = self.global_pooling(inputs)
        feature_squeeze_conv = self.squeeze_conv(feature_pooling)
        feature_squeeze_act = self.squeeze_act(feature_squeeze_conv)
        feature_excite_conv = self.excite_conv(feature_squeeze_act)
        feature_excite_act = self.excite_act(feature_excite_conv)
        return inputs * feature_excite_act
'''



class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, t, k=3, activation=nn.ReLU, use_se=False, channels=None, **kwargs):
        # this place super why have channels and **kwargs without others
        super(InvertedResidual, self).__init__()
        #print('ir inp: {}, channels: {}'.format(inp, channels))
        self.stride = stride
        self.t = t
        self.k = k
        self.use_se = use_se
        assert stride in [1, 2]
        assert 'weight_num' in kwargs, "weight_num should in kwargs, kwargs is {}".format(kwargs)
        self.weight_num = kwargs.get('weight_num', False)
        self.bn_weight_num = kwargs.get('bn_weight_num', False)
        channel_search = kwargs.get('channel_search', False)
        #self.structure_len = kwargs.get('structure_len', False)

        hidden_dim = round(inp * t)
        kwargs_weight = {'weight_num': self.weight_num, 'channel_search': channel_search}
        kwargs_bn = {'weight_num': self.bn_weight_num, 'channel_search': channel_search}
        if t == 1:
            self.conv = nn.Sequential(
                # dw            
                DynamicConv2d_multiconv2d(inp, hidden_dim, k, stride, padding=k//2, groups=hidden_dim, bias=False, **copy.deepcopy(kwargs_weight)),
                DynamicBatchNorm2d_multiBN(hidden_dim, **copy.deepcopy(kwargs_bn)),
                activation(inplace=True),
                # se
                SqueezeExcite_multiSE(hidden_dim, **copy.deepcopy(kwargs_bn)) if use_se else nn.Sequential(),
                # pw-linear
                DynamicConv2d_multiconv2d(hidden_dim, oup, 1, 1, 0, bias=False, **copy.deepcopy(kwargs_weight)),
                DynamicBatchNorm2d_multiBN(oup, **copy.deepcopy(kwargs_bn))
            )
        else:
            self.conv = nn.Sequential(
                # pw
                DynamicConv2d_multiconv2d(inp, hidden_dim, 1, 1, 0, bias=False, **copy.deepcopy(kwargs_weight)),
                DynamicBatchNorm2d_multiBN(hidden_dim, **copy.deepcopy(kwargs_bn)),
                activation(inplace=True),
                # dw
                DynamicConv2d_multiconv2d(hidden_dim, hidden_dim, k, stride, padding=k//2, groups=hidden_dim, bias=False, **copy.deepcopy(kwargs_weight)),
                DynamicBatchNorm2d_multiBN(hidden_dim, **copy.deepcopy(kwargs_bn)),
                activation(inplace=True),
                # se
                SqueezeExcite_multiSE(hidden_dim, **copy.deepcopy(kwargs_bn)) if use_se else nn.Sequential(),
                # pw-linear
                DynamicConv2d_multiconv2d(hidden_dim, oup, 1, 1, 0, bias=False, **copy.deepcopy(kwargs_weight)),
                DynamicBatchNorm2d_multiBN(oup, **copy.deepcopy(kwargs_bn))
            )
        self.use_shortcut = inp == oup and stride == 1
        self.structure_tensor = None


    def forward(self, x, **kwargs):
        if self.structure_tensor is not None:
            for i in self.conv:
                if isinstance(i, DynamicConv2d_multiconv2d) or isinstance(i, DynamicBatchNorm2d_multiBN) or isinstance(i, SqueezeExcite_multiSE):
                    i.structure_tensor = self.structure_tensor

        if self.use_shortcut:
            return self.conv(x) + x
        return self.conv(x)




class NormalResidual(BasicOp):
    def __init__(self, inp, oup, stride, t, k=3, activation=nn.ReLU, **kwargs):
        super(NormalResidual, self).__init__(oup, **kwargs)
        self.stride = stride
        assert stride in [1, 2]
        channel_search = kwargs.pop('channel_search', False)
        hidden_dim = round(inp * t)
        self.conv = nn.Sequential(
            # pw
            DynamicConv2d(inp, hidden_dim, 1, 1, 0, bias=False, channel_search=channel_search),
            DynamicBatchNorm2d(hidden_dim, channel_search=channel_search),
            activation(inplace=True),
            # dw
            DynamicConv2d(hidden_dim, hidden_dim, k, stride, padding=k//2, groups=1, bias=False, channel_search=channel_search),
            DynamicBatchNorm2d(hidden_dim, channel_search=channel_search),
            activation(inplace=True),
            # pw-linear
            DynamicConv2d(hidden_dim, oup, 1, 1, 0, bias=False, channel_search=channel_search),
            DynamicBatchNorm2d(oup, channel_search=channel_search),
        )

    def forward(self, x):

        return self.conv(x)


class DepthwiseSeparableConv(BasicOp):
    def __init__(self, inp, oup, stride, k=3, activation=nn.ReLU, **kwargs):
        super(DepthwiseSeparableConv, self).__init__(oup, **kwargs)
        self.stride = stride
        assert stride in [1, 2]
        channel_search = kwargs.pop('channel_search', False)
        self.conv_dw = nn.Sequential(
            DynamicConv2d(inp, inp, k, stride, groups=inp, bias=False, padding=k//2, channel_search=channel_search),
            DynamicBatchNorm2d(inp, eps=1e-10, momentum=0.05, channel_search=channel_search),
            activation()
        )

        self.conv_pw = nn.Sequential(
            DynamicConv2d(inp, oup, 1, 1, bias=False, channel_search=channel_search),
            DynamicBatchNorm2d(oup, eps=1e-10, momentum=0.05, channel_search=channel_search),
        )

    def forward(self, x, drop_connect_rate=None):
        x = self.conv_dw(x)
        #if self.has_se:
        #    x = self.se(x)
        x = self.conv_pw(x)
        return x


class DualBlock(BasicOp):
    def __init__(self, inp, oup, stride, t, k=3, activation=nn.ReLU, **kwargs):
        super(DualBlock, self).__init__(oup, **kwargs)
        padding = k // 2
        channel_search = kwargs.pop('channel_search', False)
        self.conv1 = DynamicConv2d(inp, inp * t, kernel_size=1, bias=False, channel_search=channel_search)
        self.bn1 = DynamicBatchNorm2d(inp * t, channel_search=channel_search)
        self.activation1 = activation()
        self.conv2_1 = DynamicConv2d(inp * t, inp * t, kernel_size=k, stride=1, padding=padding, bias=False, channel_search=channel_search)
        self.bn2_1 = DynamicBatchNorm2d(inp * t, channel_search=channel_search)
        self.conv2_2 = DynamicConv2d(inp * t, inp * t, kernel_size=k, stride=stride, padding=padding, bias=False, channel_search=channel_search)
        self.bn2_2 = DynamicBatchNorm2d(inp * t, channel_search=channel_search)
        self.activation2 = activation()
        self.conv3 = DynamicConv2d(inp * t, oup, kernel_size=1, bias=False, channel_search=channel_search)
        self.bn3 = DynamicBatchNorm2d(oup, channel_search=channel_search)
        self.activation3 = activation()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.activation2(out)

        out = self.conv2_2(out)
        out = self.bn2_2(out)
        out = self.activation3(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class RecBlock(BasicOp):
    def __init__(self, inp, oup, stride, t, k=3, activation=nn.ReLU, **kwargs):
        super(RecBlock, self).__init__(oup, **kwargs)
        padding = k // 2
        self.time = 0
        channel_search = kwargs.pop('channel_search', False)
        self.conv1 = DynamicConv2d(inp, inp * t, kernel_size=1, bias=False, channel_search=channel_search)
        self.bn1 = DynamicBatchNorm2d(inp * t, channel_search=channel_search)
        self.activation1 = activation()
        self.conv2_1 = DynamicConv2d(inp * t, inp * t, kernel_size=(1, k), stride=(1, stride),
                                 padding=(0, padding), bias=False, channel_search=channel_search)
        self.bn2_1 = DynamicBatchNorm2d(inp * t, channel_search=channel_search)
        self.activation2 = activation()
        self.conv2_2 = DynamicConv2d(inp * t, inp * t, kernel_size=(k, 1), stride=(stride, 1),
                                 padding=(padding, 0), bias=False, channel_search=channel_search)
        self.bn2_2 = DynamicBatchNorm2d(inp * t, channel_search=channel_search)
        self.activation3 = activation()
        self.conv3 = DynamicConv2d(inp * t, oup, kernel_size=1, bias=False, channel_search=channel_search)
        self.bn3 = DynamicBatchNorm2d(oup, channel_search=channel_search)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.activation2(out)

        out = self.conv2_2(out)
        out = self.bn2_2(out)
        out = self.activation3(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    model = DynamicConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2, channel_search=True)
    print(model(torch.zeros(3, 128, 112, 112)).shape)

    linear = DynamicLinear(in_features=512, out_features=1000, channel_search=True)
    print(linear(torch.zeros(3, 256)).shape)

    bn = DynamicBatchNorm2d(num_features=512, channel_search=False)
    print(bn(torch.zeros(3, 512, 112, 112)).shape)

