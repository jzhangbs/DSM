import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Callable, Any, Union
from collections import OrderedDict
import numpy as np
import itertools


def multi_dims(func: Callable,
               input_: torch.Tensor,
               dim: List[int],
               keepdim: bool,
               **kwargs) -> torch.Tensor:

    num_dims = len(input_.size())
    other_dims = list(range(num_dims))
    for d in dim:
        other_dims.remove(d)
    transpose_order = dim + other_dims
    inverse = [0]*num_dims
    for i, d in enumerate(transpose_order):
        inverse[d] = i
    size = np.array(input_.size())
    input_ = input_.permute(*transpose_order).contiguous()
    input_ = input_.view(np.product(size[dim]), *size[other_dims])
    is_reduce = keepdim is not None
    keepdim = keepdim is True
    if is_reduce:
        kwargs['keepdim'] = keepdim
    input_ = func(input_, dim=0, **kwargs)
    if keepdim or not is_reduce:
        if is_reduce:
            size[dim] = 1
            input_ = input_.view(*size)
        else:
            input_ = input_.view(*size[transpose_order])
            input_ = input_.permute(*inverse).contiguous()
    return input_


class ListModule(nn.Module):
    def __init__(self, modules: Union[List, OrderedDict]):
        super(ListModule, self).__init__()
        if isinstance(modules, OrderedDict):
            iterable = modules.items()
        elif isinstance(modules, list):
            iterable = enumerate(modules)
        else:
            raise TypeError('modules should be OrderedDict of List.')
        for name, module in iterable:
            if not isinstance(module, nn.Module):
                module = ListModule(module)
            if not isinstance(name, str):
                name = str(name)
            self.add_module(name, module)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dim=2):
        super(BasicBlock, self).__init__()

        self.conv_fn = nn.Conv2d if dim == 2 else nn.Conv3d
        self.bn_fn = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d

        self.conv1 = self.conv3x3(inplanes, planes, stride)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = self.bn_fn(planes)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = self.bn_fn(planes)
        nn.init.constant_(self.bn2.weight, 0)
        nn.init.constant_(self.bn2.bias, 0)
        self.downsample = downsample
        self.stride = stride

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return self.conv_fn(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return self.conv_fn(in_planes, out_planes, kernel_size=3, stride=stride,
                            padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def channel_shuffle(x, group=2):
    n, c, *spatial = x.size()
    return x.view(n, c//group, group, *spatial).transpose(1, 2).reshape(n, c, *spatial)


class ShuffleBlock(nn.Module):

    def __init__(self, planes, interm_scale=1., dim=2):
        super(ShuffleBlock, self).__init__()

        conv_fn = nn.Conv2d if dim == 2 else nn.Conv3d
        bn_fn = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d

        self.planes = planes
        self.interm_planes = int(planes//2*interm_scale)

        self.conv1 = conv_fn(self.planes//2, self.interm_planes, 1, 1, 0, 1, 1, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = bn_fn(self.interm_planes)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv_fn(self.interm_planes, self.interm_planes, 3, 1, 1, 1, self.interm_planes, bias=False)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = bn_fn(self.interm_planes)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
        self.conv3 = conv_fn(self.interm_planes, self.planes//2, 1, 1, 0, 1, 1, bias=False)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.bn3 = bn_fn(self.planes // 2)
        nn.init.constant_(self.bn3.weight, 1)
        nn.init.constant_(self.bn3.bias, 0)
        self.relu3 = nn.ReLU(inplace=True)
        self.layers = nn.Sequential(self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.conv3, self.bn3, self.relu3)

    def forward(self, x):
        branch1 = x[:, :self.planes//2, ...]
        branch2 = x[:, self.planes//2:, ...]
        branch2 = self.layers(branch2)
        concat = torch.cat([branch1, branch2], dim=1)
        shuffle = channel_shuffle(concat)
        return shuffle


class ShuffleBlockDownsize(nn.Module):

    def __init__(self, inplanes, planes, interm_scale=1., dim=2):
        super(ShuffleBlockDownsize, self).__init__()

        conv_fn = nn.Conv2d if dim == 2 else nn.Conv3d
        bn_fn = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d

        self.inplanes = inplanes
        self.planes = planes
        self.interm_planes = int(inplanes*interm_scale)

        self.conv1 = conv_fn(self.inplanes, self.interm_planes, 1, 1, 0, 1, 1, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = bn_fn(self.interm_planes)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv_fn(self.interm_planes, self.interm_planes, 3, 2, 1, 1, self.interm_planes, bias=False)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = bn_fn(self.interm_planes)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
        self.conv3 = conv_fn(self.interm_planes, self.planes//2, 1, 1, 0, 1, 1, bias=False)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.bn3 = bn_fn(self.planes//2)
        nn.init.constant_(self.bn3.weight, 1)
        nn.init.constant_(self.bn3.bias, 0)
        self.relu3 = nn.ReLU(inplace=True)
        self.branch2_layers = nn.Sequential(self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.conv3, self.bn3, self.relu3)

        self.conv4 = conv_fn(self.inplanes, self.inplanes, 3, 2, 1, 1, self.inplanes, bias=False)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.bn4 = bn_fn(self.inplanes)
        nn.init.constant_(self.bn4.weight, 1)
        nn.init.constant_(self.bn4.bias, 0)
        self.conv5 = conv_fn(self.inplanes, self.planes//2, 1, 1, 0, 1, 1, bias=False)
        nn.init.xavier_uniform_(self.conv5.weight)
        self.bn5 = bn_fn(self.planes//2)
        nn.init.constant_(self.bn5.weight, 1)
        nn.init.constant_(self.bn5.bias, 0)
        self.relu5 = nn.ReLU(inplace=True)
        self.branch1_layers = nn.Sequential(self.conv4, self.bn4, self.conv5, self.bn5, self.relu5)

    def forward(self, x):
        branch1 = self.branch1_layers(x)
        branch2 = self.branch2_layers(x)
        concat = torch.cat([branch1, branch2], dim=1)
        shuffle = channel_shuffle(concat)
        return shuffle


def _make_layer(inplanes, block, planes, blocks, stride=1, dim=2):
    downsample = None
    conv_fn = nn.Conv2d if dim==2 else nn.Conv3d
    bn_fn = nn.BatchNorm2d if dim==2 else nn.BatchNorm3d
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv_fn(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            bn_fn(planes * block.expansion)
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, dim=dim))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, dim=dim))

    return nn.Sequential(*layers)


def _make_layer_shuffle(inplanes, block, planes, blocks, stride=1, dim=2):
    layers = [ShuffleBlockDownsize(inplanes, planes, dim=dim) if stride!=1 else ShuffleBlock(planes, dim=dim)]
    for i in range(1, blocks):
        layers.append(ShuffleBlock(planes, dim=dim))
    return nn.Sequential(*layers)


class UNet(nn.Module):

    def __init__(self, inplanes: int, enc: int, dec: int, initial_scale: int,
                 bottom_filters: List[int], filters: List[int], head_filters: List[int],
                 prefix: str, dim: int=2):
        super(UNet, self).__init__()

        conv_fn = nn.Conv2d if dim==2 else nn.Conv3d
        bn_fn = nn.BatchNorm2d if dim==2 else nn.BatchNorm3d
        deconv_fn = nn.ConvTranspose2d if dim==2 else nn.ConvTranspose3d
        current_scale = initial_scale
        idx = 0
        prev_f = inplanes

        self.bottom_blocks = OrderedDict()
        for f in bottom_filters:
            block = _make_layer(prev_f, BasicBlock, f, enc, 1 if idx==0 else 2, dim=dim)
            self.bottom_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale *= 2
            prev_f = f
        self.bottom_blocks = ListModule(self.bottom_blocks)

        self.enc_blocks = OrderedDict()
        for f in filters:
            block = _make_layer(prev_f, BasicBlock, f, enc, 1 if idx == 0 else 2, dim=dim)
            self.enc_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale *= 2
            prev_f = f
        self.enc_blocks = ListModule(self.enc_blocks)

        self.dec_blocks = OrderedDict()
        for f in filters[-2::-1]:
            block = [
                deconv_fn(prev_f, f, 3, 2, 1, 1, bias=False),
                conv_fn(2*f, f, 3, 1, 1, bias=False),
                _make_layer(f, BasicBlock, f, dec, 1, dim=dim)
            ]
            nn.init.xavier_uniform_(block[0].weight)
            nn.init.xavier_uniform_(block[1].weight)
            self.dec_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale //= 2
            prev_f = f
        self.dec_blocks = ListModule(self.dec_blocks)

        self.head_blocks = OrderedDict()
        for f in head_filters:
            block = nn.Sequential(
                deconv_fn(prev_f, f, 3, 2, 1, 1, bias=False),
                _make_layer(f, BasicBlock, f, dec, 1, dim=dim)
            )
            nn.init.xavier_uniform_(block[0])
            self.head_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale //= 2
            prev_f = f
        self.head_blocks = ListModule(self.head_blocks)

    def forward(self, x):
        for b in self.bottom_blocks:
            x = b(x)
        enc_out = []
        for b in self.enc_blocks:
            x = b(x)
            enc_out.append(x)
        for i, b in enumerate(self.dec_blocks):
            deconv, post_concat, b = b
            x = deconv(x)
            x = torch.cat([x, enc_out[-2-i]], 1)
            x = post_concat(x)
            x = b(x)
        for b in self.head_blocks:
            x = b(x)
        return x


class CSPN(nn.Module):

    def __init__(self, kernel_size, iteration, affinity_net, dim=2):
        super(CSPN, self).__init__()

        self.kernel_size = kernel_size
        self.iteration = iteration
        self.affinity_net = affinity_net
        self.dim = dim

    def gen_kernel(self, x):
        abs_sum = torch.sum(torch.abs(x), dim=1, keepdim=True)
        x = x / abs_sum
        sum_ = torch.sum(x, dim=1, keepdim=True)
        out = torch.cat([(1 - sum_), x], dim=1)
        out = out.contiguous()
        return out

    def im2col(self, x):
        size = x.size()
        offsets = list(itertools.product([*range(self.kernel_size//2+1), *range(-(self.kernel_size//2), 0)], repeat=self.dim))
        out = torch.cuda.FloatTensor(size[0], len(offsets), *size[2:]).zero_()
        for k, o in enumerate(offsets):
            out[[slice(size[0])] + [k] + [slice(max(0, i), min(size[2+d], size[2+d] + i)) for d, i in enumerate(o)]] = \
                x[[slice(size[0])] + [0] + [slice(max(0, -i), min(size[2+d], size[2+d] - i)) for d, i in enumerate(o)]]
        out = out.contiguous()
        return out

    def forward(self, x):
        out = self.affinity_net(x)
        kernel = self.gen_kernel(out)
        for _ in range(self.iteration):
            x = torch.sum(self.im2col(x) * kernel, dim=1, keepdim=True)
        return x


class AMNet(nn.Module):

    def __init__(self, inplanes, out_planes, k=16, layer_per_scale=2, dim=2):
        super(AMNet, self).__init__()

        conv_fn = nn.Conv2d if dim == 2 else nn.Conv3d
        bn_fn = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d

        layers = []
        layers.append(conv_fn(inplanes, out_planes, 3, 1, 1, 1, bias=False))
        layers.append(bn_fn(out_planes))
        layers.append(nn.ReLU(inplace=True))
        curr_dil = 1
        while curr_dil <= k:
            for _ in range(layer_per_scale):
                layers.append(conv_fn(out_planes, out_planes, 3, 1, curr_dil, curr_dil, bias=False))
                layers.append(bn_fn(out_planes))
                layers.append(nn.ReLU(inplace=True))
            curr_dil *= 2
        layers.append(conv_fn(out_planes, out_planes, 3, 1, 1, 1, bias=False))
        layers.append(bn_fn(out_planes))
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if any([isinstance(m, T) for T in [nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d]]):
                nn.init.xavier_uniform_(m.weight)
            elif any([isinstance(m, T) for T in [nn.BatchNorm2d, nn.BatchNorm3d]]):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)


class AMNet2(nn.Module):

    def __init__(self, inplanes, out_planes=32, k=16, layer_per_scale=2, dim=2):
        super(AMNet2, self).__init__()

        conv_fn = nn.Conv2d if dim == 2 else nn.Conv3d
        bn_fn = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d

        self.init_conv = nn.Sequential(
            conv_fn(inplanes, out_planes, 3, 1, 1, 1, bias=False),
            bn_fn(out_planes),
            nn.ReLU(inplace=True)
        )
        layers = []
        curr_dil = 1
        num_scale = int(np.log(k)/np.log(2))
        while curr_dil <= k:
            scale = []
            for _ in range(layer_per_scale):
                scale.append(conv_fn(out_planes, out_planes//num_scale, 3, 1, 1, curr_dil, bias=False))
                scale.append(bn_fn(out_planes//num_scale))
                scale.append(nn.ReLU(inplace=True))
            layers.append(nn.Sequential(*scale))
            curr_dil *= 2
        self.layers = ListModule(layers)
        self.final_conv = nn.Sequential(
            conv_fn(out_planes//num_scale*num_scale, out_planes, 3, 1, 1, 1, bias=False),
            bn_fn(out_planes),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if any([isinstance(m, T) for T in [nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d]]):
                nn.init.xavier_uniform_(m.weight)
            elif any([isinstance(m, T) for T in [nn.BatchNorm2d, nn.BatchNorm3d]]):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.init_conv(x)
        scales = []
        for b in self.layers:
            scales.append(b(out))
        concat = torch.cat(scales, dim=1)
        out = self.final_conv(concat)
        return out
