"""
This started as a copy of https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py
below is the original docstring:

PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman
Copyright 2020 Ross Wightman
"""
import logging
import math
from pprint import pformat
from typing import List, Dict, Callable, Optional, Type, Union, Any

import torch.nn as nn
import torch.nn.functional as F

from .base import MetaClassifierBase
from .helpers import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ..layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, SEModule, EcaModule
from ...utils.weight_init import constant_init, kaiming_init

_logger = logging.getLogger(__name__)

__all__ = ['ResNet', 'BasicBlock', 'Bottleneck']  # model_registry will add each entrypoint fn to this


def _meta(**kwargs):
    default = {
        'url': '',
        'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        'paper': 'Deep Residual Learning for Image Recognition',
        'code': 'https://github.com/pytorch/vision/tree/master/torchvision/models'
    }
    default.update(kwargs)
    return default


def _cfg(**kwargs):
    default = {
        'block': Bottleneck,
        'num_classes': 1000,
        'in_channels': 3,
        'cardinality': 1,
        'base_width': 64,
        'stem_layer': BasicStem,
        'stem_kwargs': {},
        'output_stride': 32,
        'block_reduce_first': 1,
        'down_kernel_size': 1,
        'avg_down': False,
        'act_layer': nn.ReLU,
        'norm_layer': nn.BatchNorm2d,
        'aa_layer': None,
        'drop_rate': 0.,
        'drop_path_rate': 0.,
        'drop_block_rate': 0.,
        'global_pooling_layer': nn.AdaptiveAvgPool2d,
        'zero_init_residual': True,
        'block_kwargs': dict(

        )
    }
    default.update(kwargs)
    return default


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, attn_kwargs=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.aa = aa_layer(channels=first_planes, stride=stride) if use_aa else None

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = attn_layer(outplanes, **(attn_kwargs or dict())) if attn_layer else None

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, attn_kwargs=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = attn_layer(outplanes, **(attn_kwargs or dict())) if attn_layer else None

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act3(x)

        return x


class BasicStem(nn.Conv2d):
    def __init__(self, in_channels, **_):
        self.stem_out_channels = 64
        super(BasicStem, self).__init__(in_channels, self.stem_out_channels, kernel_size=7, stride=2, padding=3,
                                        bias=False)


class DeepStem(nn.Sequential):
    def __init__(
            self, in_channels, *,
            stem_width: int,
            norm_layer: nn.BatchNorm2d,
            act_layer: Union[Callable[..., nn.Module], Type[nn.Module]] = nn.ReLU,
            tiered=False, narrow=False, **_
    ):
        self.stem_out_channels = stem_width * 2
        stem_chs_1 = stem_chs_2 = stem_width
        if tiered:
            stem_chs_1 = 3 * (stem_width // 4)
            stem_chs_2 = stem_width if narrow else 6 * (stem_width // 4)

        super(DeepStem, self).__init__(*[
            nn.Conv2d(in_channels, stem_chs_1, 3, stride=2, padding=1, bias=False),
            norm_layer(stem_chs_1),
            act_layer(inplace=True),
            nn.Conv2d(stem_chs_1, stem_chs_2, 3, stride=1, padding=1, bias=False),
            norm_layer(stem_chs_2),
            act_layer(inplace=True),
            nn.Conv2d(stem_chs_2, self.stem_out_channels, 3, stride=1, padding=1, bias=False)
        ])


def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_block_rate=0.):
    return [
        None, None,
        DropBlock2d(drop_block_rate, 5, 0.25) if drop_block_rate else None,
        DropBlock2d(drop_block_rate, 3, 1.00) if drop_block_rate else None]


def make_blocks(
        block_fn, channels, block_repeats, inplanes, reduce_first=1, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., **kwargs):
    stages = []
    # feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        # feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages  # , feature_info


class ResNet(MetaClassifierBase):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block

    Parameters
    ----------
    variant: str
        variant name of the model
    block : Block
        Class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int
        Numbers of layers in each block
    num_classes : int, default 1000
        Number of classification classes.
    in_channels : int, default 3
        Number of input (color) channels.
    out_features : Optional[List[str]]
        Specify the intermediate features to be appended to the output dict.
    cardinality : int, default 1
        Number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64
        Factor determining bottleneck channels. `planes * base_width / 64 * cardinality`
    stem_layer : Union[Callable[..., nn.Module], Type[nn.Module]], default BasicStem
        Specify the stem layer (the first conv layer) in ResNet
    stem_kwargs : int, default 64
        Specify extra kwargs for the stem layer
        DeepStem:
          * 'stem_width' - base number of channels in stem layer
          * 'tiered (bool)' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width//4 * 6, stem_width * 2
          * 'narrow (bool)' - (use with tiered=True) three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
    block_reduce_first: int, default 1
        Reduction factor for first convolution output width of residual blocks,
        1 for all archs except senets, where 2
    down_kernel_size: int, default 1
        Kernel size of residual block downsampling path, 1x1 for most archs, 3x3 for senets
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    output_stride : int, default 32
        Set the output stride of the network, 32, 16, or 8. Typically used in segmentation.
    act_layer : Union[Callable[..., nn.Module], Type[nn.Module]], default nn.ReLU
        activation layer
    norm_layer : Union[Callable[..., nn.Module], Type[nn.Module]], default nn.BatchNorm2d
        normalization layer
    aa_layer : Optional[Union[Callable[..., nn.Module], Type[nn.Module]]],
        anti-aliasing layer
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pooling_layer : Union[Callable[..., nn.Module], Type[nn.Module]], default nn.AdaptiveAvgPool2d
        Global pooling type.
    zero_init_residual: bool, default 'True'
        Whether init
    block_kwargs: Optional[dict]
        extra kwargs for 'make_blocks'
            attn_layer : Optional[Union[Callable[..., nn.Module], Type[nn.Module]]],
                attention module used in each residual block
            attn_kwargs: Optional[Dict]
                extra kwargs for 'attn_layer'
    """

    def __init__(
            self,
            variant: str,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            in_channels: int = 3,
            *,
            out_features: Optional[List[str]] = None,
            cardinality: int = 1,
            base_width: int = 64,
            stem_layer: Union[Callable[..., nn.Module], Type[nn.Module]] = BasicStem,
            stem_kwargs: Optional[Dict[str, Any]] = None,
            output_stride: int = 32,
            block_reduce_first: int = 1,
            down_kernel_size: int = 1,
            avg_down: bool = False,
            act_layer: Union[Callable[..., nn.Module], Type[nn.Module]] = nn.ReLU,
            norm_layer: Union[Callable[..., nn.Module], Type[nn.Module]] = nn.BatchNorm2d,
            aa_layer: Optional[Union[Callable[..., nn.Module], Type[nn.Module]]] = None,
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.,
            drop_block_rate: float = 0.,
            global_pooling_layer: Union[Callable[..., nn.Module], Type[nn.Module]] = nn.AdaptiveAvgPool2d,
            zero_init_residual=True,
            block_kwargs=None
    ):
        block_kwargs = block_kwargs or dict()
        assert output_stride in (8, 16, 32)
        super(ResNet, self).__init__(
            variant,
            in_channels=in_channels,
            out_features=out_features,
            num_classes=num_classes
        )
        self.load_config(expose_in_self=True)
        _logger.info(f"building {self.variant} with configuration: \n{pformat(self.config)}")
        # Stem
        stem_kwargs = dict(stem_kwargs)
        stem_kwargs.setdefault("norm_layer", norm_layer)
        stem_kwargs.setdefault("act_layer", act_layer)
        self.conv1 = stem_layer(in_channels, **stem_kwargs)
        self.bn1 = norm_layer(self.conv1.stem_out_channels)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=self.conv1.stem_out_channels, reduction=2, module='act1')]

        # Stem Pooling
        if aa_layer is not None:
            self.maxpool = nn.Sequential(*[
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                aa_layer(channels=self.conv1.stem_out_channels, stride=2)])
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules = make_blocks(
            block, channels, layers, self.conv1.stem_out_channels, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_kwargs)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        # self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        if self.num_classes:
            self.num_features = 512 * block.expansion
            self.global_pool = global_pooling_layer
            self.fc = nn.Linear(self.num_features, self.num_classes, bias=True)
        else:
            self.add_module("global_pool", None)
            self.add_module("fc", None)

        self.init_weights()

    def init_weights(self, pretrained=False):
        if pretrained:
            self.load_pretrained()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                constant_init(m, 1)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.bn3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.bn2, 0)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_classifier(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


# pre-defined variant configurations
ResNet.variants = {
    # ResNet and Wide ResNet
    'resnet18': dict(
        config=_cfg(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
        ),
        meta=_meta(
            url='https://download.pytorch.org/models/resnet18-5c106cde.pth',
        )
    ),
    'resnet18d': dict(
        config=_cfg(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            stem_layer=DeepStem,
            stem_kwargs=dict(
                stem_width=32,
            ),
            avg_down=True
        ),
        meta=_meta(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet18d_ra2-48a79e06.pth',
            paper='Bag of Tricks for Image Classification with CNNs',
            code='https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/resnetv1b.py',
            interpolation='bicubic',
            first_conv='conv1.0'
        )
    ),
    'resnet34': dict(
        config=_cfg(
            block=BasicBlock,
            layers=[3, 4, 6, 3],
        ),
        meta=_meta(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth',
        )
    ),
    'resnet34d': dict(
        config=_cfg(
            block=BasicBlock,
            layers=[3, 4, 6, 3],
            stem_layer=DeepStem,
            stem_kwargs=dict(
                stem_width=32,
            ),
            avg_down=True
        ),
        meta=_meta(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34d_ra2-f8dcfcaf.pth',
            paper='Bag of Tricks for Image Classification with CNNs',
            code='https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/resnetv1b.py',
            interpolation='bicubic',
            first_conv='conv1.0'
        )
    ),
    'resnet50': dict(
        config=_cfg(
            layers=[3, 4, 6, 3],
        ),
        meta=_meta(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth',
            interpolation='bicubic',
        )
    ),
    'resnet50d': dict(
        config=_cfg(
            layers=[3, 4, 6, 3],
            stem_layer=DeepStem,
            stem_kwargs=dict(
                stem_width=32,
            ),
            avg_down=True
        ),
        meta=_meta(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth',
            paper='Bag of Tricks for Image Classification with CNNs',
            code='https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/resnetv1b.py',
            interpolation='bicubic',
            first_conv='conv1.0'
        )
    ),
    'resnet101': dict(
        config=_cfg(
            layers=[3, 4, 23, 3],
        ),
        meta=_meta(
            url='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        )
    ),
    'resnet101d': dict(
        config=_cfg(
            layers=[3, 4, 23, 3],
            stem_layer=DeepStem,
            stem_kwargs=dict(
                stem_width=32,
            ),
            avg_down=True
        ),
        meta=_meta(
            paper='Bag of Tricks for Image Classification with CNNs',
            code='https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/resnetv1b.py',
            interpolation='bicubic',
            first_conv='conv1.0'
        )
    ),
    'resnet152': dict(
        config=_cfg(
            layers=[3, 4, 23, 3],
        ),
        meta=_meta(
            url='https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        )
    ),
    'resnet152d': dict(
        config=_cfg(
            layers=[3, 4, 23, 3],
            stem_layer=DeepStem,
            stem_kwargs=dict(
                stem_width=32,
            ),
            avg_down=True
        ),
        meta=_meta(
            paper='Bag of Tricks for Image Classification with CNNs',
            code='https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/resnetv1b.py',
            interpolation='bicubic',
            first_conv='conv1.0'
        )
    ),
    'resnet200': dict(
        config=_cfg(
            layers=[3, 24, 36, 3],
        ),
        meta=_meta(
            interpolation='bicubic'
        )
    ),
    'resnet200d': dict(
        config=_cfg(
            layers=[3, 24, 36, 3],
            stem_layer=DeepStem,
            stem_kwargs=dict(
                stem_width=32,
            ),
            avg_down=True
        ),
        meta=_meta(
            paper='Bag of Tricks for Image Classification with CNNs',
            code='https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/resnetv1b.py',
            interpolation='bicubic',
            first_conv='conv1.0'
        )
    ),

    # ResNeXt
    'resnext50_32x4d': dict(
        config=_cfg(
            layers=[3, 4, 6, 3],
            cardinality=32,
            base_width=4
        ),
        meta=_meta(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50_32x4d_ra-d733960d.pth',
            interpolation='bicubic',
            paper="Aggregated Residual Transformations for Deep Neural Networks",
            code="https://github.com/pytorch/vision/tree/master/torchvision/models"
        )
    ),
    'resnext50d_32x4d': dict(
        config=_cfg(
            layers=[3, 4, 6, 3],
            stem_layer=DeepStem,
            stem_kwargs=dict(
                stem_width=32
            ),
            cardinality=32,
            base_width=4,
            avg_down=True
        ),
        meta=_meta(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50d_32x4d-103e99f8.pth',
            interpolation='bicubic',
            first_conv='conv1.0',
            paper="Aggregated Residual Transformations for Deep Neural Networks",
            code="https://github.com/pytorch/vision/tree/master/torchvision/models"
        )
    ),
    'resnext101_32x4d': dict(
        config=_cfg(
            layers=[3, 4, 23, 3],
            cardinality=32,
            base_width=4
        ),
        meta=_meta(
            paper="Aggregated Residual Transformations for Deep Neural Networks",
            paper_ssl="Billion-scale semi-supervised learning for image classification",
            paper_swsl="Billion-scale semi-supervised learning for image classification",
            # Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
            url_ssl='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth',
            url_swsl='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth',
        )
    ),
    'resnext101_32x8d': dict(
        config=_cfg(
            layers=[3, 4, 23, 3],
            cardinality=32,
            base_width=8
        ),
        meta=_meta(
            paper="Aggregated Residual Transformations for Deep Neural Networks",
            paper_ig="Exploring the Limits of Weakly Supervised Pretraining",
            paper_ssl="Billion-scale semi-supervised learning for image classification",
            paper_swsl="Billion-scale semi-supervised learning for image classification",
            url='https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
            # Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
            url_ig='https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
            url_ssl='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth',
            url_swsl='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth',
        )
    ),
    'resnext101_32x16d': dict(
        config=_cfg(
            layers=[3, 4, 23, 3],
            cardinality=32,
            base_width=16
        ),
        meta=_meta(
            paper="Aggregated Residual Transformations for Deep Neural Networks",
            paper_ig="Exploring the Limits of Weakly Supervised Pretraining",
            paper_ssl="Billion-scale semi-supervised learning for image classification",
            paper_swsl="Billion-scale semi-supervised learning for image classification",
            # Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
            url_ig='https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
            url_ssl='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x16-15fffa57.pth',
            url_swsl='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x16-f3559a9c.pth',
        )
    ),
    #  Squeeze-Excitation ResNets, to eventually replace the models in senet.py
    'seresnet50': dict(
        config=_cfg(
            layers=[3, 4, 6, 3],
            block_kwargs=dict(
                attn_layer=SEModule,
            ),
        ),
        meta=_meta(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet50_ra_224-8efdb4bb.pth',
            paper='Squeeze-and-Excitation Networks',
            interpolation='bicubic',
        )
    ),
    'seresnet101': dict(
        config=_cfg(
            layers=[3, 4, 23, 3],
            block_kwargs=dict(
                attn_layer=SEModule,
            ),
        ),
        meta=_meta(
            paper='Squeeze-and-Excitation Networks',
            interpolation='bicubic',
        )
    ),
    'seresnet152': dict(
        config=_cfg(
            layers=[3, 8, 36, 3],
            block_kwargs=dict(
                attn_layer=SEModule,
            ),
        ),
        meta=_meta(
            paper='Squeeze-and-Excitation Networks',
            interpolation='bicubic',
        )
    ),

    #  Squeeze-Excitation ResNeXts, to eventually replace the models in senet.py
    'seresnext26_32x4d': dict(
        config=_cfg(
            layers=[2, 2, 2, 2],
            cardinality=32,
            base_width=4,
            block_kwargs=dict(
                attn_layer=SEModule,
            ),
        ),
        meta=_meta(
            paper='Squeeze-and-Excitation Networks, Aggregated Residual Transformations for Deep Neural Networks',
            interpolation='bicubic',
        )
    ),
    'seresnext26d_32x4d': dict(
        # Constructs a SE-ResNeXt-26-D model.`
        # This is technically a 28 layer ResNet, using the 'D' modifier from Gluon / bag-of-tricks for
        # combination of deep stem and avg_pool in downsample.
        config=_cfg(
            layers=[2, 2, 2, 2],
            stem_layer=DeepStem,
            stem_kwargs=dict(
                stem_width=32,
            ),
            avg_down=True,
            cardinality=32,
            base_width=4,
            block_kwargs=dict(
                attn_layer=SEModule,
            ),
        ),
        meta=_meta(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26d_32x4d-80fa48a3.pth',
            paper='Squeeze-and-Excitation Networks, Aggregated Residual Transformations for Deep Neural Networks, Bag of Tricks for Image Classification with CNNs',
            interpolation='bicubic',
            first_conv='conv1.0'
        )
    ),
    'seresnext26t_32x4d': dict(
        # Constructs a SE-ResNet-26-T model.
        # This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 48, 64 channels
        # in the deep stem.
        config=_cfg(
            layers=[2, 2, 2, 2],
            stem_layer=DeepStem,
            stem_kwargs=dict(
                stem_width=32,
                tiered=True,
            ),
            avg_down=True,
            cardinality=32,
            base_width=4,
            block_kwargs=dict(
                attn_layer=SEModule,
            ),
        ),
        meta=_meta(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26t_32x4d-361bc1c4.pth',
            paper='Squeeze-and-Excitation Networks, Aggregated Residual Transformations for Deep Neural Networks, Bag of Tricks for Image Classification with CNNs',
            interpolation='bicubic',
            first_conv='conv1.0'
        )
    ),
    'seresnext26tn_32x4d': dict(
        # Constructs a SE-ResNeXt-26-TN model.
        # This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
        # in the deep stem. The channel number of the middle stem conv is narrower than the 'T' variant.
        config=_cfg(
            layers=[2, 2, 2, 2],
            stem_layer=DeepStem,
            stem_kwargs=dict(
                stem_width=32,
                tiered=True,
                narrow=True,
            ),
            avg_down=True,
            cardinality=32,
            base_width=4,
            block_kwargs=dict(
                attn_layer=SEModule,
            ),
        ),
        meta=_meta(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26tn_32x4d-569cb627.pth',
            paper='Squeeze-and-Excitation Networks, Aggregated Residual Transformations for Deep Neural Networks, Bag of Tricks for Image Classification with CNNs',
            interpolation='bicubic',
            first_conv='conv1.0'
        )
    ),
    'seresnext50_32x4d': dict(
        config=_cfg(
            layers=[3, 4, 6, 3],
            cardinality=32,
            base_width=4,
            block_kwargs=dict(
                attn_layer=SEModule,
            ),
        ),
        meta=_meta(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext50_32x4d_racm-a304a460.pth',
            paper='Squeeze-and-Excitation Networks, Aggregated Residual Transformations for Deep Neural Networks',
            interpolation='bicubic',
        )
    ),
    'seresnext101_32x4d': dict(
        config=_cfg(
            layers=[3, 4, 23, 3],
            cardinality=32,
            base_width=4,
            block_kwargs=dict(
                attn_layer=SEModule,
            ),
        ),
        meta=_meta(
            paper='Squeeze-and-Excitation Networks, Aggregated Residual Transformations for Deep Neural Networks',
            interpolation='bicubic',
        )
    ),
    'seresnext101_32x8d': dict(
        config=_cfg(
            layers=[3, 4, 23, 3],
            cardinality=32,
            base_width=8,
            block_kwargs=dict(
                attn_layer=SEModule,
            ),
        ),
        meta=_meta(
            paper='Squeeze-and-Excitation Networks, Aggregated Residual Transformations for Deep Neural Networks',
            interpolation='bicubic',
        )
    ),

    # Efficient Channel Attention ResNets
    'ecaresnet18': dict(
        config=_cfg(
            block=BasicBlock,
            block_kwargs=dict(
                attn_layer=EcaModule,
            ),
            layers=[2, 2, 2, 2],
        ),
        meta=_meta(
            paper='ECA-Net: Efficient Channel Attention for Deep CNN',
            code='https://github.com/BangguWu/ECANet',
        )
    ),
    'ecaresnet50': dict(
        config=_cfg(
            block_kwargs=dict(
                attn_layer=EcaModule,
            ),
            layers=[3, 4, 6, 3],
        ),
        meta=_meta(
            paper='ECA-Net: Efficient Channel Attention for Deep CNN',
            code='https://github.com/BangguWu/ECANet',
        )
    ),
    'ecaresnetlight': dict(
        config=_cfg(
            block_kwargs=dict(
                attn_layer=EcaModule,
            ),
            layers=[1, 1, 11, 3],
            stem_layer=DeepStem,
            stem_kwargs=dict(
                stem_width=32,
            ),
            avg_down=True,
        ),
        meta=_meta(
            url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45402/outputs/ECAResNetLight_4f34b35b.pth',
            paper='ECA-Net: Efficient Channel Attention for Deep CNN',
            code='https://github.com/BangguWu/ECANet',
            interpolation='bicubic'
        )
    ),
    'ecaresnet50d': dict(
        config=_cfg(
            block_kwargs=dict(
                attn_layer=EcaModule,
            ),
            layers=[3, 4, 6, 3],
            stem_layer=DeepStem,
            stem_kwargs=dict(
                stem_width=32,
            ),
            avg_down=True,
        ),
        meta=_meta(
            url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45402/outputs/ECAResNet50D_833caf58.pth',
            paper='ECA-Net: Efficient Channel Attention for Deep CNN',
            code='https://github.com/BangguWu/ECANet',
            interpolation='bicubic',
            first_conv='conv1.0'
        )
    ),
    'ecaresnet101d': dict(
        config=_cfg(
            block_kwargs=dict(
                attn_layer=EcaModule,
            ),
            layers=[3, 4, 23, 3],
            stem_layer=DeepStem,
            stem_kwargs=dict(
                stem_width=32,
            ),
            avg_down=True,
        ),
        meta=_meta(
            url='https://imvl-automl-sh.oss-cn-shanghai.aliyuncs.com/darts/hyperml/hyperml/job_45402/outputs/ECAResNet101D_281c5844.pth',
            paper='ECA-Net: Efficient Channel Attention for Deep CNN',
            code='https://github.com/BangguWu/ECANet',
            interpolation='bicubic',
            first_conv='conv1.0'
        )
    ),
    # Efficient Channel Attention ResNeXts
    'ecaresnext26tn_32x4d': dict(
        # Constructs an ECA-ResNeXt-26-TN model.
        # This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
        # in the deep stem. The channel number of the middle stem conv is narrower than the 'T' variant.
        # this model replaces SE module with the ECA module
        config=_cfg(
            block_kwargs=dict(
                attn_layer=EcaModule,
            ),
            layers=[2, 2, 2, 2],
            cardinality=32,
            base_width=4,
            stem_layer=DeepStem,
            stem_kwargs=dict(
                stem_width=32,
                tiered=True,
                narrow=True,
            ),
            avg_down=True,
        ),
        meta=_meta(
            paper='ECA-Net: Efficient Channel Attention for Deep CNN, Aggregated Residual Transformations for Deep Neural Networks, Bag of Tricks for Image Classification with CNNs',
            interpolation='bicubic',
            first_conv='conv1.0'
        )
    ),
    'ecaresnext50d_32x4d': dict(
        config=_cfg(
            block_kwargs=dict(
                attn_layer=EcaModule,
            ),
            layers=[3, 4, 6, 3],
            cardinality=32,
            base_width=4,
            stem_layer=DeepStem,
            stem_kwargs=dict(
                stem_width=32,
            ),
            avg_down=True,
        ),
        meta=_meta(
            paper='ECA-Net: Efficient Channel Attention for Deep CNN, Aggregated Residual Transformations for Deep Neural Networks, Bag of Tricks for Image Classification with CNNs',
            interpolation='bicubic',
            first_conv='conv1.0'
        )
    ),

    # ResNets with anti-aliasing blur pool
    'resnetblur18': dict(
        config=_cfg(
            block_layer=BasicBlock,
            aa_layer=BlurPool2d,
            layers=[2, 2, 2, 2],
        ),
        meta=_meta(
            interpolation='bicubic',
        )
    ),
    'resnetblur50': dict(
        config=_cfg(
            aa_layer=BlurPool2d,
            layers=[3, 4, 6, 3],
        ),
        meta=_meta(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnetblur50-84f4748f.pth',
            interpolation='bicubic',
        )
    ),
}
