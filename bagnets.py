# adapted from https://github.com/wielandbrendel/bag-of-local-features-models/blob/master/bagnets/pytorch.py

import torch.nn as nn
import math
import torch
from collections import OrderedDict
from torch.utils import model_zoo

import os

dir_path = os.path.dirname(os.path.realpath(__file__))

__all__ = ['bagnet9', 'bagnet17', 'bagnet33']


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 kernel_size=1):
        super(Bottleneck, self).__init__()
        # print('Creating bottleneck with kernel size {} and stride {} with padding {}'.format(kernel_size, stride, (kernel_size - 1) // 2))
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        #self.bn1 = nn.GroupNorm(32, planes)
        self.bn1 = nn.InstanceNorm3d(planes)
        pad = 1 if kernel_size == 3 else 0
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel_size,
                               stride=stride,
                               padding=pad,
                               bias=False)  # changed padding from (kernel_size - 1) // 2
        #self.bn2 = nn.GroupNorm(32, planes)
        self.bn2 = nn.InstanceNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        #self.bn3 = nn.GroupNorm(32, planes * 4)
        self.bn3 = nn.InstanceNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:, :, :-diff, :-diff]

        out += residual
        out = self.relu(out)

        return out


class BagNet(nn.Module):

    def __init__(self, block, layers, strides=[1, 2, 2, 2], scale_filters=0,
                 kernel3=[0, 0, 0, 0], num_classes=1000, avg_pool=True):
        self.scale_factor = 2 ** scale_filters
        start_filters = int(64 * self.scale_factor)
        self.inplanes = start_filters
        super(BagNet, self).__init__()
        self.conv1 = nn.Conv3d(1, start_filters, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv2 = nn.Conv3d(start_filters, start_filters, kernel_size=3,
                               stride=1, padding=1, bias=False)
        #self.bn1 = nn.GroupNorm(32, start_filters)
        self.bn1 = nn.InstanceNorm3d(start_filters)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, start_filters, layers[0], stride=strides[0],
                                       kernel3=kernel3[0], prefix='layer1')
        self.layer2 = self._make_layer(block, 2 * start_filters, layers[1],
                                       stride=strides[1], kernel3=kernel3[1],prefix='layer2')
        self.layer3 = self._make_layer(block, 4 * start_filters, layers[2],
                                       stride=strides[2],
                                       kernel3=kernel3[2], prefix='layer3')
        self.layer4 = self._make_layer(block, 8 * start_filters, layers[3],
                                       stride=strides[3],
                                       kernel3=kernel3[3], prefix='layer4')
        self.fc = nn.Linear(8 * start_filters * block.expansion,num_classes)
        self.avg_pool = avg_pool
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0,
                    prefix=''):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                #nn.GroupNorm(32, planes * block.expansion),
                nn.InstanceNorm3d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample,
                            kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avg_pool:
            x = x.mean(2).mean(2).mean(2)
            x = self.fc(x)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = self.fc(x)

        return x


def bagnet33(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-33 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides,
                   kernel3=[1, 1, 1, 1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet33']))
    return model


def bagnet17(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-17 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides,
                   kernel3=[1, 1, 1, 0], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet17']))
    return model


def bagnet9(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-9 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides,
                   kernel3=[1, 1, 0, 0], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet9']))
    return model

def bagnet177(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-9 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides,
                   kernel3=[3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet177']))
    return model
