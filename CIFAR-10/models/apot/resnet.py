'''
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import math
from fnmatch import fnmatch
from .quant_layer import *


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def Quantconv3x3(in_planes, out_planes, stride=1):
    " 3x3 quantized convolution with padding "
    return QuantConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, float=False):
        super(BasicBlock, self).__init__()
        if float:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv1 = Quantconv3x3(inplanes, planes, stride)
            self.conv2 = Quantconv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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

        out += residual
        out = self.relu(out)

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, stochastic=False, num_classes=10, float=False):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = first_conv(3, 16, kernel_size=3, stride=1, padding=1, bias=False) # this is essential
        # self.conv1 = QuantConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], float=float)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, float=float)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, float=float)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = last_fc(64 * block.expansion, num_classes)

        self.stochastic = stochastic
        print('float: {}, self.stochastic: {}'.format(float, self.stochastic))
        # self.mixed_mode = mixed_mode
        # print('self.mixed_mode: {}'.format(self.mixed_mode))
        # if mixed_mode != '' and mixed_mode != None:
        #     self.mixed_quantization__init__()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        

    def mixed_quantization__init__(self):
        assert fnmatch(self.mixed_mode, 'd*w*a*')
        depth = int(self.mixed_mode[1:3])
        kw = int(self.mixed_mode[4])
        ka = int(self.mixed_mode[6])
        assert kw == ka
        print('depth = {}, kw = {}, ka = {}'.format(depth, kw, ka))
        current_depth = 0
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                if current_depth > depth:
                    m.act_grid = build_power_value(self.bit, additive=True)
                    m.act_alq = act_quantization(self.bit, self.act_grid, power=True)
                current_depth += 1
        return self

    def _make_layer(self, block, planes, blocks, stride=1, float=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                QuantConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
                if float is False else nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                                                 stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, float=float))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, float=float))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.stochastic:
            self.switch()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                m.show_params()

    def reset_model(self, p=0.5):
        import random
        seed = random.random()
        quant = False if seed > p else True
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                m.quant = quant
        return self

    def stochastic_depth(self):
        import random
        seed = random.randint(self.min_depth, self.max_depth)
        depth = 0
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                if depth < seed:
                    m.quant = False
                else:
                    m.quant = True
                depth += 1
        return self


    def switch(self):
        '''for ablation'''
        # for m in self.modules():
        #     if isinstance(m, QuantConv2d):
        #         m.a_quant = False
        # return self
        '''version 120501'''
        # if hasattr(self, 'quant'):
        #     self.quant = not self.quant
        # else:
        #     self.quant = True
        # if self.quant == False:
        #     for m in self.modules():
        #         if isinstance(m, QuantConv2d):
        #             m.bit = 32
        # else:
        #     for m in self.modules():
        #         if isinstance(m, QuantConv2d):
        #             m.bit = self.bit
        # return self

        '''version 120502'''
        # if hasattr(self, 'a_quant'):
        #     self.a_quant = not self.a_quant
        # else:
        #     self.a_quant = True
        #     for m in self.modules():
        #         if isinstance(m, QuantConv2d):
        #             m.bit = 32
        # return self

        '''version 120601, 120602, 120603'''
        if hasattr(self, 'a_quant'):
            self.a_quant = not self.a_quant
        else:
            self.a_quant = True
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                m.a_quant = self.a_quant
        return self


    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                m.show_params()





def resnet8_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [1, 1, 1], **kwargs)
    return model


def resnet14_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [2, 2, 2], **kwargs)
    return model


def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model



if __name__ == '__main__':
    pass
    # net = resnet20_cifar(float=True)
    # y = net(torch.randn(1, 3, 64, 64))
    # print(net)
    # print(y.size())