from collections import OrderedDict
from . models.resnet import resnet18 as _resnet18
from . models.resnet import resnet50 as _resnet50
from . models.mobilenetv2 import mobilenetv2 as _mobilenetv2
from . models.mnasnet import mnasnet as _mnasnet
from . models.regnet import regnetx_600m as _regnetx_600m
from . models.regnet import regnetx_3200m as _regnetx_3200m
import torch
dependencies = ['torch']
# prefix = '/data/yyl/source/Adv_Quantization_243/CIFAR10/checkpoints/fp'
prefix = './checkpoints/fp'
model_path = {
    # 'resnet18': prefix+'/model_zoo/resnet18_imagenet.pth.tar',
    'resnet18': prefix+'/resnet18_qdrop_32bit.pth.tar',
    'resnet50': prefix+'/model_zoo/resnet50_imagenet.pth.tar',
    'mbv2': prefix+'/model_zoo/mobilenetv2.pth.tar',
    'reg600m': prefix+'/model_zoo/regnet_600m.pth.tar',
    'reg3200m': prefix+'/model_zoo/regnet_3200m.pth.tar',
    'mnasnet': prefix+'/model_zoo/mnasnet.pth.tar',
    'spring_resnet50': prefix+'/model_zoo/spring_resnet50.pth',
}


def resnet18(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _resnet18(**kwargs)
    if pretrained:
        import os
        print("current path: {}".format(os.getcwd()))
        state_dict = torch.load(model_path['resnet18'], map_location='cpu')
        # checkpoint = {k: v for k, v in state_dict.items() if (k in model.state_dict() and 'fc' not in k)}
        checkpoint = state_dict
        model.load_state_dict(checkpoint, strict=True)
    return model


def resnet50(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _resnet50(**kwargs)
    if pretrained:
        state_dict = torch.load(model_path['resnet50'], map_location='cpu')
        checkpoint = {k: v for k, v in state_dict.items() if (k in model.state_dict() and 'fc' not in k)}
        model.load_state_dict(checkpoint, strict=False)
    return model


def spring_resnet50(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _resnet50(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path['spring_resnet50'], map_location='cpu')
        q = OrderedDict()
        for k, v in checkpoint.items():
            q[k[7:]] = v
        model.load_state_dict(q)
    return model


def mobilenetv2(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _mobilenetv2(**kwargs)
    if pretrained:
        state_dict = torch.load(model_path['mbv2'], map_location='cpu')
        checkpoint = {k: v for k, v in state_dict.items() if (k in model.state_dict() and 'fc' not in k)}
        model.load_state_dict(checkpoint, strict=False)
    return model


def regnetx_600m(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _regnetx_600m(**kwargs)
    if pretrained:
        state_dict = torch.load(model_path['reg600m'], map_location='cpu')
        checkpoint = {k: v for k, v in state_dict.items() if (k in model.state_dict() and 'fc' not in k)}
        model.load_state_dict(checkpoint, strict=False)
    return model


def regnetx_3200m(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _regnetx_3200m(**kwargs)
    if pretrained:
        state_dict = torch.load(model_path['reg3200m'], map_location='cpu')
        checkpoint = {k: v for k, v in state_dict.items() if (k in model.state_dict() and 'fc' not in k)}
        model.load_state_dict(checkpoint, strict=False)
    return model


def mnasnet(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _mnasnet(**kwargs)
    if pretrained:
        state_dict = torch.load(model_path['mnasnet'], map_location='cpu')
        checkpoint = {k: v for k, v in state_dict.items() if (k in model.state_dict() and 'fc' not in k)}
        model.load_state_dict(checkpoint, strict=False)
    return model
