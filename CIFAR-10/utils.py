
import numpy as np
import os
import robustbench.utils
import robustbench.model_zoo.cifar10
import torch
import torch.nn as nn

import sys
sys.path.append('/data/yyl/source/Adv_Quantization/CIFAR-10')

from fnmatch import fnmatch
from PIL import Image


# copy from advertorch
class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

def tensor2np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((0, 2, 3, 1))
    return img


# copy from advertorch
class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

# save the adversarial examples in numpy format
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform, output_dir):
        images = np.load(os.path.join(output_dir, "images.npy"))
        labels = np.load(os.path.join(output_dir, "labels.npy"))
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        # assert images.shape[0] == 10000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels.astype(np.longlong)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)


class MergedModel(nn.Module):
    def __init__(self, model_list=None, merge_mode=None):
        super(MergedModel, self).__init__()
        self.model_list = model_list
        self.num_models = len(model_list)   
        assert merge_mode in ['logits', 'softmax', 'sampling']
        self.merge_mode = merge_mode
        self.softmax = nn.Softmax(dim=1)
        print("self.merge_mode: {}".format(self.merge_mode))

    def forward(self, x):
        if self.merge_mode == 'softmax':
            out = self.softmax(self.model_list[0](x))
            for i in range(1, self.num_models):
                out = out + self.softmax(self.model_list[i](x))   # strange bug here
        elif self.merge_mode == 'logits':
            out = self.model_list[0](x)
            for i in range(1, self.num_models):
                out += self.model_list[i](x)
        elif self.merge_mode == 'sampling':
            import random
            seed = random.randint(0, self.num_models - 1)
            out = self.model_list[seed](x) 
        else:
            raise Exception('merge_mode {} not implemented!'.format(self.merge_mode))
        return out

def get_cifar10_dataloader(train, batch_size, normalize):
    import torchvision.transforms as transforms
    import torchvision
    if normalize == True:   
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    data_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,
        transform=transform,
    )
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False,
                                        num_workers=4)
    return data_loader



def load_model(args):
    print('args.quantize_method: {}'.format(args.quantize_method))

    if args.quantize_method == "" or args.quantize_method == "fp" or args.quantize_method == None:
        if args.arch == "resnet18":
            from models.fp_models.resnet import resnet18
            model = resnet18()
            model_dir = os.path.join(args.ckpt_dir, "fp", args.arch + "_32bit.pth.tar")
            model.load_state_dict(torch.load(model_dir))
        elif args.arch == "resnet20":
            import models.apot as models
            from models.apot.quant_layer import QuantConv2d, weight_quantize_fn, build_power_value, act_quantization
            model = models.resnet20_cifar(float=True)
            model = nn.DataParallel(model).cuda()
            model_dir = os.path.join(args.ckpt_dir, "apot", args.arch + "_" + str(args.w_bit) + "bit", "model_best.pth.tar")
            model.load_state_dict(torch.load(model_dir)["state_dict"])
        elif args.arch == "resnet56":
            import models.apot as models
            from models.apot.quant_layer import QuantConv2d, weight_quantize_fn, build_power_value, act_quantization
            model = models.resnet56_cifar(float=True)
            model = nn.DataParallel(model).cuda()
            model_dir = os.path.join(args.ckpt_dir, "apot", args.arch + "_" + str(args.w_bit) + "bit", "model_best.pth.tar")
            model.load_state_dict(torch.load(model_dir)["state_dict"])
        elif args.arch == "resnet50":
            from models.fp_models.resnet import resnet50
            model = resnet50()
            model_dir = os.path.join(args.ckpt_dir, "fp", args.arch + "_32bit.pth.tar")
            model.load_state_dict(torch.load(model_dir))
        elif args.arch == "resnet_cifar10":
            from models.bnn_models.resnet import resnet
            model = resnet(dataset="cifar10")
            model_dir = os.path.join(args.ckpt_dir, "fp", args.arch + "_32bit.pth.tar")
            model.load_state_dict(torch.load(model_dir, map_location='cuda:0')['state_dict'])
        elif args.arch == "vgg16":
            from models.fp_models.vgg import VGG
            model = VGG("VGG16")
            model_dir = os.path.join(args.ckpt_dir, "fp", args.arch + "_32bit.pth.tar")
            model.load_state_dict(torch.load(model_dir))
        elif args.arch == "densenet121":
            from models.fp_models.densenet import densenet121
            model = densenet121()
            model_dir = os.path.join(args.ckpt_dir, "fp", args.arch + "_32bit.pth.tar")
            model.load_state_dict(torch.load(model_dir))
        elif args.arch == "mobilenet_v2":
            from models.fp_models.mobilenetv2 import MobileNetV2
            model = MobileNetV2()
            model_dir = os.path.join(args.ckpt_dir, "fp", args.arch + "_32bit.pth.tar")
            model.load_state_dict(torch.load(model_dir))
        elif args.arch == "nin":
            from models.fp_models.nin import NIN
            model = NIN()
            model_dir = os.path.join(args.ckpt_dir, "fp", "nin" + "_32bit.pth.tar")
            model.load_state_dict(torch.load(model_dir))
        elif args.arch == "vgg_cifar10":
            import models.bnn_models as models
            model = models.vgg_cifar10
            model_config = {'input_size': None, 'dataset': "cifar10"}
            model = model(**model_config)
            model_dir = os.path.join(args.ckpt_dir, "fp", args.arch + "_32bit.pth.tar")
            model.load_state_dict(torch.load(model_dir, map_location="cuda:0")["state_dict"], strict=False)
        else:
            raise Exception("arch {} not implemented!".format(args.arch))

    elif args.quantize_method == 'standard_fp':
        if args.arch == 'resnet18':
            from models.PyTorch_CIFAR10.cifar10_models.resnet import resnet18
            model = resnet18()
        elif args.arch == 'resnet34':
            from models.PyTorch_CIFAR10.cifar10_models.resnet import resnet34
            model = resnet34()
        elif args.arch == 'resnet50':
            from models.PyTorch_CIFAR10.cifar10_models.resnet import resnet50
            model = resnet50()
        elif args.arch == 'vgg19_bn':
            from models.PyTorch_CIFAR10.cifar10_models.vgg import vgg19_bn
            model = vgg19_bn()
        elif args.arch == 'densenet121':
            from models.PyTorch_CIFAR10.cifar10_models.densenet import densenet121
            model = densenet121()
        elif args.arch == 'mobilenet_v2':
            from models.PyTorch_CIFAR10.cifar10_models.mobilenetv2 import mobilenet_v2
            model = mobilenet_v2()
        elif args.arch == 'inception_v3':
            from models.PyTorch_CIFAR10.cifar10_models.inception import inception_v3
            model = inception_v3()
        else:
            raise Exception('arch {} not implemented!'.format(args.arch))
        model_dir = os.path.join(args.ckpt_dir, './fp/{}.pt'.format(args.arch))
        model.load_state_dict(torch.load(model_dir))

    elif args.quantize_method == "dsq":
        if args.arch == "resnet18":
            from models.fp_models.resnet import resnet18
            model = resnet18()
        elif args.arch == "resnet20":
            from models.fp_models.resnet20 import resnet20_cifar
            model = resnet20_cifar(float=True)
        elif args.arch == "resnet50":
            from models.fp_models.resnet import resnet50
            model = resnet50()
        elif args.arch == "vgg16":
            from models.fp_models.vgg import VGG
            model = VGG("VGG16")
        elif args.arch == "densenet121":
            from models.fp_models.densenet import densenet121
            model = densenet121()
        elif args.arch == "mobilenet_v2":
            from models.fp_models.mobilenetv2 import MobileNetV2
            model = MobileNetV2()
        elif args.arch == "nin":
            from models.fp_models.nin import NIN
            model = NIN()
        elif args.arch == "resnet20":
            from models.apot.resnet import resnet20_cifar
            model = resnet20_cifar(float=True)
        elif args.arch == "vgg_cifar10":
            from models.bnn_models.vgg_cifar10 import vgg_cifar10
            model_config = {'input_size': None, 'dataset': "cifar10"}
            model = vgg_cifar10(**model_config)
        else:
            raise Exception("arch {} not implemented!".format(args.arch))
        model_dir = os.path.join(args.ckpt_dir, args.quantize_method, args.arch + "_w" + str(args.w_bit) + "a" + str(args.a_bit) + ".pth.tar")
        from models.DSQ.PyTransformer.transformers.torchTransformer import TorchTransformer
        from models.DSQ.PyTransformer.transformers.quantize import QConv2d, QuantConv2d, QLinear, ReLUQuant
        from models.DSQ.DSQConv import DSQConv
        from models.DSQ.DSQLinear import DSQLinear
        from models.DSQ.utils import set_quanbit, set_quanInput, set_nin
        transformer = TorchTransformer()
        transformer.register(nn.Conv2d, DSQConv)
        model = transformer.trans_layers(model)
        if args.arch == "nin":
            model = set_nin(model, args.w_bit, args.a_bit)
        else:
            model = set_quanbit(model, args.w_bit)      
            model = set_quanInput(model, args.a_bit)
        model.load_state_dict(torch.load(model_dir, map_location="cuda:0")["state_dict"])

    elif args.quantize_method == "bnn":
        import models.bnn_models as models
        if args.arch == "resnet_cifar10":
            model = models.resnet_binary
        elif args.arch == "vgg_cifar10":
            model = models.vgg_cifar10_binary
        else:
            raise Exception("arch {} not implemented!".format(args.arch))
        model_config = {'input_size': None, 'dataset': "cifar10"}
        model = model(**model_config)
        model_dir = os.path.join(args.ckpt_dir, args.quantize_method, args.arch + "_1bit.pth.tar")
        model.load_state_dict(torch.load(model_dir, map_location="cuda:0")["state_dict"])

    elif args.quantize_method == "apot":
        from models.apot import resnet56_cifar, resnet20_cifar
        from models.apot.quant_layer import QuantConv2d, weight_quantize_fn, build_power_value, act_quantization
        if args.arch == 'resnet56':
            model = resnet56_cifar(float=False, stochastic=args.stochastic)
        elif args.arch == 'resnet20':
            model = resnet20_cifar(float=False, stochastic=args.stochastic)
        else:
            raise Exception('arch {} not implemented!'.format(args.arch))
        for m in model.modules():
            if isinstance(m, QuantConv2d):
                m.update_params(bit=args.w_bit)
                m.weight_quant = weight_quantize_fn(w_bit=args.w_bit)
                m.act_grid = build_power_value(args.w_bit)
                m.act_alq = act_quantization(args.w_bit, m.act_grid)
        model = nn.DataParallel(model).cuda()
        stochastic_name = '_stochastic_' + args.ckpt_name if args.stochastic else ''
        model_dir = os.path.join(args.ckpt_dir, './apot/{}_{}bit{}/model_best.pth.tar'.format(args.arch, args.w_bit, stochastic_name)) 
        model.load_state_dict(torch.load(model_dir)["state_dict"])
        print('model successfully loaded from {}'.format(model_dir))

    elif args.quantize_method == "xnor_net":
        model_dir = os.path.join(args.ckpt_dir, args.quantize_method, "nin" + "_" + str(args.w_bit) + "bit.pth.tar")
        from models.xnor_net.nin import Net
        from models.xnor_net.utils import BinOp
        model = Net()
        model = nn.DataParallel(model).cuda()
        model.load_state_dict(torch.load(model_dir))
        bin_op = BinOp(model)
        bin_op.binarization()

    elif args.quantize_method == "dorefa_net":  
        model_dir = os.path.join(args.ckpt_dir, args.quantize_method, args.arch + "_w{}a{}.pth.tar".format(args.w_bit, args.a_bit))
        from models.dorefa_net.nets.cifar_resnet import resnet20
        wbits = args.w_bit
        if args.arch == "resnet20":
            model = resnet20(wbits=wbits, abits=args.a_bit).cuda()
        model.load_state_dict(torch.load(model_dir))

    elif args.quantize_method == "pact":
        if args.arch == "resnet20":
            from models.pact.resnet import resnet20
            model = resnet20()
            model.set_quanbit(bitwidth = args.w_bit)
            model_dir = os.path.join(args.ckpt_dir, args.quantize_method, args.arch + "_{}bit.pth.tar".format(args.w_bit))
            model = torch.nn.DataParallel(model).cuda()
            model.load_state_dict(torch.load(model_dir)["state_dict"])
        else:
            raise Exception("arch {} not implemented!".format(args.arch))

    elif args.quantize_method == "dwq":
        num_bits_list = list(range(4, 16+1))
        if args.arch == "preact_resnet18":
            from models.DWQ.model.preact_resnet import PreActResNet18
            model = PreActResNet18(num_bits_list, num_classes=10, normalize=None).cuda()
        elif args.arch == "wideresnet":
            from models.DWQ.model.wide_resnet import WideResNet32
            model = WideResNet32(num_bits_list, num_classes=10, normalize=None).cuda()
        model = torch.nn.DataParallel(model)
        model_dir = os.path.join(args.ckpt_dir, args.quantize_method, args.arch + ".pth")
        model.load_state_dict(torch.load(model_dir)["state_dict"])

    else:
        raise Exception("quantize method {} not implemented!".format(args.quantize_method))
    print("target model {} loaded successfully!".format(model_dir))
    return model