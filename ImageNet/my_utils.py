
import numpy as np
import os
import pandas as pd
import robustbench
import torch
import torch.nn as nn
import torchvision

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

def jpeg2png(name):
    name_list = list(name)
    name_list[-4:-1] = 'png'
    name_list.pop(-1)
    return ''.join(name_list)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, transform=None, png=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.labels = pd.read_csv(self.label_dir).to_numpy()
        self.png = png

    def __getitem__(self, index):
        file_name, label = self.labels[index]
        label = torch.tensor(label) - 1
        file_dir = os.path.join(self.img_dir, file_name)
        if self.png:
            file_dir = jpeg2png(file_dir)
        img = Image.open(file_dir).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)

def save_images(adv, i, batch_size, label_dir, output_dir, png=True):
    '''
    save the adversarial images
    :param adv: adversarial images in [0, 1]
    :param i: batch index of images
    :return:
    '''
    dest_dir = output_dir
    labels = pd.read_csv(label_dir).to_numpy()
    base_idx = i * batch_size
    for idx, img in enumerate(adv):
        fname = labels[idx + base_idx][0]
        dest_name = os.path.join(dest_dir, fname)
        if png:
            dest_name = jpeg2png(dest_name)
        torchvision.utils.save_image(img, dest_name)
    return


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


class LGVModel(nn.Module):

    def __init__(self, arch, qaa):
        super(LGVModel, self).__init__()
        self.sequence = np.random.permutation(np.arange(1, 41))
        self.id = 0
        self.arch = arch
        self.qaa = qaa
        if self.qaa:
            if self.arch == 'resnet34':
                import archs.apot as models
                model = models.__dict__[self.arch](pretrained=False, bit=2, stochastic=True)
                model = torch.nn.DataParallel(model).cuda()
                ckpt_dir = './checkpoints/apot/{}_w2a2_stochastic_120603.pth.tar'.format(self.arch)
                model.load_state_dict(torch.load(ckpt_dir)["state_dict"])
                self.qnn = model.eval().cuda()
            elif self.arch == 'resnet50':
                pass
            else:
                raise Exception('arch {} not implemented!'.format(self.arch))
            
    def forward(self, x):
        if self.qaa:
            if self.id % 2 == 0:
                out = self.qnn(x)
            else:
                import torchvision.models as models
                model = models.__dict__[self.arch](pretrained=False).cuda().eval()
                ckpt_dir = './checkpoints/fp/lgv/%s/cSGD/seed0/iter-%05d.pt' % (self.arch, self.sequence[self.id])
                # print('ckpt_dir: {}'.format(ckpt_dir))
                model.load_state_dict(torch.load(ckpt_dir)['state_dict'])
                out = model(x)
            self.id = (self.id + 1) % 40
        else:
            import torchvision.models as models
            model = models.__dict__[self.arch](pretrained=False).cuda().eval()
            ckpt_dir = './checkpoints/fp/lgv/%s/cSGD/seed0/iter-%05d.pt' % (self.arch, self.sequence[self.id])
            model.load_state_dict(torch.load(ckpt_dir)['state_dict'])
            out = model(x)
            self.id = (self.id + 1) % 40
        return out



'''for DSQ'''
def set_quanbit(model, w_bit):

    for module_name in model._modules:
        if len(model._modules[module_name]._modules) > 0:
            set_quanbit(model._modules[module_name], w_bit)
        else:
            if hasattr(model._modules[module_name], "w_bit"):                
                setattr(model._modules[module_name], "w_bit", w_bit) 
                setattr(model._modules[module_name], "w_range", 2 ** w_bit - 1) 
    return model

def set_quanInput(model, a_bit):
    for module_name in model._modules:        
        if len(model._modules[module_name]._modules) > 0:
            set_quanInput(model._modules[module_name], a_bit)
        else:
            # for DSQ
            if hasattr(model._modules[module_name], "a_bit"):
                setattr(model._modules[module_name], "a_bit", a_bit) 
                setattr(model._modules[module_name], "a_range", 2 ** a_bit - 1) 
                QInput = False if a_bit == 32 else True
                setattr(model._modules[module_name], "quan_input", QInput) 
    return model



def load_model(args):
    if args.quantize_method == "fp":
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        if args.arch in ['inception_v4', 'inception_resnet_v2']:
            import archs.fp as models
            model = models.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
        else:
            import torchvision.models as models
            model = models.__dict__[args.arch](pretrained=True)
        print("model {} successfully loaded".format(args.arch))
    elif args.quantize_method == "pytorch":
        assert args.w_bit == 8 and args.a_bit == 8
        import torchvision.models.quantization as models
        model = models.__dict__[args.arch](pretrained=True, quantize=True)
        print("8-bit model {} loaded successfully!".format(args.arch))
    elif args.quantize_method == "apot":
        if args.stochastic == True and args.ckpt_id == '120603':
            import archs.apot as models
            model = models.__dict__[args.arch](pretrained=False, bit=args.w_bit, stochastic=True)
            model = torch.nn.DataParallel(model).cuda()
            model_dir = os.path.join(args.ckpt_dir, "apot", args.arch + "_w{}a{}_stochastic_120603.pth.tar".format(args.w_bit, args.a_bit))
            model.load_state_dict(torch.load(model_dir)["state_dict"])
        else:
            import archs.apot as models
            model = models.__dict__[args.arch](pretrained=False, bit=args.w_bit, stochastic=False)
            model = torch.nn.DataParallel(model).cuda()
            model_dir = os.path.join(args.ckpt_dir, "apot", args.arch + "_w{}a{}.pth.tar".format(args.w_bit, args.a_bit))
            model.load_state_dict(torch.load(model_dir)["model"])
        print("model successfully loaded from {}".format(model_dir))
    elif args.quantize_method == "dsq":
        import torchvision.models as models
        model = models.__dict__[args.arch](pretrained=False)
        from archs.dsq.PyTransformer.transformers.torchTransformer import TorchTransformer
        from archs.dsq.PyTransformer.transformers.quantize import QConv2d
        from archs.dsq.DSQConv import DSQConv
        from archs.dsq.DSQLinear import DSQLinear
        transformer = TorchTransformer()
        transformer.register(nn.Conv2d, DSQConv)
        model = transformer.trans_layers(model)          
        model = set_quanbit(model, args.w_bit)   
        model = set_quanInput(model, args.a_bit)
        model_dir = os.path.join(args.ckpt_dir, args.quantize_method, args.arch + "_w{}a{}.pth.tar".format(args.w_bit, args.a_bit))
        model.load_state_dict(torch.load(model_dir, map_location="cuda:0")["state_dict"])
        print("model successfully loaded from {}".format(model_dir))
    elif args.quantize_method == "qdrop":
        from archs.qdrop.load import load_model
        if args.arch == "mobilenet_v2":
            args.arch = "mobilenetv2"
        model = load_model(arch=args.arch, n_bits_w=args.w_bit, n_bits_a=args.a_bit, stochastic_mode=args.stochastic_mode)
    elif args.quantize_method == 'xnor_net':
        assert args.w_bit == 1 and args.a_bit == 1
        import quantize_methods.XNOR_Net.networks.model_list as models
        if args.arch == 'alexnet':
            model = models.alexnet()
            model.features = torch.nn.DataParallel(model.features).cuda()
            ckpt_dir = './quantize_methods/XNOR_Net/checkpoints/alexnet.baseline.pth.tar'
            model.load_state_dict(torch.load(ckpt_dir)["state_dict"])
            print("model successfully loaded from {}".format(ckpt_dir))
        else:
            raise Exception('arch {} not implemented!'.format(args.arch))
    else:
        raise Exception('quantize method {} not implemented!'.format(args.quantize_method))

    return model