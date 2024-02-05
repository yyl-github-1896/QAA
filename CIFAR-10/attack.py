import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from ops_attack import *
from utils import *


parser = argparse.ArgumentParser(description='generating adversarial exampels')
parser.add_argument('--batch_size', default=100, type=int, help='evaluation batch size')
parser.add_argument('--attack', default="pgd", type=str, help='clean, mi_fgsm, ci_fgsm, fia, rpa')
parser.add_argument('--epsilon', default=8/255, type=float, help='attack magnitude')
parser.add_argument('--arch', default='resnet56', type=str)
parser.add_argument('--ckpt_dir', default="./checkpoints", type=str)
parser.add_argument('--w_bit', default=32, type=int, help='quantization bitwidth of target model')
parser.add_argument('--a_bit', default=32, type=int, help='quantization bitwidth of target model')
parser.add_argument('--quantize_method', default="fp", type=str, help='quantize method of target model')
parser.add_argument('--merge_mode', default="", type=str, help='ensemble mode', choices=['logits', 'softmax', 'sampling'])
parser.add_argument('--ckpt_name', default="", type=str, help='quantize method of target model')
parser.add_argument('--stochastic', default=False, type=bool, help='for QAA, stochastic=True')

parser.add_argument('--device', default='0', type=str, help='gpu device')
parser.add_argument('--use_gpu', default=True, type=bool, help='')

args = parser.parse_args()


def evaluate(net, test_loader, use_gpu=True):
    if use_gpu == True:
        net = net.cuda().eval()

    correct = 0
    for (x, y) in tqdm(test_loader):
        if use_gpu == True:
            x, y = x.cuda(), y.cuda()
        if args.quantize_method == 'dwq':
            num_bits_list = list(range(4, 16+1))
            net[1].module.set_precision(num_bits=np.random.choice(num_bits_list), num_grad_bits=0)
        output = net(x)
        pred_top1 = output.topk(k=1, largest=True).indices
        if pred_top1.dim() >= 2:
            pred_top1 = pred_top1.squeeze()
        correct += (pred_top1 == y).sum().item()
    return correct / len(test_loader.dataset)


def get_gradient(model, x, y):
    x.requires_grad = True
    outputs = model(x)
    loss = F.cross_entropy(outputs, y)
    loss.backward()
    grad = x.grad.sum(dim=[1,2,3])
    return grad


def evaluate_adv(net, test_loader, output_dir, targeted=False):
    net.cuda().eval()
    success = 0
    adv_images = np.zeros((len(test_loader.dataset), 32, 32, 3), dtype=np.uint8)
    true_labels = np.zeros((len(test_loader.dataset)), dtype=np.longlong)
    i = 0
    
    for (x, y) in tqdm(test_loader):
        if args.quantize_method == "dwq":
            num_bits_list = list(range(4, 16+1))
            net[1].module.set_precision(num_bits=np.random.choice(num_bits_list), num_grad_bits=0)

        if args.attack == "clean":
            _, adv, is_adv = clean(net, x, y, epsilon=args.epsilon)
        elif args.attack == "fgsm":
            _, adv, is_adv = fgsm(net, x, y, epsilon=args.epsilon)
        elif args.attack == "mi_fgsm":
            _, adv, is_adv = mi_fgsm(net, x, y, epsilon=args.epsilon)
        elif args.attack == "ci_fgsm":
            _, adv, is_adv = ci_fgsm(net, x, y, epsilon=args.epsilon)  
        elif args.attack == "fia":
            arch = '{}_{}'.format(args.arch, args.quantize_method)
            _, adv, is_adv = fia(net, x, y, epsilon=args.epsilon, arch=arch)
        elif args.attack == "rpa":
            arch = '{}_{}'.format(args.arch, args.quantize_method)
            _, adv, is_adv = rpa(net, x, y, epsilon=args.epsilon, arch=arch)
        elif args.attack == "pgd":
            _, adv, is_adv = pgd(net, x, y, epsilon=args.epsilon)
        else: 
            raise Exception("attack {} is not implemented!".format(args.attack))
        attack_success_rate = is_adv.sum().item() / len(is_adv)
        print("\n i = {}, attack success rate: {}".format(i, attack_success_rate))
        success += is_adv.sum().item()
        '''save the adversarial images'''
        adv_images[i * args.batch_size: i * args.batch_size + len(y)] = \
            tensor2np(adv).astype(np.uint8)
        true_labels[i * args.batch_size: i * args.batch_size + len(y)] \
            = (y.cpu().numpy()).astype(np.longlong)
        i += 1
    '''save images and labels in numpy format'''
    data_dest = os.path.join(output_dir, "images.npy")
    label_dest = os.path.join(output_dir, "labels.npy")
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    np.save(data_dest, adv_images)
    print("file saved at: ", data_dest)
    np.save(label_dest, true_labels)
    print("file saved at: ", label_dest)
    return success / len(test_loader.dataset)

model_configs = [
    {
        "quantize_method": "standard_fp",
        "bit": 32,
        "arch": "resnet18",
        'ckpt_name': '',
    },
    {
        "quantize_method": "standard_fp",
        "bit": 32,
        "arch": "resnet50",
        'ckpt_name': '',
    },
    {
        "quantize_method": "standard_fp",
        "bit": 32,
        "arch": "vgg19_bn",
        'ckpt_name': '',
    },
    {
        "quantize_method": "standard_fp",
        "bit": 32,
        "arch": "densenet121",
        'ckpt_name': '',
    },
    {
        "quantize_method": "standard_fp",
        "bit": 32,
        "arch": "mobilenet_v2",
        'ckpt_name': '',
    },
]



def main():

    stochastic_name = '_stochastic' if args.stochastic else ''
    data_centric_name = '_{}'.format(args.ckpt_name) if args.ckpt_name else ''
    output_dir = os.path.join("./adv_imgs", args.quantize_method, args.arch + data_centric_name + '_w{}a{}'.format(args.w_bit, args.a_bit) + stochastic_name, args.attack)

    if args.quantize_method == "merged":
        model_list = []
        for config in model_configs:
            args.quantize_method = config["quantize_method"]
            args.w_bit = config["bit"]
            args.a_bit = config["bit"]
            args.arch = config["arch"]
            args.ckpt_name = config['ckpt_name']
            raw_model = load_model(args).cuda().eval()     # eval here is important
            model_list.append(raw_model)
        model = MergedModel(model_list=model_list, merge_mode=args.merge_mode)
    else:
        model = load_model(args)

    if args.quantize_method not in ['trades', 'fat', 'simi', 'fda', 'Gowal2021Improving_28_10_ddpm_100m', 'Carmon2019Unlabeled', 'augmax'] and args.ckpt_name not in ['eat']:
        normalize = NormalizeByChannelMeanStd((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        model = nn.Sequential(normalize, model)
    elif args.quantize_method == 'standard_fp':
        normalize = NormalizeByChannelMeanStd(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
        model = nn.Sequential(normalize, model)
    model.eval()
    
    test_loader = get_cifar10_dataloader(False, args.batch_size, False)

    clean_acc = evaluate(model, test_loader, use_gpu=True)
    print("clean_acc: {}".format(clean_acc))
    attack_success_rate = evaluate_adv(model, test_loader, output_dir)
    print("attack success rate: %.4f" % (attack_success_rate))

    return



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()