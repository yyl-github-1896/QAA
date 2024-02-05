import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from utils import *


parser = argparse.ArgumentParser(description='evaluating adversarial robustness')
parser.add_argument('--batch_size', default=50, type=int, help='batch size')
parser.add_argument('--num_steps', default=10, type=int, help='for mi-fgsm')
parser.add_argument('--step_size', default=2/255, type=float, help='for pgd')
parser.add_argument('--epsilon', default=8/255, type=float, help='perturbation magnitude')
parser.add_argument('--w_bit', default=32, type=int, help='quantization bits of the target model')
parser.add_argument('--a_bit', default=32, type=int, help='quantization bits of the target model')
parser.add_argument('--quantize_method', default="fp", type=str, help='quantize method of target model')
parser.add_argument('--stochastic_name', default="", type=str, help='')
parser.add_argument('--stochastic', default=False, type=bool, help='')
parser.add_argument('--mixed_mode', default="", type=str, help='')

parser.add_argument('--arch', default='mobilenet_v2', type=str, help='target model architecture')
parser.add_argument('--ckpt_name', default='', type=str, help='for data-centric')
parser.add_argument('--output_dir', default="./adv_imgs/fp/resnet56_w32a32/clean", type=str)

parser.add_argument('--ckpt_dir', default="./checkpoints", type=str)
parser.add_argument('--device', default='0', type=str, help='gpu device')
parser.add_argument('--use_gpu', default=True, type=bool, help='')

args = parser.parse_args()

'''for DSQ'''
def set_quanbit(model, quan_bit = 8):
    
    for module_name in model._modules:
        # print("module_name: {}".format(module_name))
        if len(model._modules[module_name]._modules) > 0:
            set_quanbit(model._modules[module_name], quan_bit)
        else:
            if hasattr(model._modules[module_name], "num_bit"):                
                setattr(model._modules[module_name], "num_bit", quan_bit) 
    return model

def set_quanInput(model, quan_input = True):
    for module_name in model._modules:        
        if len(model._modules[module_name]._modules) > 0:
            set_quanInput(model._modules[module_name], quan_input)
        else:
            # for DSQ
            if hasattr(model._modules[module_name], "quan_input"):
                setattr(model._modules[module_name], "quan_input", quan_input) 
    return model


def evaluate(net, test_set, use_gpu=True):
    net.eval()
    if use_gpu == True:
        net = net.cuda()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=1, pin_memory=True)
    correct = 0
    for i, (x, y) in enumerate(test_loader):
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
    
    return correct / len(test_set)



def evaluate_fairness(net, test_set, use_gpu=True):
    net.eval()
    if use_gpu == True:
        net = net.cuda()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=1, pin_memory=True)
    correct_per_class = np.zeros(10, dtype=int)
    for i, (x, y) in enumerate(test_loader):
        if use_gpu == True:
            x, y = x.cuda(), y.cuda()
        if args.quantize_method == 'dwq':
            num_bits_list = list(range(4, 16+1))
            net[1].module.set_precision(num_bits=np.random.choice(num_bits_list), num_grad_bits=0)
        output = net(x)
        pred_top1 = output.topk(k=1, largest=True).indices
        if pred_top1.dim() >= 2:
            pred_top1 = pred_top1.squeeze()
        correct = (pred_top1 == y)
        for idx, corr in enumerate(correct):
            correct_per_class[y[idx].cpu().item()] += corr.cpu().item()
    return correct_per_class * 10 / len(test_set)



def main():
    print("attack source: {}".format(args.output_dir))
    if args.quantize_method == 'empir':
        model_configs = [
            {
                'quantize_method': 'fp',
                'w_bit': 32,
                'a_bit': 32,
            },
            {
                'quantize_method': 'apot',
                'w_bit': 2,
                'a_bit': 2,
            }
        ]
        model_list = []
        for config in model_configs:
            args.quantize_method = config['quantize_method']
            args.w_bit = config['w_bit']
            args.a_bit = config['a_bit']
            args.stochastic = False
            args.ckpt_name = ''
            raw_model = load_model(args).cuda()
            model_list.append(raw_model)
        model = MergedModel(model_list, 'logits')
    else: 
        model = load_model(args)
 
    if args.quantize_method not in ['trades', 'fat', 'simi', 'fda', 'Gowal2021Improving_28_10_ddpm_100m', 'augmax'] and args.ckpt_name not in ['eat']:
        normalize = NormalizeByChannelMeanStd(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = nn.Sequential(normalize, model)
    elif args.quantize_method == 'standard_fp':
        normalize = NormalizeByChannelMeanStd(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
        model = nn.Sequential(normalize, model)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_set = MyDataset(transform=transform, output_dir=args.output_dir)
    adv_acc = evaluate(model, test_set, use_gpu=True)
    adv_acc_per_class = evaluate_fairness(model, test_set, use_gpu=True)
    print('attack success rate: %.4f' % (1 - adv_acc))
    return
    


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()