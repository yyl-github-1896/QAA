import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torchvision.transforms as transforms

from my_utils import *


parser = argparse.ArgumentParser(description='evaluating adversarial robustness')
parser.add_argument('--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--data_source', default='val_clean', type=str, help='')

'''Quantization Settings'''
parser.add_argument('--arch', default="inception_v3", type=str, help='target model architecture')
parser.add_argument('--w_bit', default=32, type=int, help='quantization bits of the target model')
parser.add_argument('--a_bit', default=32, type=int, help='quantization bits of the target model')
parser.add_argument('--quantize_method', default="fp", type=str, help='quantize method of target model')

'''QAA Settings'''
parser.add_argument('--stochastic', default=False, type=bool, help='activation quantization or not')
parser.add_argument('--ckpt_id', default=None, type=str, help='QAA checkpoint id. default: 120603')

parser.add_argument('--output_dir', default="./adv_imgs/fp/resnet34_w32a32/mi_fgsm", type=str, 
                        help='directory to the adversarial images')
parser.add_argument('--ckpt_dir', default="./checkpoints", type=str)
parser.add_argument('--device', default='0', type=str, help='gpu device')
parser.add_argument('--use_gpu', default=True, type=bool, help='')

args = parser.parse_args()



def evaluate(net, test_set, use_gpu=True):
    use_gpu = False if args.quantize_method == "pytorch" else True
    net.eval()
    if use_gpu == True:
        net = net.cuda()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=1, pin_memory=True)
    correct = 0
    for i, (x, y) in enumerate(test_loader):
        if use_gpu == True:
            x, y = x.cuda(), y.cuda()
        output = net(x)
        pred_top1 = output.topk(k=1, largest=True).indices
        if pred_top1.dim() >= 2:
            pred_top1 = pred_top1.squeeze()
        correct += (pred_top1 == y).sum().item()
    return correct / len(test_set)


def main():
    print("attack source: {}".format(args.output_dir))

    '''load target model'''
    model = load_model(args)
    normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = nn.Sequential(normalize, model)

    '''load adversarial images'''
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_set = MyDataset(img_dir=args.output_dir, label_dir='val_rs.csv', transform=transform, png=True if args.data_source == 'val_clean_qnn' else False)
    
    '''evaluate'''
    adv_acc = evaluate(model, test_set, use_gpu=True)
    print("attack success rate: %.3f" % (1 - adv_acc))

    return
    

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()