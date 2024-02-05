import argparse
import copy
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

from my_utils import *
from ops_attack import *

parser = argparse.ArgumentParser(description='generating adversarial exampels')
parser.add_argument('--batch_size', default=50, type=int, help='evaluation batch size')
parser.add_argument('--attack', default="mi_fgsm", type=str, choices=['clean', 'mi_fgsm', 'fia', 'rpa', 'ci_fgsm', 'admix', 'ssa'])
parser.add_argument('--epsilon', default=16/255, type=float, help='attack magnitude')

parser.add_argument('--data_source', required=True, type=str, 
                        help='directory to the validation images used to generate adversarial images')
parser.add_argument('--ckpt_dir', default="./checkpoints", type=str)

'''QNN settings'''
parser.add_argument('--arch', default="resnet34", type=str)
parser.add_argument('--w_bit', default=32, type=int, help='quantization bitwidth of target model')
parser.add_argument('--a_bit', default=32, type=int, help='quantization bitwidth of target model')
parser.add_argument('--quantize_method', default="fp", type=str, help='quantize method of target model')

'''for ensemble attacks'''
parser.add_argument('--merge_mode', default="", type=str, 
                        help='ensemble mode', choices=['logits', 'softmax', 'sampling'])

'''for qaa'''
parser.add_argument('--stochastic', default=False, type=bool, help='for QAA, stochastic=True')
parser.add_argument('--ckpt_id', default='', type=str, help='QAA checkpoint id. default=120603')

parser.add_argument('--device', default='0', type=str, help='gpu device')
parser.add_argument('--use_gpu', default=True, type=bool, help='')

args = parser.parse_args()


def evaluate(net, test_set, use_gpu=True):
    use_gpu = False if args.quantize_method == "pytorch" else True
    if use_gpu == True:
        net = net.cuda()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=4)
    correct = 0
    length = 0
    for i, (x, y) in enumerate(tqdm(test_loader)):
        if use_gpu == True:
            x, y = x.cuda(), y.cuda()
        output = net(x)
        pred_top1 = output.topk(k=1, largest=True).indices
        if pred_top1.dim() >= 2:
            pred_top1 = pred_top1.squeeze()
        correct += (pred_top1 == y).sum().item()
        length += len(y)
    return correct / length

def evaluate_adv(net, test_set, output_dir):
    net.cuda()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=4)
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    correct = 0
    i = 0
    for (x, y) in tqdm(test_loader):
        x, y = x.cuda(), y.cuda()
        if args.attack == "clean":
            _, adv, is_adv = clean(net, x, y, epsilon=args.epsilon)
        elif args.attack == "mi_fgsm":
            _, adv, is_adv = mi_fgsm(net, x, y, epsilon=args.epsilon)
        elif args.attack == "fia":
            arch = '{}_{}'.format(args.arch, args.quantize_method)
            _, adv, is_adv = fia(net, x, y, epsilon=args.epsilon, arch=arch)
        elif args.attack == "rpa":
            arch = '{}_{}'.format(args.arch, args.quantize_method)
            _, adv, is_adv = rpa(net, x, y, epsilon=args.epsilon, arch=arch)
        elif args.attack == "ci_fgsm":
            _, adv, is_adv = ci_fgsm(net, x, y, epsilon=args.epsilon)
        elif args.attack == "ssa":
            _, adv, is_adv = ssa(net, x, y, epsilon=args.epsilon, mi=False, di=False, ti=False)
        elif args.attack == 'admix':
            _, adv, is_adv = admix(net, x, y, epsilon=args.epsilon)
        else: 
            raise Exception("attack {} is not implemented!".format(args.attack))
        print("\nattack success rate: {}".format(is_adv.sum().item() / len(is_adv)))
        output = net(adv)
        pred_top1 = output.topk(k=1, largest=True).indices
        if pred_top1.dim() >= 2:
            pred_top1 = pred_top1.squeeze()
        correct += (pred_top1 == y).sum().item()
        '''save the adversarial images'''
        label_dir = './val_rs.csv'
        save_images(
            adv, i, args.batch_size, 
            label_dir=label_dir,
            output_dir=output_dir, 
            png=False
            )
        i += 1
    return correct / len(test_set)



model_configs = [
    {
        "quantize_method": "fp",
        "bit": 32,
        "arch": "vgg16",
        'stochastic': False,
        'stochastic_mode': '',
        'ckpt_id': '',
    },
    {
        "quantize_method": "qdrop",
        "bit": 5,
        "arch": "vgg16",
        'stochastic': False,
        'stochastic_mode': '',
        'ckpt_id': '',
    },
]




def main():
    print('data_source: {}'.format(args.data_source))
    if args.stochastic:
        output_dir = os.path.join("./adv_imgs", args.quantize_method, args.arch + "_w{}a{}_stochastic".format(args.w_bit, args.a_bit), args.attack)
    else:
        output_dir = os.path.join("./adv_imgs", args.quantize_method, args.arch + "_w{}a{}".format(args.w_bit, args.a_bit), args.attack)
    print("output_dir: {}".format(output_dir))

    if args.quantize_method == "merged":
        model_list = []
        for config in model_configs:
            args.quantize_method = config["quantize_method"]
            args.w_bit = config["bit"]
            args.a_bit = config["bit"]
            args.arch = config["arch"]
            args.stochastic_mode = config["stochastic_mode"]
            args.stochastic = config['stochastic']
            args.ckpt_id = config['ckpt_id']
            raw_model = load_model(args).cuda().eval()     # eval here is important
            model_list.append(raw_model)
        model = MergedModel(model_list=model_list, merge_mode=args.merge_mode)
    elif args.quantize_method == 'lgv':
        model = LGVModel(args.arch, qaa=False)
    elif args.quantize_method == 'lgv_qaa':
        model = LGVModel(args.arch, qaa=True)
    else:
        model = load_model(args)

    normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = nn.Sequential(normalize, model).eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_set = MyDataset(
        img_dir=args.data_source,
        label_dir='./val_rs.csv',
        transform=transform,
        png=False,
    )

    clean_acc = evaluate(model, val_set, use_gpu=True)
    print("clean_acc: %.3f" % (clean_acc))
    adv_acc = evaluate_adv(model, val_set, output_dir)
    print("adv_acc: %.3f" % (adv_acc))
    print("file saved at {}".format(output_dir))
    return



def attack_trump(name='xi'):
    from PIL import Image
    import torchvision
    img_trump = Image.open('./clean_{}.png'.format(name)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    tensor_trump = transform(img_trump).unsqueeze(0)
    print('shape of tensor_{}: {}'.format(name, tensor_trump.shape))
    args.quantize_method = 'apot'
    args.w_bit = 2
    args.a_bit = 2
    args.arch = 'resnet34'
    args.stochastic = True
    # model = torchvision.models.resnet50(pretrained=True)
    model = load_model(args).cuda().eval()
    normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = nn.Sequential(normalize, model).eval()
    _, adv_trump, _ = fia_ci_fgsm(model, tensor_trump, torch.tensor(0), arch='resnet34_apot', epsilon=float(32/255))
    torchvision.utils.save_image(adv_trump, './adv_{}.png'.format(name))
    print('attack finished!')
    



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()
    # test_clean_acc()
    # attack_trump()