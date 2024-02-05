import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import random
import time
from . import hubconf
from . quant import (
    block_reconstruction,
    layer_reconstruction,
    BaseQuantBlock,
    QuantModule,
    QuantModel,
    set_weight_quantize_params,
    set_act_quantize_params,
)
# from QDrop.data.imagenet import build_imagenet_data


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train(train_loader, model, optimizer):
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    correct = 0
    length = 0
    for (x, y) in train_loader:
        length += len(y)
        x, y = x.cuda(), y.cuda()
        outputs = model(x)
        pred = outputs.topk(k=1, largest=True).indices
        if pred.dim() >= 2:
            pred = pred.squeeze()
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (pred == y).sum().cpu().item()
    return correct / length

def validate(test_loader, model, gpu=True):
    if gpu:
        model = model.cuda().eval()
    else:
        model = model.cpu().eval()
    correct = 0
    length = 0
    for i, (x, y) in enumerate(test_loader):
        if gpu:
            x, y = x.cuda(), y.cuda()
        else:
            x, y = x.cpu(), y.cpu()
        outputs = model(x)
        pred = outputs.topk(k=1, largest=True).indices
        if pred.dim() >= 2:
            pred = pred.squeeze()
        correct += (pred == y).sum().cpu().item()
        length += len(y)
    return correct / length


def fine_tune(model, num_epoch, train_loader, test_loader):
    
    best_acc = 0

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    model = model.cuda()

    from tqdm import tqdm
    for epoch in tqdm(range(num_epoch)):
        train_acc = train(train_loader, model, optimizer)
        test_acc = validate(test_loader, model)
        if test_acc > best_acc:
            best_acc = test_acc
            dest_dir = os.path.join("./checkpoints", args.arch + "_32bit.pth.tar")
            torch.save(model.state_dict(), dest_dir)
        print('\ntrain_acc: {}  test_acc: {} best_acc: {}'.format(train_acc, test_acc, best_acc))
        scheduler.step()
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def validate_model(val_loader, model, device=None, print_freq=100):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def get_train_samples(train_loader, num_samples):
    train_data, target = [], []
    for batch in train_loader:
        train_data.append(batch[0])
        target.append(batch[1])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples], torch.cat(target, dim=0)[:num_samples]


def quant_model(save_dir):

    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--arch', default='resnet18', type=str, help='model name',
                        choices=['resnet18', 'resnet50', 'spring_resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')

    # quantization parameters
    parser.add_argument('--n_bits_w', default=4, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=4, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--act_quant', action='store_true', help='apply activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')

    # weight calibration parameters
    parser.add_argument('--num_samples', default=1024, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')

    parser.add_argument('--wwq', default=True, help='weight_quant for input in weight reconstruction')
    parser.add_argument('--waq', default=True, help='act_quant for input in weight reconstruction')

    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')

    # activation calibration parameters
    parser.add_argument('--lr', default=4e-5, type=float, help='learning rate for LSQ')

    parser.add_argument('--awq', action='store_true', help='weight_quant for input in activation reconstruction')
    parser.add_argument('--aaq', action='store_true', help='act_quant for input in activation reconstruction')

    parser.add_argument('--init_wmode', default='mse', type=str, choices=['minmax', 'mse', 'minmax_scale'],
                        help='init opt mode for weight')
    parser.add_argument('--init_amode', default='mse', type=str, choices=['minmax', 'mse', 'minmax_scale'],
                        help='init opt mode for activation')
    # order parameters
    parser.add_argument('--order', default='before', type=str, choices=['before', 'after', 'together'], help='order about activation compare to weight')
    parser.add_argument('--prob', default=1.0, type=float)
    parser.add_argument('--input_prob', default=1.0, type=float)
    args = parser.parse_args()

    # save_dir = os.path.join("./checkpoints", args.arch + "_w" + str(args.n_bits_w) + "a" + str(args.n_bits_a) + ".pth.tar")
    # print("model will save at: {}".format(save_dir))

    # seed_all(args.seed)
    # # build imagenet data loader
    # train_loader, test_loader = build_imagenet_data(batch_size=args.batch_size, workers=args.workers,
    #                                                 data_path=args.data_path)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform_train,
                                             download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=4)

    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform_test,
                                            download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                               num_workers=4)
    # # # load model
    # cnn = eval('hubconf.{}(pretrained=False)'.format(args.arch))
    
    # cnn.cuda()
    # cnn.eval()
    # val_acc = validate(test_loader, cnn)
    # print("accuracy of the pre-trained model: {}".format(val_acc))

    # # build quantization parameters
    # wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': args.init_wmode}
    # aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': args.init_amode,
    #              'leaf_param': True, 'prob': args.prob}

    # qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    # qnn.cuda()
    # qnn.eval()
    # if not args.disable_8bit_head_stem:
    #     print('Setting the first and the last layer to 8-bit')
    #     qnn.set_first_last_layer_to_8bit()

    # qnn.disable_network_output_quantization()
    # cali_data, cali_target = get_train_samples(train_loader, num_samples=args.num_samples)
    # device = next(qnn.parameters()).device
    # # print('the quantized model is below!')
    # # Kwargs for weight rounding calibration
    # assert args.wwq is True
    # kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight,
    #               b_range=(args.b_start, args.b_end), warmup=args.warmup, opt_mode='mse',
    #               wwq=args.wwq, waq=args.waq, order=args.order, act_quant=args.act_quant,
    #               lr=args.lr, input_prob=args.input_prob)

    # if args.act_quant and args.order == 'before' and args.awq is False:
    #     '''Case 2'''
    #     set_act_quantize_params(qnn, cali_data=cali_data, awq=args.awq, order=args.order)

    # '''init weight quantizer'''
    # set_weight_quantize_params(qnn)

    # def set_weight_act_quantize_params(module):
    #     if isinstance(module, QuantModule):
    #         layer_reconstruction(qnn, module, **kwargs)
    #     elif isinstance(module, BaseQuantBlock):
    #         block_reconstruction(qnn, module, **kwargs)
    #     else:
    #         raise NotImplementedError

    # def recon_model(model: nn.Module):
    #     """
    #     Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
    #     """
    #     for name, module in model.named_children():
    #         if isinstance(module, QuantModule):
    #             print('Reconstruction for layer {}'.format(name))
    #             set_weight_act_quantize_params(module)
    #         elif isinstance(module, BaseQuantBlock):
    #             print('Reconstruction for block {}'.format(name))
    #             set_weight_act_quantize_params(module)
    #         else:
    #             recon_model(module)
    # # Start calibration
    # recon_model(qnn)

    # if args.act_quant and args.order == 'after' and args.waq is False:
    #     '''Case 1'''
    #     set_act_quantize_params(qnn, cali_data=cali_data, awq=args.awq, order=args.order)

    # qnn.set_quant_state(weight_quant=True, act_quant=args.act_quant)
    # print('Full quantization (W{}A{}) accuracy: {}'.format(args.n_bits_w, args.n_bits_a,
    #                                                        validate_model(test_loader, qnn)))
    # torch.save(qnn.state_dict(), save_dir)



    '''load quantized model from checkpoint. by yyl'''
    cnn = eval('hubconf.{}(pretrained=True)'.format(args.arch)).cuda().eval()
    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': args.init_wmode}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': args.init_amode,
                 'leaf_param': True, 'prob': args.prob}
    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    if not args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()

    qnn.disable_network_output_quantization()
    cali_data, cali_target = get_train_samples(train_loader, num_samples=args.num_samples)
    device = next(qnn.parameters()).device
    # print('the quantized model is below!')
    # Kwargs for weight rounding calibration
    assert args.wwq is True
    kwargs = dict(cali_data=cali_data, iters=0, weight=args.weight,
                  b_range=(args.b_start, args.b_end), warmup=args.warmup, opt_mode='mse',
                  wwq=args.wwq, waq=args.waq, order=args.order, act_quant=args.act_quant,
                  lr=args.lr, input_prob=args.input_prob)

    if args.act_quant and args.order == 'before' and args.awq is False:
        '''Case 2'''
        set_act_quantize_params(qnn, cali_data=cali_data, awq=args.awq, order=args.order)

    '''init weight quantizer'''
    set_weight_quantize_params(qnn)

    def set_weight_act_quantize_params(module):
        if isinstance(module, QuantModule):
            layer_reconstruction(qnn, module, **kwargs)
        elif isinstance(module, BaseQuantBlock):
            block_reconstruction(qnn, module, **kwargs)
        else:
            raise NotImplementedError

    def recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                print('Reconstruction for layer {}'.format(name))
                set_weight_act_quantize_params(module)
            elif isinstance(module, BaseQuantBlock):
                print('Reconstruction for block {}'.format(name))
                set_weight_act_quantize_params(module)
            else:
                recon_model(module)
    # Start calibration
    recon_model(qnn)
    print("quantization finished!")

    if args.act_quant and args.order == 'after' and args.waq is False:
        '''Case 1'''
        set_act_quantize_params(qnn, cali_data=cali_data, awq=args.awq, order=args.order)

    qnn.set_quant_state(weight_quant=True, act_quant=args.act_quant)
    qnn.load_state_dict(torch.load(save_dir))
    print("checkpoint successfully loaded from {}".format(save_dir))
    acc = validate(test_loader, qnn)
    print("val_acc: {}".format(acc))
    return qnn


if __name__ == "main":
    quant_model()