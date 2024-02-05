from distutils.dep_util import newer_group
import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
from tqdm import tqdm


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

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


"""
adversarial attacks: 
Input: model, x, y, and other attack algorithm hyper-parameters 
Output: x, adv, is_adv
---------------------------------------------------------------
clean: No attack
white-box attack: fgsm, pgd, cw pgd. epsilon by default = 8/255
transfer attack: mi-fgsm, vmi-fgsm, vmi-ci,fgsm, fia. epsilon by default = 16/255
"""

def clean(model, x, y, epsilon=float(8/255), targeted=False):
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    adv = x.clone()
    # adv.requires_grad = True
    # outputs = model(adv)
    # loss = F.cross_entropy(outputs, y)
    # loss.backward()
    # adv = adv + epsilon * adv.grad.sign()
    # adv = torch.clamp(adv, 0.0, 1.0).detach()

    "validate in memory"
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()
    
    is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv


def fgsm(model, x, y, epsilon=float(8/255), targeted=False):
    """
    reference: Goodfellow I J, Shlens J, Szegedy C. 
    Explaining and harnessing adversarial examples[J]. 
    arXiv preprint arXiv:1412.6572, 2014.
    """
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    adv = x.clone()
    adv.requires_grad = True
    outputs = model(adv)
    loss = F.cross_entropy(outputs, y)
    loss.backward()
    if targeted:
        adv = adv - epsilon * adv.grad.sign()
    else:
        adv = adv + epsilon * adv.grad.sign()
    adv = torch.clamp(adv, 0.0, 1.0).detach()

    "validate in memory"
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv



def r_fgsm(model, x, y, epsilon=float(8/255), targeted=False):
    """
    reference: Tramèr F, Kurakin A, Papernot N, et al. Ensemble adversarial training: Attacks and defenses[C]. ICLR 2018.
    """
    x, y, model= x.cuda(), y.cuda(), model.cuda().eval()

    adv = x.clone()
    adv += (torch.rand_like(adv) * 2 - 1) * epsilon / 2
    adv = torch.clamp(adv, 0.0, 1.0).detach()

    min_x = x - epsilon
    max_x = x + epsilon

    adv.requires_grad = True
    outputs = model(adv)
    loss = F.cross_entropy(outputs, y)
    loss.backward()
    if targeted:
        adv = adv - epsilon * adv.grad.sign()
    else:
        adv = adv + epsilon * adv.grad.sign()
        
    adv = torch.clamp(adv, 0.0, 1.0).detach()
    adv = torch.max(torch.min(adv, max_x), min_x).detach()

    "validate in memory"
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    if y.dim() >= 2:
        '''switch one-hot label to class label'''
        target = y.topk(k=1, largest=True).indices
        is_adv = (pred_top1 == target) if targeted else (pred_top1 != target)
    else:
        is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv


def bim(model, x, y, epsilon=float(16/255), num_steps=10, targeted=False):
    """
    reference: Dong Y, Liao F, Pang T, et al. 
    Boosting adversarial attacks with momentum[C]//
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 9185-9193.
    """
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    alpha = epsilon / num_steps   # attack step size
    min_x = x - epsilon
    max_x = x + epsilon

    adv = x.clone()

    for _ in range(num_steps):
        adv.requires_grad = True
        outputs = model(adv)
        loss = F.cross_entropy(outputs, y)
        loss.backward()
        if targeted:
            adv = adv + alpha * adv.grad.sign()
        else:
            adv = adv + alpha * adv.grad.sign()

        adv = torch.clamp(adv, 0.0, 1.0).detach()
        adv = torch.max(torch.min(adv, max_x), min_x).detach()

    '''validate in memory'''
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv    



def pgd(model, x, y, epsilon=float(8/255), num_steps=20, step_size=float(2/255), targeted=False):
    """
    reference: Madry A, Makelov A, Schmidt L, et al. 
    Towards deep learning models resistant to adversarial attacks[J]. 
    arXiv preprint arXiv:1706.06083, 2017.
    """
    x, y, model= x.cuda(), y.cuda(), model.cuda().eval()

    adv = x.clone()
    adv += (torch.rand_like(adv) * 2 - 1) * epsilon
    adv = torch.clamp(adv, 0.0, 1.0).detach()

    min_x = x - epsilon
    max_x = x + epsilon

    for _ in range(num_steps):
        adv.requires_grad = True
        outputs = model(adv)
        loss = F.cross_entropy(outputs, y)
        loss.backward()
        if targeted:
            adv = adv - step_size * adv.grad.sign()
        else:
            adv = adv + step_size * adv.grad.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()
        adv = torch.max(torch.min(adv, max_x), min_x).detach()

    '''validate in memory'''
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv


def odi(model, X, y, epsilon=float(8/255), num_steps=20, step_size=float(2/255), ODI_num_steps=20, ODI_step_size=float(8/255), random=True, lossFunc='margin'):

    def margin_loss(logits,y):

        logit_org = logits.gather(1,y.view(-1,1))
        logit_target = logits.gather(1,(logits - torch.eye(10)[y].to("cuda") * 9999).argmax(1, keepdim=True))
        loss = -logit_org + logit_target
        loss = torch.sum(loss)
        return loss

    model, X, y = model.cuda(), X.cuda(), y.cuda()
    X_pgd = Variable(X.data, requires_grad=True).cuda()

    randVector_ = torch.FloatTensor(*model(X_pgd).shape).uniform_(-1.,1.).cuda()
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for i in range(ODI_num_steps + num_steps):
        opt = torch.optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            if i < ODI_num_steps:
                loss = (model(X_pgd) * randVector_).sum()
            elif lossFunc == 'xent':
                loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            else:
                loss = margin_loss(model(X_pgd),y)
        loss.backward()
        if i < ODI_num_steps: 
            eta = ODI_step_size * X_pgd.grad.data.sign()
        else:
            eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    is_adv = (model(X_pgd).data.max(1)[1] != y.data).detach().cpu().numpy() 
    return X, X_pgd.detach().clone(), is_adv


def cw_pgd(model, x, y, epsilon=float(8/255), num_steps=20, targeted=False):
    """
    pgd attack with cw loss, untargeted.
    reference: Carlini N, Wagner D. 
    Towards evaluating the robustness of neural networks[C]
    //2017 ieee symposium on security and privacy (sp). IEEE, 2017: 39-57.
    """
    x, y, model= x.cuda(), y.cuda(), model.cuda().eval()

    adv = x.clone()
    adv += (torch.rand_like(adv) * 2 - 1) * epsilon
    adv = torch.clamp(adv, 0.0, 1.0).detach()

    min_x = x - epsilon
    max_x = x + epsilon

    one_hot_y = torch.zeros(y.size(0), 10).cuda()
    one_hot_y[torch.arange(y.size(0)), y] = 1

    for _ in range(num_steps):
        adv.requires_grad = True
        outputs = model(adv)
        correct_logit = torch.sum(one_hot_y * outputs, dim=1)
        wrong_logit, _ = torch.max((1 - one_hot_y) * outputs - 1e4 * one_hot_y, dim=1)
        loss = -torch.sum(F.relu(correct_logit - wrong_logit + 50))
        loss.backward()
        if targeted:
            adv = adv - 0.00392 * adv.grad.sign()
        else:
            adv = adv + 0.00392 * adv.grad.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()
        adv = torch.max(torch.min(adv, max_x), min_x).detach()

    '''validate in memory'''
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv


def adaptive_pgd(model, x, y, epsilon=float(8/255), max_iters=20, target=None, _type='linf', gpu_idx=None):

    def predict_from_logits(logits, dim=1):
        return logits.max(dim=dim, keepdim=False)[1]

    def check_oscillation(x, j, k, y5, k3=0.5):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]
        return t <= k*k3*np.ones(t.shape)
    
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()
    adv = x.clone()
    pred = predict_from_logits(model(x))
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    # print(x.shape)

    max_x = x + epsilon
    min_x = x - epsilon

    x.requires_grad = True 
 
    n_iter_2, n_iter_min, size_decr = max(int(0.22 * max_iters), 1), max(int(0.06 * max_iters), 1), max(int(0.03 * max_iters), 1)
    '''random initialization'''
    t = 2 * torch.rand(x.shape).cuda().detach() - 1
    x_adv = x.detach() + epsilon * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
    x_adv = torch.clamp(torch.min(torch.max(x_adv, min_x), max_x), 0.0, 1.0)

    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([max_iters, x.shape[0]])
    loss_best_steps = torch.zeros([max_iters + 1, x.shape[0]])
    acc_steps = torch.zeros_like(loss_best_steps)
    
    x_adv.requires_grad_()
    with torch.enable_grad():
        logits = model(x_adv) # 1 forward pass (eot_iter = 1)
        if target is not None:
            loss_indiv = -F.cross_entropy(logits, target, reduce=False)
        else:
            loss_indiv = F.cross_entropy(logits, y, reduce=False)
        loss = loss_indiv.sum()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    
    grad_best = grad.clone()
    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()

    step_size = epsilon * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * torch.Tensor([2.0]).cuda().detach().reshape([1, 1, 1, 1])
    x_adv_old = x_adv.clone()

    k = n_iter_2 + 0
    u = np.arange(x.shape[0])
    counter3 = 0
    
    loss_best_last_check = loss_best.clone()
    reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
    n_reduced = 0

    for i in range(max_iters):
        with torch.no_grad():
            x_adv = x_adv.detach()
            x_adv_old = x_adv.clone()
            
            x_adv_1 = x_adv + step_size * torch.sign(grad)
            x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - epsilon), x + epsilon), 0.0, 1.0)
            x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv), x - epsilon), x + epsilon), 0.0, 1.0)
            x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, min_x), max_x), 0.0, 1.0)

            x_adv = x_adv_1 + 0.
        
        x_adv.requires_grad_()

        with torch.enable_grad():
            logits = model(x_adv) # 1 forward pass (eot_iter = 1)
            if target is not None:
                loss_indiv = -F.cross_entropy(logits, target, reduce=False)
            else:
                loss_indiv = F.cross_entropy(logits, y, reduce=False)
            loss = loss_indiv.sum()
        
        grad = torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.

        ### check step size
        with torch.no_grad():
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1.cpu() + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0
            
            counter3 += 1
        
            if counter3 == k:
                fl_oscillation = check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=.75)
                fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                reduced_last_check = np.copy(fl_oscillation)
                loss_best_last_check = loss_best.clone()
                
                if np.sum(fl_oscillation) > 0:
                    step_size[u[fl_oscillation]] /= 2.0
                    n_reduced = fl_oscillation.astype(float).sum()
                    
                    fl_oscillation = np.where(fl_oscillation)
                    
                    x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                    grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                    
                counter3 = 0
                k = np.maximum(k - size_decr, n_iter_min)


    adv = x_best_adv
    now_p = x_best_adv-x

    '''validate in memory'''
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    is_adv = (pred_top1 != y)

    return x, adv, is_adv



def mi_fgsm(model, x, y, epsilon=float(16/255), num_steps=10, targeted=False):
    """
    reference: Dong Y, Liao F, Pang T, et al. 
    Boosting adversarial attacks with momentum[C]//
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 9185-9193.
    """
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    alpha = epsilon / num_steps   # attack step size
    momentum = 1.0
    grads = torch.zeros_like(x, requires_grad=False)
    min_x = x - epsilon
    max_x = x + epsilon

    adv = x.clone()

    for _ in range(num_steps):
        adv.requires_grad = True
        outputs = model(adv)
        loss = F.cross_entropy(outputs, y)
        loss.backward()
        new_grad = adv.grad
        noise = momentum * grads + (new_grad) / torch.norm(new_grad, dim=[1,2,3], p=1, keepdim=True)
        if targeted:
            adv = adv + alpha * noise.sign()
        else:
            adv = adv + alpha * noise.sign()

        adv = torch.clamp(adv, 0.0, 1.0).detach()
        adv = torch.max(torch.min(adv, max_x), min_x).detach()
        grads = noise

    '''validate in memory'''
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv


def vmi_fgsm(model, x, y, epsilon=float(16/255), targeted=False):
    """
    reference: Wang X, He K. 
    Enhancing the transferability of adversarial attacks through variance tuning[C]//
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 1924-1933.
    """
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    num_steps = 10
    alpha = epsilon / num_steps   # attack step size
    momentum = 1.0
    number = 20
    beta = 1.5
    grads = torch.zeros_like(x, requires_grad=False)
    variance = torch.zeros_like(x, requires_grad=False)
    min_x = x - epsilon
    max_x = x + epsilon

    adv = x.clone()

    for _ in range(num_steps):
        adv.requires_grad = True
        outputs = model(adv)
        loss = F.cross_entropy(outputs, y)
        new_grad = torch.autograd.grad(loss, adv, grad_outputs=None, only_inputs=True)[0]
        noise = new_grad + variance
        noise = momentum * grads + noise / torch.norm(noise, p=1)

        # update variance
        sample = adv.clone().detach()
        global_grad = torch.zeros_like(x, requires_grad=False)
        for _ in range(number):
            sample = sample.detach()
            sample.requires_grad = True
            rd = (torch.rand_like(x) * 2 - 1) * beta * epsilon
            sample = sample + rd
            outputs_sample = model(sample)
            loss_sample = F.cross_entropy(outputs_sample, y)
            grad_vanilla_sample = torch.autograd.grad(loss_sample, sample, grad_outputs=None, only_inputs=True)[0]
            global_grad += grad_vanilla_sample
        variance = global_grad / (number * 1.0) - new_grad

        if targeted:
            adv = adv - alpha * noise.sign()
        else:
            adv = adv + alpha * noise.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()   # range [0, 1]
        adv = torch.max(torch.min(adv, max_x), min_x).detach()
        grads = noise

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()
    is_adv = (pred_top1 != y)

    return x, adv, is_adv




def vmi_ci_fgsm(model, x, y, epsilon=float(16/255), dataset="ImageNet", targeted=False):
    """
    reference: Wang X, He K. 
    Enhancing the transferability of adversarial attacks through variance tuning[C]//
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 1924-1933.
    """
    def input_diversity(input_tensor):
        """apply input transformation to enhance transferability: padding and resizing (DIM)"""
        if dataset == "CIFAR":
            image_width = 32
            image_height = 32
            image_resize = 34
        
        elif dataset == "ImageNet":
            image_width = 299
            image_height = 299
            image_resize = 331
        prob = 0.5        # probability of using diverse inputs

        rnd = torch.randint(image_width, image_resize, ())   # uniform distribution
        rescaled = F.interpolate(input_tensor, size=[rnd, rnd], mode='nearest')
        h_rem = image_resize - rnd
        w_rem = image_resize - rnd
        pad_top = torch.randint(0, h_rem, ())
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_rem, ())
        pad_right = w_rem - pad_left
        # pad的参数顺序在pytorch里面是左右上下，在tensorflow里是上下左右，而且要注意pytorch的图像格式是BCHW, tensorflow是CHWB
        padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom, 0, 0, 0, 0))
        if torch.rand(1) < prob:
            ret = padded
        else:
            ret = input_tensor
        ret = F.interpolate(ret, [image_height, image_width], mode='nearest')
        return ret

    def gkern(kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    kernel = gkern(7, 3).astype(np.float32)
    # 要注意Pytorch是BCHW, tensorflow是BHWC
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)  # batch, channel, height, width = 3, 1, 7, 7
    stack_kernel = torch.tensor(stack_kernel).cuda()

    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    num_steps = 10
    alpha = epsilon / num_steps   # attack step size
    momentum = 1.0
    number = 20
    beta = 1.5
    grads = torch.zeros_like(x, requires_grad=False)
    variance = torch.zeros_like(x, requires_grad=False)
    min_x = x - epsilon
    max_x = x + epsilon

    adv = x.clone()
    y_batch = torch.cat((y, y, y, y, y), dim=0)

    for _ in range(num_steps):
        adv.requires_grad = True
        x_batch = torch.cat((adv, adv / 2., adv / 4., adv / 8., adv / 16.), dim=0)
        outputs = model(input_diversity(x_batch))
        loss = F.cross_entropy(outputs, y_batch)
        grad_vanilla = torch.autograd.grad(loss, x_batch, grad_outputs=None, only_inputs=True)[0]
        grad_batch_split = torch.split(grad_vanilla, split_size_or_sections=len(y), dim=0)
        grad_in_batch = torch.stack(grad_batch_split, dim=4)
        new_grad = torch.sum(grad_in_batch * torch.tensor([1., 1 / 2., 1 / 4., 1 / 8, 1 / 16.]).cuda(), dim=4, keepdim=False)
        
        current_grad = new_grad + variance
        noise = F.conv2d(input=current_grad, weight=stack_kernel, stride=1, padding=3, groups=3)
        noise = momentum * grads + noise / torch.norm(noise, p=1)

        # update variance
        sample = x_batch.clone().detach()
        global_grad = torch.zeros_like(x, requires_grad=False)
        for _ in range(number):
            sample = sample.detach()
            sample.requires_grad = True
            rd = (torch.rand_like(x) * 2 - 1) * beta * epsilon
            rd_batch = torch.cat((rd, rd / 2., rd / 4., rd / 8., rd / 16.), dim=0)
            sample = sample + rd_batch
            outputs_sample = model(input_diversity(sample))
            loss_sample = F.cross_entropy(outputs_sample, y_batch)
            grad_vanilla_sample = torch.autograd.grad(loss_sample, sample, grad_outputs=None, only_inputs=True)[0]
            grad_batch_split_sample = torch.split(grad_vanilla_sample, split_size_or_sections=len(y),
                                                    dim=0)
            grad_in_batch_sample = torch.stack(grad_batch_split_sample, dim=4)
            global_grad += torch.sum(grad_in_batch_sample * torch.tensor([1., 1 / 2., 1 / 4., 1 / 8, 1 / 16.]).cuda(), dim=4, keepdim=False)
        variance = global_grad / (number * 1.0) - new_grad

        if targeted:
            adv = adv - alpha * noise.sign()
        else:
            adv = adv + alpha * noise.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()  
        adv = torch.max(torch.min(adv, max_x), min_x).detach()
        grads = noise

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv


def ci_fgsm(model, x, y, epsilon=float(16/255), dataset="cifar10", targeted=False):
    """
    reference: Wang X, He K. 
    Enhancing the transferability of adversarial attacks through variance tuning[C]//
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 1924-1933.
    """
    def input_diversity(input_tensor):
        """apply input transformation to enhance transferability: padding and resizing (DIM)"""
        if dataset == "cifar10":
            image_width = 32
            image_height = 32
            image_resize = 34
        
        elif dataset == "imagenet":
            image_width = 299
            image_height = 299
            image_resize = 331
        prob = 0.5        # probability of using diverse inputs

        rnd = torch.randint(image_width, image_resize, ())   # uniform distribution
        rescaled = F.interpolate(input_tensor, size=[rnd, rnd], mode='nearest')
        h_rem = image_resize - rnd
        w_rem = image_resize - rnd
        pad_top = torch.randint(0, h_rem, ())
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_rem, ())
        pad_right = w_rem - pad_left
        # pad的参数顺序在pytorch里面是左右上下，在tensorflow里是上下左右，而且要注意pytorch的图像格式是BCHW, tensorflow是CHWB
        padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom, 0, 0, 0, 0))
        if torch.rand(1) < prob:
            ret = padded
        else:
            ret = input_tensor
        ret = F.interpolate(ret, [image_height, image_width], mode='nearest')
        return ret

    def gkern(kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    kernel = gkern(7, 3).astype(np.float32)
    # 要注意Pytorch是BCHW, tensorflow是BHWC
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)  # batch, channel, height, width = 3, 1, 7, 7
    stack_kernel = torch.tensor(stack_kernel).cuda()

    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    num_steps = 10
    alpha = epsilon / num_steps   # attack step size
    momentum = 1.0
    number = 20
    beta = 15
    grads = torch.zeros_like(x, requires_grad=False)
    # variance = torch.zeros_like(x, requires_grad=False)
    min_x = x - epsilon
    max_x = x + epsilon

    adv = x.clone()
    y_batch = torch.cat((y, y, y, y, y), dim=0)


    for _ in range(num_steps):
        adv.requires_grad = True
        x_batch = torch.cat((adv, adv / 2., adv / 4., adv / 8., adv / 16.), dim=0)
        outputs = model(input_diversity(x_batch))
        loss = F.cross_entropy(outputs, y_batch)
        grad_vanilla = torch.autograd.grad(loss, x_batch, grad_outputs=None, only_inputs=True)[0]
        grad_batch_split = torch.split(grad_vanilla, split_size_or_sections=len(y), dim=0)
        grad_in_batch = torch.stack(grad_batch_split, dim=4)
        new_grad = torch.sum(grad_in_batch * torch.tensor([1., 1 / 2., 1 / 4., 1 / 8, 1 / 16.]).cuda(), dim=4, keepdim=False)
        
        # current_grad = new_grad + variance
        current_grad = new_grad
        noise = F.conv2d(input=current_grad, weight=stack_kernel, stride=1, padding=3, groups=3)
        noise = momentum * grads + noise / torch.norm(noise, p=1)
        
        if targeted:
            adv = adv - alpha * noise.sign()
        else:
            adv = adv + alpha * noise.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()  
        adv = torch.max(torch.min(adv, max_x), min_x).detach()
        grads = noise

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv


def fia(model, x, y, epsilon=float(16/255), arch="vgg16", dataset='cifar10'):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook

    model, x, y = model.cuda().eval(), x.cuda(), y.cuda()

    if dataset == 'cifar10':
        num_classes = 10 
    elif dataset == 'imagenet':
        num_classes = 1000
    else:
        raise Exception('dataset {} not implemented!'.format(dataset))
    
    momentum = 1.0
    num_steps = 10
    num_ens = 30
    probb = 0.9
    alpha = epsilon / num_steps

    gradients = torch.zeros_like(x, requires_grad=False)
    min_x = x - epsilon
    max_x = x + epsilon


    if arch == "vgg16_fp":
        model[1].features[15].register_forward_hook(get_activation("features"))
    elif arch == "vgg16_bn_fp":
        model[1].features[22].register_forward_hook(get_activation("features"))
    elif arch == "vgg16_qdrop":
        model[1].model.features[15].register_forward_hook(get_activation("features"))
    elif arch == 'resnet20_fp' or arch == 'resnet56_fp' or arch == 'resnet50_fp':
        model[1].module.layer2.register_forward_hook(get_activation("features"))
    elif arch == 'resnet18_qdrop' or arch == 'resnet50_qdrop':
        model[1].model.layer2.register_forward_hook(get_activation("features"))
    elif arch == 'resnet20_apot' or arch == 'resnet56_apot':
        model[1].module.layer2.register_forward_hook(get_activation("features"))
    else:
        raise Exception("arch {} not implemented!".format(arch))


    # initializing weights as 0
    outputs = model(x)
    features = activation["features"].detach()
    weights = torch.zeros_like(features)

    for _ in range(num_ens):
        x.requires_grad = True
        mask = torch.bernoulli(torch.ones_like(x) * probb)
        input = torch.mul(x, mask)   # element-wise product
        logits = model(input)
        label_one_hot = torch.nn.functional.one_hot(y, num_classes).float().cuda().squeeze()
        features = activation["features"]
        weights += torch.autograd.grad(torch.mul(logits, label_one_hot).sum(), features)[0].detach()
    weights /= torch.norm(weights, dim=[1,2,3], p=2, keepdim=True)

    adv = x.clone().detach()
    for _ in range(num_steps):
        adv.requires_grad = True
        logits = model(adv)
        features = activation["features"]
        loss = torch.mul(weights, features).sum()
        loss.backward()
        new_grad = adv.grad
        gradients = momentum * gradients + new_grad / torch.norm(new_grad, dim=[1,2,3], p=1, keepdim=True)
        adv = adv - alpha * gradients.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()
        adv = torch.max(torch.min(adv, max_x), min_x).detach()
        
    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)



def rpa(model, x, y, epsilon=float(16/255), arch="vgg16", dataset='cifar10'):
    '''
    Reference: Zhang Y, Tan Y, Chen T, et al. Enhancing the Transferability of Adversarial Examples with Random Patch[C] IJCAI'21.
    '''
    num_iter = 10
    alpha = epsilon / num_iter
    momentum = 1.0

    if dataset == 'cifar10':
        num_classes = 10
        image_size = 32
    elif dataset == 'imagenet':
        num_classes = 1000
        image_size = 299
    ens = 60
    probb = 0.7

    gradients = torch.zeros_like(x, requires_grad=False)
    min_x = x - epsilon
    max_x = x + epsilon

    batch_shape = [len(y), 3, image_size, image_size]

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook

    if arch == "vgg16_fp":
        model[1].features[15].register_forward_hook(get_activation("features"))
    elif arch == "vgg16_bn_fp":
        model[1].features[22].register_forward_hook(get_activation("features"))
    elif arch == "vgg16_qdrop":
        model[1].model.features[15].register_forward_hook(get_activation("features"))
    elif arch == 'resnet20_fp' or arch == 'resnet56_fp' or arch == 'resnet50_fp':
        model[1].module.layer2.register_forward_hook(get_activation("features"))
    elif arch == 'resnet18_qdrop' or arch == 'resnet50_qdrop':
        model[1].model.layer2.register_forward_hook(get_activation("features"))
    elif arch == 'resnet20_apot' or arch == 'resnet56_apot':
        model[1].module.layer2.register_forward_hook(get_activation("features"))
    else:
        raise Exception("arch {} not implemented!".format(arch))


    def patch_by_strides(img_shape, patch_size, prob):
        img_shape = (img_shape[0], img_shape[2], img_shape[3], img_shape[1])  # from pytorch (BCHW) to tf (BHWC)
        X_mask = np.ones(img_shape)
        N0, H0, W0, C0 = X_mask.shape
        ph = H0 // patch_size[0]
        pw = W0 // patch_size[1]
        X = X_mask[:, :ph * patch_size[0], :pw * patch_size[1]]
        N, H, W, C = X.shape
        shape = (N, ph, pw, patch_size[0], patch_size[1], C)
        strides = (X.strides[0], X.strides[1] * patch_size[0], X.strides[2] * patch_size[0], *X.strides[1:])
        mask_patchs = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        mask_len = mask_patchs.shape[1] * mask_patchs.shape[2] * mask_patchs.shape[-1]
        ran_num = int(mask_len * (1 - prob))
        rand_list = np.random.choice(mask_len, ran_num, replace=False)
        for i in range(mask_patchs.shape[1]):
            for j in range(mask_patchs.shape[2]):
                for k in range(mask_patchs.shape[-1]):
                    if i * mask_patchs.shape[2] * mask_patchs.shape[-1] + j * mask_patchs.shape[-1] + k in rand_list:
                        mask_patchs[:, i, j, :, :, k] = np.random.uniform(0, 1,
                                                                        (N, mask_patchs.shape[3], mask_patchs.shape[4]))
        img2 = np.concatenate(mask_patchs, axis=0, )
        img2 = np.concatenate(img2, axis=1)
        img2 = np.concatenate(img2, axis=1)
        img2 = img2.reshape((N, H, W, C))
        X_mask[:, :ph * patch_size[0], :pw * patch_size[1]] = img2
        return X_mask.swapaxes(1, 3)   # from tf to pytorch


    # initializing weights as 0
    outputs = model(x)
    features = activation["features"].detach()
    weights = torch.zeros_like(features)
    for l in range(ens):
        if l % 1 == 0:
            mask1 = np.random.binomial(1, probb, size=(batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3]))
            mask2 = np.random.uniform(0, 1, size=(batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3]))
            mask = np.where(mask1 == 1, 1, mask2)
        elif l % 3 == 1:
            mask = patch_by_strides((batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3]), (3, 3), probb)
        elif l % 3 == 2:
            mask = patch_by_strides((batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3]), (5, 5), probb)
        else:
            mask = patch_by_strides((batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3]), (7, 7), probb)
        mask = torch.tensor(mask, dtype=torch.float32).cuda()
        images_tmp2 = torch.mul(x, mask)
        images_tmp2.requires_grad = True

        logits = model(images_tmp2)
        features = activation["features"]
        label_one_hot = torch.nn.functional.one_hot(y, num_classes).float().cuda().squeeze()
        weights += torch.autograd.grad(torch.mul(logits, label_one_hot).sum(), features)[0].detach()
    weights /= torch.norm(weights, dim=[1,2,3], p=2, keepdim=True)


    adv = x.clone().detach()
    for _ in range(num_iter):
        adv.requires_grad = True
        logits = model(adv)
        features = activation["features"]
        loss = torch.mul(weights, features).sum()
        loss.backward()
        new_grad = adv.grad
        gradients = momentum * gradients + new_grad / torch.norm(new_grad, dim=[1,2,3], p=1, keepdim=True)
        adv = adv - alpha * gradients.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()
        adv = torch.max(torch.min(adv, max_x), min_x).detach()
        
    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)


def evaluate(net, test_set, output_dir=None):
    net.eval().cuda()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=20, shuffle=False,
                                              num_workers=4)
    correct = 0
    i = 0
    for (x, y) in tqdm(test_loader):
        x, y = x.cuda(), y.cuda()
        output = net(x)
        pred_top1 = output.topk(k=1, largest=True).indices
        if pred_top1.dim() >= 2:
            pred_top1 = pred_top1.squeeze()
        correct += (pred_top1 == y).sum().item()
    return correct / len(test_set)

def evaluate_adv(net, test_set, output_dir=None):
    net.eval().cuda()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=20, shuffle=False,
                                              num_workers=4)
    correct = 0
    i = 0
    for (x, y) in tqdm(test_loader):
        x, y = x.cuda(), y.cuda()
        # _, adv, is_adv = clean(net, x, y)
        _, adv, is_adv = fgsm(net, x, y)
        # _, adv, is_adv = pgd(net, x, y)
        # _, adv, is_adv = mi_fgsm(net, x, y)
        # _, adv, is_adv = cw_pgd(net, x, y)
        # _, adv, is_adv = vmi_fgsm(net, x, y)
        # _, adv, is_adv = vmi_ci_fgsm(net, x, y)
        # _, adv, is_adv = fia_attack(net, x, y)
        print("\nattack success rate: {}".format(is_adv.sum().item() / len(is_adv)))
        output = net(adv)
        pred_top1 = output.topk(k=1, largest=True).indices
        if pred_top1.dim() >= 2:
            pred_top1 = pred_top1.squeeze()
        correct += (pred_top1 == y).sum().item()
    return correct / len(test_set)

def main():
    # from vgg import VGG
    # model = VGG("VGG16")
    # model.load_state_dict(torch.load("./vgg_32bit.pth.tar"))
    # normalize = NormalizeByChannelMeanStd(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    # model = nn.Sequential(normalize, model)

    model = torchvision.models.mobilenet_v2(pretrained=True)
    normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = nn.Sequential(normalize, model)


    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # test_set = torchvision.datasets.CIFAR10(
    #     root="./data",
    #     train=False,
    #     download=True,
    #     transform=transform,
    # )
    from utils import MyDataset
    test_set = MyDataset(transform=transform)
    clean_acc = evaluate(model, test_set)
    adv_acc = evaluate_adv(model, test_set)
    print("clean_acc: {}".format(clean_acc))
    print("adv_acc: {}".format(adv_acc))
    return


if __name__ == "__main__":
    main()