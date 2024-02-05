from distutils.dep_util import newer_group
import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

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

def clean(model, x, y, epsilon=float(8/255)):
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    adv = x.clone()

    "validate in memory"
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)


def fgsm(model, x, y, epsilon=float(8/255)):
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
    adv = adv + epsilon * adv.grad.sign()
    adv = torch.clamp(adv, 0.0, 1.0).detach()

    "validate in memory"
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)


def pgd(model, x, y, epsilon=float(8/255), num_steps=7, step_size=float(2/255)):
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
        adv = adv + step_size * adv.grad.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()
        adv = torch.max(torch.min(adv, max_x), min_x).detach()

    '''validate in memory'''
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)



def mi_fgsm(model, x, y, epsilon=float(16/255), num_steps=10):
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
        adv = adv + alpha * noise.sign()

        adv = torch.clamp(adv, 0.0, 1.0).detach()
        adv = torch.max(torch.min(adv, max_x), min_x).detach()
        grads = noise

    '''validate in memory'''
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)


def vmi_fgsm(model, x, y, epsilon=float(16/255)):
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
    beta = 15
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

        adv = adv + alpha * noise.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()   # range [0, 1]
        adv = torch.max(torch.min(adv, max_x), min_x).detach()
        grads = noise

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()


    return x, adv, (pred_top1 != y)




def vmi_ci_fgsm(model, x, y, epsilon=float(16/255), dataset="ImageNet"):
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
    beta = 15
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

        adv = adv + alpha * noise.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()  
        adv = torch.max(torch.min(adv, max_x), min_x).detach()
        grads = noise

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)



def fia(model, x, y, epsilon=float(16/255), arch="vgg16", dataset='imagenet'):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook

    model, x, y = model.cuda().eval(), x.cuda(), y.cuda()
    
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'imagenet':
        num_classes = 1000  # for ImageNet
    else:
        raise Exception('dataset {} not specified!'.format(dataset))
    
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
    elif arch == 'resnet18_fp' or arch == 'resnet34_fp' or arch == 'resnet50_fp':
        model[1].layer2.register_forward_hook(get_activation("features"))
    elif arch == 'resnet18_qdrop' or arch == 'resnet50_qdrop':
        model[1].model.layer2.register_forward_hook(get_activation("features"))
    elif arch == 'resnet18_apot' or arch == 'resnet34_apot':
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


def ci_fgsm(model, x, y, epsilon=float(16/255), dataset="imagenet"):
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
    grads = torch.zeros_like(x, requires_grad=False)
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

        
        current_grad = new_grad
        noise = F.conv2d(input=current_grad, weight=stack_kernel, stride=1, padding=3, groups=3)
        noise = momentum * grads + noise / torch.norm(noise, p=1)

        adv = adv + alpha * noise.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()  
        adv = torch.max(torch.min(adv, max_x), min_x).detach()
        grads = noise

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)


def rpa(model, x, y, epsilon=float(16/255), arch="vgg16", dataset='imagenet'):
    '''
    Reference: Zhang Y, Tan Y, Chen T, et al. Enhancing the Transferability of Adversarial Examples with Random Patch[C] IJCAI'21.
    Wang X, He K. Enhancing the transferability of adversarial attacks through variance tuning[C]//
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 1924-1933.
    '''
    min_x = x - epsilon
    max_x = x + epsilon

    adv = x.clone()

    num_iter = 10
    alpha = epsilon / num_iter
    momentum = 1.0

    if dataset == 'cifar10':
        num_classes = 10
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
    elif arch == 'resnet18_fp' or arch == 'resnet34_fp' or arch == 'resnet50_fp':
        model[1].layer2.register_forward_hook(get_activation("features"))
    elif arch == 'resnet18_qdrop' or arch == 'resnet50_qdrop':
        model[1].model.layer2.register_forward_hook(get_activation("features"))
    elif arch == 'resnet18_apot' or arch == 'resnet34_apot':
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


def ssa(model, x, y, epsilon=float(16/255), dataset="imagenet", mi=False, di=False, ti=False):

    def DI(x, resize_rate=1.15, diversity_prob=0.5):
        assert resize_rate >= 1.0
        assert diversity_prob >= 0.0 and diversity_prob <= 1.0
        img_size = x.shape[-1]
        img_resize = int(img_size * resize_rate)
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
        ret = padded if torch.rand(1) < diversity_prob else x
        return ret

    def gkern(kernlen=15, nsig=3):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel.astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
        gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
        return gaussian_kernel

    def dct(x, norm=None):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = torch.fft.fft(v)

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
        V = Vc.real * W_r - Vc.imag * W_i
        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)
        return V


    def idct(X, norm=None):
        """
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

        Our definition of idct is that idct(dct(x)) == x

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the inverse DCT-II of the signal over the last dimension
        """

        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
        tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
        v = torch.fft.ifft(tmp)

        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape).real


    def dct_2d(x, norm=None):
        """
        2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last 2 dimensions
        """
        X1 = dct(x, norm=norm)
        X2 = dct(X1.transpose(-1, -2), norm=norm)
        return X2.transpose(-1, -2)


    def idct_2d(X, norm=None):
        """
        The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

        Our definition of idct is that idct_2d(dct_2d(x)) == x

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last 2 dimensions
        """
        x1 = idct(X, norm=norm)
        x2 = idct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)


    def clip_by_tensor(t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result


    def Spectrum_Simulation_Attack(images, gt, model, t_min, t_max, mi, di, ti):
        from torch.autograd import Variable as V
        """
        The attack algorithm of our proposed Spectrum Simulate Attack
        :param images: the input images
        :param gt: ground-truth
        :param model: substitute model
        :param mix: the mix the clip operation 
        :param max: the max the clip operation
        :return: the adversarial images
        """
        image_width = 299
        momentum = 1.0
        num_iter = 10
        eps = 16 / 255.0
        alpha = eps / num_iter
        x = images.clone()
        grad = 0
        rho = 0.5
        N = 20
        sigma = 16.0
        

        for i in range(num_iter):
            noise = 0
            for n in range(N):
                gauss = torch.randn(x.size()[0], 3, image_width, image_width) * (sigma / 255)
                gauss = gauss.cuda()
                x_dct = dct_2d(x + gauss).cuda()
                mask = (torch.rand_like(x) * 2 * rho + 1 - rho).cuda()
                x_idct = idct_2d(x_dct * mask)
                x_idct = V(x_idct, requires_grad = True)
                
                if di:
                    # DI-FGSM https://arxiv.org/abs/1803.06978
                    output_v3 = model(DI(x_idct))
                else:
                    output_v3 = model(x_idct)
                loss = F.cross_entropy(output_v3, gt)
                loss.backward()
                noise += x_idct.grad.data
            noise = noise / N

            if ti:
                # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
                T_kernel = gkern(7, 3)
                noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

            if mi:
                # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
                noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
                noise = momentum * grad + noise
                grad = noise

            x = x + alpha * torch.sign(noise)
            x = clip_by_tensor(x, t_min, t_max)
        return x.detach()


    images_min = clip_by_tensor(x - epsilon, 0.0, 1.0)
    images_max = clip_by_tensor(x + epsilon, 0.0, 1.0)

    adv = Spectrum_Simulation_Attack(images=x, gt=y, model=model, t_min=images_min, t_max=images_max, mi=mi, di=di, ti=ti)

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)


def admix(model, x, y, epsilon=float(16/255), mi=True):

    epsilon = 16.0 / 255.0
    step_size = 2.0 / 255.0
    num_iteration = 50
    check_point = 5
    multi_copies = 5
    momentum = 1.0
    grads = 0

    images = x
    labels = y

    suc = np.zeros((3, num_iteration // check_point))

    images = images.cuda()
    labels = labels.cuda()
    img = images.clone()
    for j in range(num_iteration):
        img_x = img
        img_x.requires_grad_(True) 
        input_grad = 0
        for c in range(multi_copies):
            #For a dataset of 5000 images with 5 images per class, this simple random selection will almost always (4995/5000=99.9%) yield an image from a different class.
            img_other = img[torch.randperm(img.shape[0])].view(img.size())
            logits = model(img_x + 0.2 * img_other)
            loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
            loss.backward()
            input_grad = input_grad + img_x.grad.clone()  
            
        img_x.grad.zero_()
        if mi:
            noise = momentum * grads + (input_grad) / torch.norm(input_grad, dim=[1,2,3], p=1, keepdim=True)
            grads = noise
            img = img.data + step_size * torch.sign(noise)
        else:
            img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)

    adv = img

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)



def auto_attack(model, x, y, epsilon=float(16/255)):
    from autoattack import AutoAttack

    adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard')
    adv = adversary.run_standard_evaluation(x, y, bs=len(y))

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)