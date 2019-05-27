import os
from copy import deepcopy

import torch
import torch.nn as nn
from PIL import Image

from nn_utils import LeakyConv


def save(path, model, optimizer, loss, epoch):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, path)


def load(path):
    checkpoint = torch.load(path)
    model_weights = checkpoint['model']
    opt_state = checkpoint['optimizer']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model_weights, opt_state, epoch, loss


def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False


def unfreeze_model(model):
    model.train()
    for params in model.parameters():
        params.requires_grad = True


def init_weight(model_layer, gain=1.0, sigma=0.02):  # TODO add initialization with different properties
    #  according to authors code
    classname = model_layer.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('LeakyConv') == -1:
        nn.init.orthogonal_(model_layer.weight.data, gain=gain)
    elif classname.find('BatchNorm') != -1:
        model_layer.weight.data.normal_(1.0, sigma)
        model_layer.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(model_layer.weight.data, 1.0)
        if model_layer.bias is not None:
            model_layer.bias.data.fill_(0.0)


def save_images(images, filenames, save_dir, iter, size):
    num_images = images.size(0)
    folder = os.path.join(save_dir, 'images', 'iter'+str(iter), str(size))
    os.makedirs(folder, exist_ok=True)

    if filenames is None:
        filenames = [str(i) for i in range(num_images)]
    # [-1, 1] --> [0, 1]
    img_tensor = images.add(1).div(2).detach().cpu()

    for i in range(num_images):
        im = images[i].detach().cpu()
        fullpath = os.path.join(folder, filenames[i]+'.jpg')
        # [-1, 1] --> [0, 1]
        img = im.add(1).div(2).mul(255).clamp(0, 255).byte()
        # [0, 1] --> [0, 255]
        ndarr = img.permute(1, 2, 0).data.cpu().numpy()
        im = Image.fromarray(ndarr)
        im.save(fullpath)
    
    return img_tensor


def copy_params(net):
    copy_params = deepcopy(list(p.data for p in net.parameters()))
    return copy_params


def load_params(net, new_param):
    for p, new_p in zip(net.parameters(), new_param):
        p.data.copy_(new_p)


def set_requires_grad_value(models_list, require_grad):
    for i in range(len(models_list)):
        for p in models_list[i].parameters():
            p.requires_grad = require_grad


def get_top_bottom_mean_grad(params):
    # First  layer
    mean_grad_first = next(params).grad.mean().item()
    # Last layer
    for p in params:
        pass
    mean_grad_last = p.grad.mean().item()

    return mean_grad_first, mean_grad_last