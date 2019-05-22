import torch
import torch.nn as nn

from nn_utils import LeakyConv


class Downsample16(nn.Sequential):
    def __init__(self, num_out):
        super().__init__(nn.Sequential(
        nn.Conv2d(3, num_out, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        LeakyConv(num_out, num_out * 2, 4, 2, 1),
        LeakyConv(num_out * 2, num_out * 4, 4, 2, 1),
        LeakyConv(num_out * 4, num_out * 8, 4, 2, 1)
        )
    )


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

