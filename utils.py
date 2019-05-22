import torch

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