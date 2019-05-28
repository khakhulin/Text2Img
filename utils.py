import os
from copy import deepcopy

import torch
import torch.nn as nn
from nn_utils import LeakyConv

import numpy as np

from PIL import Image, ImageDraw, ImageFont
import skimage.transform


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


COLOR_DIC = {0:[128,64,128],  1:[244, 35,232],
             2:[70, 70, 70],  3:[102,102,156],
             4:[190,153,153], 5:[153,153,153],
             6:[250,170, 30], 7:[220, 220, 0],
             8:[107,142, 35], 9:[152,251,152],
             10:[70,130,180], 11:[220,20, 60],
             12:[255, 0, 0],  13:[0, 0, 142],
             14:[119,11, 32], 15:[0, 60,100],
             16:[0, 80, 100], 17:[0, 0, 230],
             18:[0,  0, 70],  19:[0, 0,  0]}
FONT_MAX = 50


def draw_attentions(real_imgs, captions, cap_lens, ixtoword,
                        attn_maps, att_sze, vis_size=256, topK=5):
    batch_size = real_imgs.size(0)
    max_word_num = np.max(cap_lens)
    text_convas = np.ones([batch_size * FONT_MAX,
                           max_word_num * (vis_size + 2), 3],
                           dtype=np.uint8)

    real_imgs = nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)

    w = real_imgs.size(3)
    middle_pad = np.zeros([w, 2, 3])

    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    img_set = []
    num = len(attn_maps)

    text_map, sentences = drawCaption(text_convas, captions, ixtoword, vis_size, off1=0)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)

        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = cap_lens[i]
        thresh = 2./float(num_attn)

        img = real_imgs[i]
        row = []
        row_merge = []
        row_txt = []
        row_beforeNorm = []
        conf_score = []
        for j in range(num_attn):
            one_map = attn[j]
            mask0 = one_map > (2. * thresh)
            conf_score.append(np.sum(one_map * mask0))
            mask = one_map > thresh
            one_map = one_map * mask
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze)
            minV = one_map.min()
            maxV = one_map.max()
            one_map = (one_map - minV) / (maxV - minV)
            row_beforeNorm.append(one_map)
        sorted_indices = np.argsort(conf_score)[::-1]

        for j in range(num_attn):
            one_map = row_beforeNorm[j]
            one_map *= 255

            PIL_im = Image.fromarray(np.uint8(img))
            PIL_att = Image.fromarray(np.uint8(one_map))
            merged = \
                Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
            mask = Image.new('L', (vis_size, vis_size), (180))  # (210)
            merged.paste(PIL_im, (0, 0))
            merged.paste(PIL_att, (0, 0), mask)
            merged = np.array(merged)[:, :, :3]

            row.append(np.concatenate([one_map, middle_pad], 1))

            row_merge.append(np.concatenate([merged, middle_pad], 1))

            txt = text_map[i * FONT_MAX:(i + 1) * FONT_MAX,
                           j * (vis_size + 2):(j + 1) * (vis_size + 2), :]
            row_txt.append(txt)

        row_new = []
        row_merge_new = []
        txt_new = []
        for j in range(num_attn):
            idx = sorted_indices[j]
            row_new.append(row[idx])
            row_merge_new.append(row_merge[idx])
            txt_new.append(row_txt[idx])
        row = np.concatenate(row_new[:topK], 1)
        row_merge = np.concatenate(row_merge_new[:topK], 1)
        txt = np.concatenate(txt_new[:topK], 1)
        if txt.shape[1] != row.shape[1]:
            print('Warnings: txt', txt.shape, 'row', row.shape,
                  'row_merge_new', row_merge_new.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None, None


def drawCaption(convas, captions, ixtoword, vis_size, off1=2, off2=2):
    num = captions.size(0)
    img_txt = Image.fromarray(convas)

    #fnt = ImageFont.truetype('FreeMono.ttf', 50)
    fnt = ImageFont.truetype('Keyboard.ttf', 30)

    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    for i in range(num):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
            d.text(((j + off1) * (vis_size + off2), i * FONT_MAX), '%d:%s' % (j, word[:6]),
                   font=fnt, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list