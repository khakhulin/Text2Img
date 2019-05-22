import os
import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from arguments import init_config
from custom_inception_v3 import custom_inception_v3
from losses import func_attention, cosine_similarity, sent_loss, words_loss
from data_utils import BirdsPreprocessor, BirdsDataset, CaptionTokenizer
from utils import save, load, freeze_model

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
UPDATE_INTERVAL = 4

class TextEncoder(nn.Module):

    def __init__(self, n_tokens, emb_size=256, hid_size=128,
                 n_layers=1, bidirectional=True):
        super(TextEncoder, self).__init__()
        self.hid_size = hid_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        if bidirectional:
            self.n_directions = 2
        else:
            self.n_directions = 1
        
        self.emb = nn.Embedding(n_tokens, emb_size)
        self.drop = nn.Dropout(0.5)
        self.lstm = nn.LSTM(emb_size, hid_size, n_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.init_weights()
    
    def forward(self, cap, cap_len, hidden):
        h = self.drop(self.emb(cap))
        h = pack_padded_sequence(h, cap_len, batch_first=True,
                                 enforce_sorted=False)

        #print(h.size())
        words, h = self.lstm(h, hidden) # B x T x D
        words, _ = pad_packed_sequence(words, batch_first=True)
        #print(words.size())
        sent = h[0].transpose(0, 1).contiguous()
        #print(words.size(), sentence.size())
        sent = sent.view(-1, self.hid_size * self.n_directions)
        
        return words, sent

    def init_hidden(self, batch_size):
        n_hid = self.n_layers * self.n_directions
        h0 = torch.randn(n_hid, batch_size, self.hid_size)
        c0 = torch.randn(n_hid, batch_size, self.hid_size)
        return h0, c0
    
    def init_weights(self):
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
    
    def feat_size(self):
        size = self.hid_size
        if self.bidirectional:
            size *= 2
        return size


class ImageEncoder(nn.Module):

    def __init__(self, multimodal_feat_size=256):
        super(ImageEncoder, self).__init__()
        self.inception = custom_inception_v3()
        freeze_model(self.inception)

        self.map_global = nn.Linear(2048, multimodal_feat_size)
        self.map_local = nn.Linear(768, multimodal_feat_size)
    
    def forward(self, x):
        x, global_feat, _ = self.inception(x)
        # B x 768 x 17 x 17 --> B x 17 x 17 x 768
        # swap channels and sub-regions to apply linear transformation
        local_feat = self.map_local(x.permute(0, 2, 3, 1))
        global_feat = self.map_global(global_feat)
        #print(local_feat.size(), global_feat.size())

        return local_feat, global_feat
    
    def init_trainable_weights(self):
        initrange = 0.1
        self.map_global.weight.data.uniform_(-initrange, initrange)
        self.map_local.weight.data.uniform_(-initrange, initrange)


class DAMSM(nn.Module):
    
    def __init__(self, text_encoder, image_encoder):
        super(DAMSM, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
    
    def forward(self, imgs, caps, caps_len, hidden, args):
        # Bx(HxW)xD, BxD
        img_f_w, img_f_s = self.image_encoder(imgs)
        # BxTxD, BxD
        text_f_w, text_f_s = self.text_encoder(caps, caps_len, hidden)
        s_loss0, s_loss1 = sent_loss(img_f_s, text_f_s, args)
        w_loss0, w_loss1, _ = words_loss(img_f_w, text_f_w, caps_len, args)

        return w_loss0, w_loss1, s_loss0, s_loss1

    def train_epoch(self, epoch, dataloader, optimizer, image_dir,
                    args, device='cpu'):
        self.train()
        
        s_total_loss0 = 0
        s_total_loss1 = 0
        w_total_loss0 = 0
        w_total_loss1 = 0

        for data in tqdm(dataloader, total=len(dataloader)):

            imgs, caps, caps_len = data
            imgs = imgs[-1].to(device)
            caps = caps.to(device)
            caps_len = caps_len.to(device)
            # Initialize hidden state for LSTM
            h0, c0 = self.text_encoder.init_hidden(imgs.size(0))
            h0, c0 = h0.to(device), c0.to(device)
            w_loss0, w_loss1, s_loss0, s_loss1 = \
                self.forward(imgs, caps, caps_len, (h0, c0), args)
            loss = s_loss0 + s_loss1 + w_loss0 + w_loss1

            w_total_loss0 += w_loss0.item()
            w_total_loss1 += w_loss1.item()
            s_total_loss0 += s_loss0.item()
            s_total_loss1 += s_loss1.item()

            self.text_encoder.zero_grad()
            self.image_encoder.zero_grad()

            loss.backward()
            # `clip_grad_norm` helps prevent
            # the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(),
                                           args.damsm_rnn_grad_clip)
            optimizer.step()

        s_total_loss0 /= len(dataloader)
        s_total_loss1 /= len(dataloader)
        w_total_loss0 /= len(dataloader)
        w_total_loss1 /= len(dataloader)
        sumloss = s_total_loss0 + s_total_loss1 + w_total_loss0 + w_total_loss1

        print('[TRAIN] Epoch {:3d} | '
            's_loss {:5.2f} {:5.2f} | '
            'w_loss {:5.2f} {:5.2f} | Sum {:5.2f}'
            .format(epoch,
                    s_total_loss0, s_total_loss1,
                    w_total_loss0, w_total_loss1,
                    sumloss))

        return
    
    def evaluate(self, epoch, loader, image_dir, args, device='cpu'):
        self.eval()
        
        s_total_loss0 = 0
        s_total_loss1 = 0
        w_total_loss0 = 0
        w_total_loss1 = 0

        with torch.no_grad():
            for data in tqdm(loader, total=len(loader)):

                imgs, caps, caps_len = data
                imgs = imgs[-1].to(device)
                caps = caps.to(device)
                caps_len = caps_len.to(device)

                h0, c0 = self.text_encoder.init_hidden(imgs.size(0))
                h0, c0 = h0.to(device), c0.to(device)
                w_loss0, w_loss1, s_loss0, s_loss1 = \
                    self.forward(imgs, caps, caps_len, (h0, c0), args)
                # loss = w_loss0 + w_loss1 + s_loss0 + s_loss1

                w_total_loss0 += w_loss0.item()
                w_total_loss1 += w_loss1.item()
                s_total_loss0 += s_loss0.item()
                s_total_loss1 += s_loss1.item()

        s_cur_loss0 = s_total_loss0 / len(loader)
        s_cur_loss1 = s_total_loss1 / len(loader)
        w_cur_loss0 = w_total_loss0 / len(loader)
        w_cur_loss1 = w_total_loss1 / len(loader)

        sum_loss = s_cur_loss0 + s_cur_loss1 + w_cur_loss0 + w_cur_loss1

        print('[VALID] Epoch {:3d} | s_loss {:5.2f} {:5.2f} | '
              'w_loss {:5.2f} {:5.2f} | Sum {:5.2f}'
              .format(epoch, s_cur_loss0, s_cur_loss1,
                             w_cur_loss0, w_cur_loss1, sum_loss))

        return sum_loss

if __name__ == '__main__':
    args = init_config()
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    # Load data (Birds)
    preproc = BirdsPreprocessor(data_path='datasets/CUB_200_2011',
        dataset_name='cub'
    )
    tokenizer = CaptionTokenizer(word_to_idx=preproc.word_to_idx)

    n_tokens = len(preproc.vocabs['idx_to_word'])

    train_data = BirdsDataset(mode='train', tokenizer=tokenizer,
        preprocessor=preproc, base_size=299, branch_num=1
    )
    val_data = BirdsDataset(mode='val', tokenizer=tokenizer,
        preprocessor=preproc, base_size=299, branch_num=1
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.damsm_batch_size,
        shuffle=True, num_workers=6
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=args.damsm_batch_size,
        shuffle=True, num_workers=6
    )
    # CUDA
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = 'cpu'
    # Create model and optimizer
    text_encoder = TextEncoder(n_tokens, args.embd_size)
    image_encoder = ImageEncoder(text_encoder.feat_size())
    damsm = DAMSM(text_encoder, image_encoder).to(device)
    optimizer = torch.optim.Adam(damsm.parameters(),
        lr=args.damsm_lr, betas=(0.5, 0.999)
    )    
    # Train
    image_dir = 'attn_images'
    save_dir = os.path.join('pretrained/DAMSM', run_name)
    os.makedirs(save_dir, exist_ok=True)

    min_loss = np.inf
    timer = 0
    start_epoch = 0
    # Continue training
    if len(args.damsm_start_from):
        weights, opt, loss, epoch = load(args.damsm_start_from)
        damsm.load_state_dict(weights)
        optimizer.load_state_dict(opt)
        min_loss = loss
        start_epoch = int(epoch) + 1
    # Main loop
    for epoch in range(start_epoch, start_epoch + args.damsm_n_epoch):
        damsm.train_epoch(epoch, train_loader, optimizer, image_dir,
            args, device
        )
        loss = damsm.evaluate(epoch, val_loader, image_dir, args, device)
        # # Save best model
        if loss < min_loss:
            min_loss = loss
            best_path = os.path.join(save_dir, 'best.pt')
            save(best_path, damsm, optimizer, loss, epoch)
        # Save checkpoint
        timer += 1

        if timer == args.damsm_snapshot_interval:
            timer = 0
            ckpt_path = os.path.join(save_dir, 'weights%03d.pt' % (epoch+1))
            save(ckpt_path, damsm, optimizer, loss, epoch)
