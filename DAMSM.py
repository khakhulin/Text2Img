import datetime
import os

import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import torchvision.transforms as transforms

from arguments import init_config
from custom_inception_v3 import custom_inception_v3
from data_utils import BirdsPreprocessor, BirdsDataset, CaptionTokenizer, BertCaptionTokenizer, prepare_data, \
    CocoPreprocessor, CocoDataset
from modules.losses import sent_loss, words_loss
from utils import freeze_model


class TextEncoder(nn.Module):

    def __init__(self, n_tokens, emb_size=256, text_feat_size=128,
                 n_layers=1, bidirectional=True):
        super(TextEncoder, self).__init__()
        self.hid_size = text_feat_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        if bidirectional:
            assert self.hid_size % 2 == 0, \
                'text_feat_size must be devisible by 2 if bidirectional=True'
            self.n_directions = 2
        else:
            self.n_directions = 1
        
        self.emb = nn.Embedding(n_tokens, emb_size)
        self.drop = nn.Dropout(0.5)
        self.lstm = nn.LSTM(emb_size, self.hid_size//2, n_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.init_weights()
    
    def forward(self, cap, cap_len):
        bs = cap.size(0)
        h = self.drop(self.emb(cap))
        h = pack_padded_sequence(h, cap_len, batch_first=True)
        #print(h.size())
        # initialize hidden state
        h0 = torch.randn(
            self.n_layers*self.n_directions, bs, self.hid_size//2
        ).to(next(iter(self.parameters())).device)
        c0 = torch.randn(
            self.n_layers*self.n_directions, bs, self.hid_size//2
        ).to(next(iter(self.parameters())).device)

        words, h = self.lstm(h, (h0, c0)) # B x T x D
        words, _ = pad_packed_sequence(words, batch_first=True)
        #print(words.size())
        sent = h[0].transpose(0, 1).contiguous()
        #print(words.size(), sentence.size())
        sent = sent.view(-1, self.hid_size)
        
        return words, sent
    
    def init_weights(self):
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)


class BertEncoder(nn.Module):

    def __init__(self, emb_size=128,
                 n_layers=1):
        super(BertEncoder, self).__init__()
        self.hid_size = emb_size
        self.n_layers = n_layers
        self.inp_ch = 30
        # hidden size per each encoding vector
        self.enc_size = emb_size
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters():
            param.requires_grad = False

        self.word_embeddings = nn.Sequential(
            nn.Linear(768*4, 768*2),
            nn.Linear(768*2, self.enc_size)
        )
        self.sent_embeddings = nn.Conv1d(in_channels=self.inp_ch, out_channels=1, kernel_size=1)
        self.init_weights()

    def forward(self, captions, cap_len, input_mask):
        # captions = pack_padded_sequence(captions, cap_len, batch_first=True,
        #                          enforce_sorted=False)
        batch_size, seq_len = captions.size(0), captions.size(1)
        #print('Load pretrained BERT')
        all_encoder_layers, _ = self.bert(captions, token_type_ids=None, attention_mask=input_mask)

        words_features = torch.cat(all_encoder_layers[-5:-1], dim=-1).view(-1, 768*4)
        words_emb_out = self.word_embeddings(words_features).view(batch_size, seq_len, -1)
        words_emb = words_emb_out#.transpose(1, 2)

        sent_emb = self.sent_embeddings(words_emb_out).squeeze()
        return words_emb, sent_emb

    def init_weights(self):
        initrange = 0.1
        for m in self.word_embeddings:
            nn.init.orthogonal_(m.weight.data, 1.0)
        self.sent_embeddings.weight.data.uniform_(-initrange, initrange)


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
    
    def __init__(self, text_encoder, image_encoder, is_bert):
        super(DAMSM, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.is_bert = is_bert

    def forward(self, imgs, caps, caps_len, args, class_ids=None, bert_mask=None):
        # Bx(HxW)xD, BxD
        img_f_w, img_f_s = self.image_encoder(imgs)
        # BxTxD, BxD
        if self.is_bert:
            text_f_w, text_f_s = self.text_encoder(caps, caps_len, bert_mask)
        else:
            text_f_w, text_f_s = self.text_encoder(caps, caps_len)

        s_loss0, s_loss1 = sent_loss(
            img_f_s, text_f_s, args, class_ids=class_ids
        )
        w_loss0, w_loss1, _ = words_loss(
            img_f_w, text_f_w, caps_len, args, class_ids=class_ids
        )

        return w_loss0, w_loss1, s_loss0, s_loss1

    def train_epoch(self, epoch, dataloader, optimizer, image_dir,
                    args, device='cpu'):
        self.train()
        
        s_total_loss0 = 0
        s_total_loss1 = 0
        w_total_loss0 = 0
        w_total_loss1 = 0

        for data in tqdm(dataloader, total=len(dataloader)):

            imgs, caps, caps_len, masks, class_ids = \
                prepare_data(data, device, is_damsm=True)

            if self.is_bert:
                w_loss0, w_loss1, s_loss0, s_loss1 = \
                    self.forward(
                        imgs, caps, caps_len, args,
                        class_ids=class_ids, bert_mask=masks
                    )
            else:
                w_loss0, w_loss1, s_loss0, s_loss1 = \
                    self.forward(
                        imgs, caps, caps_len, args, class_ids=class_ids
                    )

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
            for data in loader:

                imgs, caps, caps_len, masks, class_ids = \
                    prepare_data(data, device, is_damsm=True)

                if self.is_bert:
                    w_loss0, w_loss1, s_loss0, s_loss1 = \
                        self.forward(
                            imgs, caps, caps_len, args,
                            class_ids=class_ids, bert_mask=masks
                        )
                else:
                    w_loss0, w_loss1, s_loss0, s_loss1 = \
                        self.forward(
                            imgs, caps, caps_len, args,
                            class_ids=class_ids
                        )
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
    run_name = datetime.datetime.now().strftime('%d:%m:%Y:%H-%M-%S')
    # Load data (Birds)
    if args.datasets == 'birds':
        preproc = BirdsPreprocessor(data_path=args.data_path,
            dataset_name='cub'
        )
    else:
        preproc = CocoPreprocessor(data_path=args.data_path, dataset_name='coco')

    if args.is_bert:
        tokenizer = BertCaptionTokenizer(word_to_idx=preproc.word_to_idx)
    else:
        tokenizer = CaptionTokenizer(word_to_idx=preproc.word_to_idx)

    n_tokens = len(preproc.vocabs['idx_to_word'])
    imsize = 299

    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 86 / 64)),
        transforms.CenterCrop(imsize),
        transforms.RandomHorizontalFlip()
    ])
    if args.datasets == 'birds':
        train_data = BirdsDataset(
            mode='train', tokenizer=tokenizer,
            preprocessor=preproc, base_size=imsize, branch_num=1,
            transform=image_transform
        )
        val_data = BirdsDataset(
            mode='val', tokenizer=tokenizer,
            preprocessor=preproc, base_size=imsize, branch_num=1
        )
    else:
        train_data = CocoDataset(
            mode='train', tokenizer=tokenizer,
            preprocessor=preproc, base_size=imsize, branch_num=1,
            transform=image_transform
        )
        val_data = CocoDataset(
            mode='val', tokenizer=tokenizer,
            preprocessor=preproc, base_size=imsize, branch_num=1
        )
        
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, drop_last=True,
        batch_size=args.damsm_batch_size,
        shuffle=True, num_workers=6
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_data, drop_last=True,
        batch_size=args.damsm_batch_size,
        shuffle=True, num_workers=6
    )
    # CUDA
    if args.cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.cuda_device}')
    else:
        device = 'cpu'
    # Create model and optimizer
    print("Embdding dim", args.embd_size)
    if args.is_bert:
        text_encoder = BertEncoder(emb_size=args.embd_size)
    else:
        text_encoder = TextEncoder(
            n_tokens=n_tokens, emb_size=args.text_enc_emb_size,
            text_feat_size=args.embd_size
        )
    image_encoder = ImageEncoder(args.embd_size)
    damsm = DAMSM(text_encoder, image_encoder, is_bert=args.is_bert).to(device)
    optimizer = torch.optim.Adam(
        damsm.parameters(),
        lr=args.damsm_lr, betas=(0.5, 0.999),
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.8, verbose=True
    )
    # Train
    image_dir = 'attn_images'
    save_dir = os.path.join('trained_models/DAMSM', run_name)
    os.makedirs(save_dir, exist_ok=True)

    min_loss = np.inf
    timer = 0
    start_epoch = 0
    # Continue training
    if len(args.damsm_text_encoder):
        weights = torch.load(args.damsm_text_encoder)
        damsm.text_encoder.load_state_dict(weights)

    if len(args.damsm_image_encoder):
        weights = torch.load(args.damsm_image_encoder)
        damsm.image_encoder.load_state_dict(weights)
    # Main loop
    for epoch in range(start_epoch, start_epoch + args.damsm_n_epoch):
        damsm.train_epoch(
            epoch, train_loader, optimizer, image_dir,
            args, device
        )
        loss = damsm.evaluate(epoch, val_loader, image_dir, args, device)
        scheduler.step(loss)
        # Save best model
        if loss < min_loss:
            min_loss = loss
            torch.save(
                damsm.image_encoder.state_dict(),
                os.path.join(save_dir, 'best_image_encoder.pt')
            )
            torch.save(
                damsm.text_encoder.state_dict(),
                os.path.join(save_dir, 'best_text_encoder.pt')
            )
        # Save checkpoint
        timer += 1

        if timer == args.damsm_snapshot_interval:
            timer = 0
            torch.save(
                damsm.image_encoder.state_dict(),
                os.path.join(save_dir, 'image_encoder%03d.pt' % (epoch+1))
            )
            torch.save(
                damsm.text_encoder.state_dict(),
                os.path.join(save_dir, 'text_encoder%03d.pt' % (epoch+1))
            )